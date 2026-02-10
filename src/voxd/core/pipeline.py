"""Multi-pass transcript post-processing pipeline.

Replaces the single-shot AIPP with a chain of discrete passes:

  Pass 1 — CLEAN:  regex filler removal + optional small LLM disfluency repair
  Pass 2 — GRAMMAR: punctuation, capitalization, sentence structure
  Pass 3 — FORMAT:  context-aware formatting using the app profile from app_detect

Each pass can be independently enabled/disabled and configured with its own
provider + model.  The pipeline is a drop-in replacement for ``get_final_text``
when ``pipeline_enabled: true`` in config.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from voxd.core.aipp import run_aipp
from voxd.utils.libw import verbo, verr


# ---------------------------------------------------------------------------
# Pass 1: Filler / disfluency removal (regex + optional LLM)
# ---------------------------------------------------------------------------

# Common English fillers, hedges, and restart patterns
_FILLER_PATTERN = re.compile(
    r"""(?xi)                  # verbose, case-insensitive
    # Standalone fillers (word boundary)
    \b(?:
        u[hm]+|er+|ah+|
        like,?\s*|
        you\s+know,?\s*|
        basically,?\s*|
        i\s+mean,?\s*|
        sort\s+of,?\s*|
        kind\s+of,?\s*|
        right,?\s*(?=so|and|but|um|uh)|
        well,?\s*(?=so|and|but|um|uh)|
        okay\s+so,?\s*|
        so,?\s*(?=so|um|uh|like)
    )
    |
    # Mid-sentence corrections: "at 2... actually 3" → keep "3"
    (?:\.{2,}|—|–)\s*(?:actually|wait|no(?:,)?|I\s+mean)\s+
    """,
)

# Repeated words: "the the", "I I"
_REPEATED_WORD = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

# Collapse multiple spaces
_MULTI_SPACE = re.compile(r'[ \t]{2,}')


def _clean_pass(text: str, context: dict) -> str:
    """Pass 1: strip fillers and disfluencies."""
    original_len = len(text)

    # Stage A: regex
    cleaned = _FILLER_PATTERN.sub(' ', text)
    cleaned = _REPEATED_WORD.sub(r'\1', cleaned)
    cleaned = _MULTI_SPACE.sub(' ', cleaned).strip()

    # Stage B: if regex changed >15% of text, run a small LLM for cleanup
    clean_cfg = context.get("clean", {})
    llm_enabled = clean_cfg.get("enabled", True) and clean_cfg.get("llm_repair", True)

    if llm_enabled and original_len > 0 and len(cleaned) < original_len * 0.85:
        try:
            prompt = (
                "Fix any remaining disfluencies in the following text. "
                "Keep the meaning exactly intact. Output ONLY the cleaned text, "
                "nothing else:\n\n" + cleaned
            )
            repaired = _run_pipeline_llm(prompt, clean_cfg, context)
            if repaired and len(repaired) > 5:
                cleaned = repaired.strip()
        except Exception as e:
            verr(f"[pipeline/clean] LLM repair failed: {e}")

    verbo(f"[pipeline/clean] {original_len} → {len(cleaned)} chars")
    return cleaned


# ---------------------------------------------------------------------------
# Pass 2: Grammar / punctuation
# ---------------------------------------------------------------------------

def _grammar_pass(text: str, context: dict) -> str:
    """Pass 2: fix grammar, punctuation, capitalization."""
    grammar_cfg = context.get("grammar", {})
    if not grammar_cfg.get("enabled", True):
        return text

    prompt = grammar_cfg.get(
        "prompt",
        "Fix grammar, add correct punctuation and capitalization. "
        "Preserve the original meaning exactly. "
        "Output ONLY the corrected text, nothing else."
    )
    full_prompt = f"{prompt}\n\n{text}"

    try:
        result = _run_pipeline_llm(full_prompt, grammar_cfg, context)
        if result and len(result) > 3:
            verbo(f"[pipeline/grammar] {len(text)} → {len(result)} chars")
            return result.strip()
    except Exception as e:
        verr(f"[pipeline/grammar] LLM failed: {e}")

    return text


# ---------------------------------------------------------------------------
# Pass 3: Context-aware formatting
# ---------------------------------------------------------------------------

def _format_pass(text: str, context: dict) -> str:
    """Pass 3: adapt text to the target application's style."""
    format_cfg = context.get("format", {})
    if not format_cfg.get("enabled", True):
        return text

    # Get the app profile prompt (injected by app_detect)
    profile_prompt = context.get("profile_prompt", "")
    if not profile_prompt:
        return text  # No app context — skip

    full_prompt = (
        f"{profile_prompt}\n\n"
        f"Rewrite the following text accordingly. "
        f"Output ONLY the result, nothing else:\n\n{text}"
    )

    try:
        result = _run_pipeline_llm(full_prompt, format_cfg, context)
        if result and len(result) > 3:
            verbo(f"[pipeline/format] {len(text)} → {len(result)} chars "
                  f"(profile={context.get('profile', 'unknown')})")
            return result.strip()
    except Exception as e:
        verr(f"[pipeline/format] LLM failed: {e}")

    return text


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result of running the transcript pipeline."""
    raw_text: str
    final_text: str
    passes_applied: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class TranscriptPipeline:
    """Chain of text processing passes."""

    PASSES = [
        ("clean", _clean_pass),
        ("grammar", _grammar_pass),
        ("format", _format_pass),
    ]

    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self, raw_text: str, app_context: Optional[dict] = None) -> PipelineResult:
        """Run the full pipeline on raw transcript text.

        Args:
            raw_text: The raw Whisper transcript.
            app_context: Dict from ``app_detect.detect_focused_app()``.
                         If None, Pass 3 (format) will be a no-op.

        Returns:
            PipelineResult with the cleaned/formatted text.
        """
        if not raw_text or not raw_text.strip():
            return PipelineResult(raw_text=raw_text, final_text=raw_text)

        start = time.monotonic()
        pipeline_cfg = self.cfg.data.get("pipeline_passes", {})

        # Build context dict that each pass can read
        context: dict = {
            "cfg": self.cfg,
            "clean": pipeline_cfg.get("clean", {}),
            "grammar": pipeline_cfg.get("grammar", {}),
            "format": pipeline_cfg.get("format", {}),
        }
        # Merge app context
        if app_context:
            context.update(app_context)

        text = raw_text.strip()
        applied: list[str] = []

        for pass_name, pass_fn in self.PASSES:
            pass_cfg = context.get(pass_name, {})
            if not pass_cfg.get("enabled", True):
                verbo(f"[pipeline] Skipping disabled pass: {pass_name}")
                continue
            try:
                text = pass_fn(text, context)
                applied.append(pass_name)
            except Exception as e:
                verr(f"[pipeline] Pass '{pass_name}' failed: {e}")

        duration = (time.monotonic() - start) * 1000
        verbo(f"[pipeline] Done in {duration:.0f}ms, passes={applied}")

        return PipelineResult(
            raw_text=raw_text,
            final_text=text,
            passes_applied=applied,
            duration_ms=duration,
        )


# ---------------------------------------------------------------------------
# Helper: route LLM calls through AIPP provider infrastructure
# ---------------------------------------------------------------------------

def _run_pipeline_llm(prompt: str, pass_cfg: dict, context: dict) -> str:
    """Run a single LLM call using the AIPP provider infrastructure.

    Each pass can specify its own ``provider`` and ``model`` in config.
    Falls back to the global AIPP provider/model if not specified.
    """
    cfg = context.get("cfg")
    if cfg is None:
        from voxd.core.config import get_config
        cfg = get_config()

    # Per-pass provider/model override, or fall back to global AIPP settings
    provider = pass_cfg.get("provider") or cfg.data.get("aipp_provider", "llamacpp_server")
    model = pass_cfg.get("model") or cfg.get_aipp_selected_model(provider)

    # Build a temporary cfg-like object for run_aipp compatibility
    # We override the prompt to be our pipeline prompt, not the stored AIPP prompts
    from voxd.core.aipp import (
        run_ollama_aipp, run_llamacpp_server_aipp, run_openai_aipp,
        run_anthropic_aipp, run_xai_aipp, run_gemini_aipp,
        run_groq_aipp, run_openrouter_aipp, run_lmstudio_aipp,
    )

    provider_fns = {
        "ollama": run_ollama_aipp,
        "llamacpp_server": run_llamacpp_server_aipp,
        "openai": run_openai_aipp,
        "anthropic": run_anthropic_aipp,
        "xai": run_xai_aipp,
        "gemini": run_gemini_aipp,
        "groq": run_groq_aipp,
        "openrouter": run_openrouter_aipp,
        "lmstudio": run_lmstudio_aipp,
    }

    fn = provider_fns.get(provider)
    if fn is None:
        verr(f"[pipeline] Unknown provider: {provider}")
        return ""

    return fn(prompt, model)


# ---------------------------------------------------------------------------
# Public convenience: drop-in for get_final_text when pipeline is enabled
# ---------------------------------------------------------------------------

def pipeline_get_final_text(transcript: str, cfg, app_context: Optional[dict] = None) -> str:
    """Drop-in replacement for ``aipp.get_final_text`` that uses the pipeline.

    If ``pipeline_enabled`` is False in config, falls back to legacy
    single-shot AIPP via ``get_final_text``.
    """
    if not cfg.data.get("pipeline_enabled", False):
        from voxd.core.aipp import get_final_text
        return get_final_text(transcript, cfg)

    pipeline = TranscriptPipeline(cfg)
    result = pipeline.execute(transcript, app_context)
    return result.final_text
