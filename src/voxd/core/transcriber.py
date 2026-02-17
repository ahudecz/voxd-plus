import subprocess
import os
from pathlib import Path
import re
from voxd.utils.libw import verbo, verr
from voxd.paths import find_whisper_cli, find_base_model
from voxd.utils.languages import normalize_lang_code, is_valid_lang


class WhisperTranscriber:
    def __init__(self, model_path, binary_path, delete_input=True, language: str | None = None, cfg=None):
        # --- Model path: try config, else auto-discover ---
        if model_path and Path(model_path).is_file():
            self.model_path = model_path
        else:
            # Try to use the default model in cache
            self.model_path = find_base_model()
            verbo(f"[transcriber] Falling back to cached model: {self.model_path}")

        # --- Binary path: try config, else auto-discover ---
        if binary_path and Path(binary_path).is_file() and os.access(binary_path, os.X_OK):
            self.binary_path = binary_path
        else:
            self.binary_path = find_whisper_cli()
            verbo(f"[transcriber] Falling back to auto-detected whisper-cli: {self.binary_path}")

        self.delete_input = delete_input
        self.cfg = cfg
        from voxd.paths import OUTPUT_DIR
        self.output_dir = OUTPUT_DIR

        # Language (default en)
        lang = normalize_lang_code(language or "en")
        if not is_valid_lang(lang):
            verr(f"[transcriber] Invalid language '{language}', using 'en'")
            lang = "en"
        self.language = lang

        # Determine GPU device
        self.device = self._get_device()
        verbo(f"[transcriber] Using device: {self.device}")

        # Warn if likely mismatch with an English-only model
        try:
            mp = str(self.model_path).lower()
            if self.language != "en" and mp.endswith(".en.bin"):
                verr("[transcriber] Non-English language selected but an English-only (*.en) model is configured.")
        except Exception:
            pass

    def _get_device(self) -> str:
        """Determine the device to use for transcription."""
        try:
            from voxd.utils.gpu_detect import get_whisper_device_flag
            return get_whisper_device_flag(self.cfg)
        except ImportError:
            return "cpu"

    def transcribe(self, audio_path):
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"[transcriber] Audio file not found: {audio_file}")

        # Try whisper-server first (model stays in RAM = much faster)
        if self.cfg and self.cfg.data.get("whisper_server_enabled", True):
            result = self._transcribe_via_server(audio_file)
            if result is not None:
                return result

        # Output prefix (no extension!)
        output_prefix = self.output_dir / audio_file.stem
        output_txt = output_prefix.with_suffix(".txt")

        # Thread count: GPU offloads inference so CPU threads only handle
        # pre/post-processing — 4 is enough.  For CPU-only, use more.
        cpu_count = os.cpu_count() or 4
        if self.device == "cuda":
            n_threads = 4
        else:
            n_threads = min(12, max(4, cpu_count // 2))

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-f", str(audio_file),
            "-l", self.language,
            "-t", str(n_threads),
            "-np",   # suppress progress/timestamp prints
            "-of", str(self.output_dir / audio_file.stem),
            "-otxt",
        ]

        # Flash attention: major GPU speedup (2-3×)
        if self.device == "cuda":
            cmd.append("-fa")

        # Whisper vocabulary hints (--prompt)
        whisper_prompt = (self.cfg.data.get("whisper_prompt", "") if self.cfg else "").strip()
        if whisper_prompt:
            cmd.extend(["--prompt", whisper_prompt])

        # Beam search size (-bs)
        beam_size = int(self.cfg.data.get("whisper_beam_size", 5) if self.cfg else 5)
        if beam_size != 5:
            cmd.extend(["-bs", str(beam_size)])

        # GPU handling: new whisper.cpp has GPU on by default, use --no-gpu to disable
        if self.device == "cpu":
            cmd.append("--no-gpu")

        verbo(f"[transcriber] Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # If GPU failed (CUDA errors), try falling back to CPU
        gpu_error_indicators = ["cuda", "gpu", "device", "out of memory"]
        has_gpu_error = any(indicator in result.stderr.lower() for indicator in gpu_error_indicators) if result.stderr else False

        if result.returncode != 0 and self.device == "cuda" and has_gpu_error:
            verr("[transcriber] GPU transcription failed, falling back to CPU...")
            cmd_cpu = cmd + ["--no-gpu"]
            verbo(f"[transcriber] Retrying with CPU: {' '.join(cmd_cpu)}")
            result = subprocess.run(cmd_cpu, capture_output=True, text=True)

        if result.returncode != 0:
            verr("[transcriber] whisper.cpp failed:")
            verr(f"stderr: {result.stderr}")
            verr(f"stdout: {result.stdout}")
            return None, None

        if not output_txt.exists():
            verr(f"[transcriber] Transcription failed: Expected output not found at {output_txt}")
            return None, None

        verbo(f"[transcriber] Transcription complete: {output_txt}")

        # Optionally delete the input audio
        if self.delete_input:
            try:
                audio_file.unlink()
                verbo(f"[transcriber] Deleted input file: {audio_file}")
            except Exception as e:
                verr(f"[transcriber] Could not delete input file: {e}")

        return self._parse_transcript(output_txt)

    def _transcribe_via_server(self, audio_file: Path):
        """Try to transcribe via whisper-server HTTP API.

        Returns (text, original_text) tuple on success, or None to fall back
        to subprocess.
        """
        try:
            from voxd.core.whisper_server_manager import get_whisper_server_manager
            mgr = get_whisper_server_manager()

            # Only use server if process is alive
            if not mgr.is_process_alive():
                return None

            whisper_prompt = (self.cfg.data.get("whisper_prompt", "") if self.cfg else "").strip()
            text = mgr.transcribe(str(audio_file), language=self.language, prompt=whisper_prompt)
            if text is None:
                return None

            verbo(f"[transcriber] Server transcription: '{text[:80]}...'")

            # Delete input if configured
            if self.delete_input:
                try:
                    audio_file.unlink()
                    verbo(f"[transcriber] Deleted input file: {audio_file}")
                except Exception as e:
                    verr(f"[transcriber] Could not delete input file: {e}")

            # Strip timestamps and normalize whitespace (same as _parse_transcript)
            tscript = re.sub(r"\[\d{2}:\d{2}[\.:]\d{3}\]|\(\d{2}:\d{2}\)", "", text)
            tscript = re.sub(r"\s+", " ", tscript).strip()

            return tscript, text

        except ImportError:
            return None
        except Exception as e:
            verr(f"[transcriber] Server transcription failed: {e}")
            return None

    def _parse_transcript(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[transcriber] Failed to read transcript file: {e}")
            return None, None

        orig_tscript = "".join(lines)

        # Strip timestamps like [00:00.000] or (00:00)
        tscript = re.sub(r"\[\d{2}:\d{2}[\.:]\d{3}\]|\(\d{2}:\d{2}\)", "", orig_tscript)
        tscript = re.sub(r"\s+", " ", tscript).strip()

        return tscript, orig_tscript
