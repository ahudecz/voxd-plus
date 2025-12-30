# voxd-plus: Claude Code Handoff

## Project Summary

Forking [jakovius/voxd](https://github.com/jakovius/voxd) to create **voxd-plus** - an enhanced speech-to-text dictation app for Linux with audio cues, waveform visualization, and expanded LLM provider support.

## What's Already Done

I've created a recording overlay module with:
- ✅ Audio cues (start/stop/error tones via numpy+sounddevice)
- ✅ Live waveform visualization (PyQt6 custom widget)
- ✅ Recording time indicator (MM:SS.ss format)
- ✅ Level meter with color gradient
- ✅ Draggable, always-on-top overlay window
- ✅ Socket-based IPC for hotkey triggering

Files are in the uploaded `voxd_overlay.zip` or can be recreated from our conversation.

---

## Refined Roadmap

### Phase 1: Foundation
**Merge community fixes from voxd PRs:**

```bash
# PRs to cherry-pick/merge:
#15 - Remove hard coded python versions
#18 - Fix invalid version string (semver)
#22 - Fix 285-character truncation in ydotool
#25 - Dark mode settings background fix
#26 - Make copy to clipboard configurable
#27 - Add OPENAI_API_BASE env variable (enables LM Studio, etc)
#23 - Add GUI settings for llama.cpp paths
#16 - GPU Acceleration Support (evaluate carefully)
#31 - Streaming transcription (evaluate - just submitted)
```

**Bugs to fix:**
- #24 - GPU detection/fallback
- #19 - Keyboard layout issues with ydotool

### Phase 2: UX Enhancements
**Integrate our overlay module:**
- Audio cues on record start/stop
- Waveform overlay during recording  
- Recording time display
- Level meter/clipping indicator
- Overlay position persistence
- Optional: customizable cue sounds

### Phase 3: LLM Provider Expansion
**Current voxd providers (keep all):**
- llama.cpp (local)
- Ollama (local)
- OpenAI
- Anthropic
- xAI

**Add these:**
- Google Gemini API
- Groq (ultra-fast inference)
- OpenRouter (access to many models)
- LM Studio API (local, OpenAI-compatible) - PR #27 helps!

### Phase 4: Settings Overhaul
- Unified settings UI
- Audio device selection with preview
- Microphone input level visualization
- Per-provider prompt templates
- Hotkey configuration with conflict detection
- VAD sensitivity tuning
- Export/import settings

---

## Key Files Structure

```
voxd-plus/
├── src/voxd/
│   ├── __init__.py
│   ├── main.py           # Entry point
│   ├── cli.py            # CLI interface
│   ├── gui.py            # PyQt6 GUI
│   ├── tray.py           # System tray
│   ├── recorder.py       # Audio recording
│   ├── transcriber.py    # Whisper integration
│   ├── aipp.py           # AI post-processing
│   ├── config.py         # Configuration
│   ├── typing_output.py  # ydotool/xdotool
│   │
│   ├── overlay/          # NEW - our additions
│   │   ├── __init__.py
│   │   ├── recording_overlay.py
│   │   ├── waveform_widget.py
│   │   ├── audio_cues.py
│   │   └── trigger_server.py
│   │
│   ├── providers/        # NEW - expanded LLM providers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llamacpp.py
│   │   ├── ollama.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── xai.py
│   │   ├── gemini.py     # NEW
│   │   ├── groq.py       # NEW
│   │   └── openrouter.py # NEW
│   │
│   └── assets/
│       └── sounds/       # Optional custom cue sounds
│
├── packaging/
├── tests/
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

---

## Commands to Get Started

```bash
# 1. Fork on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/voxd-plus.git
cd voxd-plus

# 2. Add upstream remote
git remote add upstream https://github.com/jakovius/voxd.git
git fetch upstream

# 3. Create enhancement branch
git checkout -b enhance/voxd-plus

# 4. Cherry-pick the PRs (get commit SHAs from GitHub)
# Or merge PR branches if available

# 5. Set up dev environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## PR URLs for Reference

- #15: https://github.com/jakovius/voxd/pull/15
- #16: https://github.com/jakovius/voxd/pull/16
- #18: https://github.com/jakovius/voxd/pull/18
- #22: https://github.com/jakovius/voxd/pull/22
- #23: https://github.com/jakovius/voxd/pull/23
- #25: https://github.com/jakovius/voxd/pull/25
- #26: https://github.com/jakovius/voxd/pull/26
- #27: https://github.com/jakovius/voxd/pull/27
- #31: https://github.com/jakovius/voxd/pull/31

---

## LLM Provider API Reference

### Gemini
```python
# pip install google-generativeai
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
```

### Groq
```python
# pip install groq
from groq import Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
)
```

### OpenRouter
```python
# OpenAI-compatible API
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": prompt}]
)
```

### LM Studio (local)
```python
# OpenAI-compatible, use OPENAI_API_BASE
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)
```

---

## Audio Cue Implementation (from our overlay)

```python
class AudioCue:
    SAMPLE_RATE = 44100
    
    @staticmethod
    def generate_tone(freq: float, duration: float, volume: float = 0.25) -> np.ndarray:
        samples = int(AudioCue.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        tone = np.sin(2 * np.pi * freq * t) * volume
        # Fade in/out to avoid clicks
        fade = int(AudioCue.SAMPLE_RATE * 0.01)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        return tone
    
    @staticmethod
    def play_start():
        # Ascending two-tone: A5 -> E6
        t1 = AudioCue.generate_tone(880, 0.08)
        t2 = AudioCue.generate_tone(1320, 0.12)
        gap = np.zeros(int(44100 * 0.02), dtype=np.float32)
        sd.play(np.concatenate([t1, gap, t2]), 44100)
    
    @staticmethod
    def play_stop():
        # Descending: E6 -> A5
        t1 = AudioCue.generate_tone(1320, 0.08)
        t2 = AudioCue.generate_tone(880, 0.12)
        gap = np.zeros(int(44100 * 0.02), dtype=np.float32)
        sd.play(np.concatenate([t1, gap, t2]), 44100)
```

---

## Testing Checklist

```
□ Fresh install on Ubuntu 24.04
□ Hotkey recording works
□ Audio cues play on start/stop
□ Waveform displays correctly
□ Transcription completes
□ Text types into target app (Wayland + X11)
□ No 285 char truncation
□ GPU mode works (if available)
□ CPU fallback works (if no GPU)
□ All LLM providers connect
□ Settings persist across restarts
□ Overlay position saves
```

---

## Context for Claude Code

Ryan is the Principal Web Developer & Technical Architect at Yeeboo Digital, a nonprofit digital agency. He uses voxd for dictation when interacting with LLMs. His main pain points with voxd:

1. No audio feedback when recording starts/stops
2. No visual indicator during recording (waveform, timer)
3. Limited LLM provider options

He wants a polished fork that addresses these and incorporates community fixes that are languishing in PRs.

Hardware context: Ryzen 9 9900X3D, 128GB RAM, dual RTX 5060 Ti - so GPU acceleration is relevant.

---

## First Task for Claude Code

1. Help Ryan fork voxd on GitHub
2. Clone the fork locally
3. Set up the dev environment
4. Start evaluating/merging the priority PRs
5. Create the overlay module integration plan

---

*Handoff created: Dec 30, 2025*
