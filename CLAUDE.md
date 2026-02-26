# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

voxd-plus is an enhanced fork of [jakovius/voxd](https://github.com/jakovius/voxd) - a voice-to-text dictation app for Linux. It uses whisper.cpp for local speech recognition and can type transcribed text directly into any application via ydotool (Wayland) or xdotool (X11).

Key enhancements over upstream voxd:
- Recording overlay with animated frequency visualizer
- Audio cues (start/stop/success tones)
- Additional LLM providers (Gemini, Groq, OpenRouter, LM Studio)
- GPU acceleration support with CUDA
- Streaming transcription mode
- Customizable audio cue sounds with volume control

## Common Commands

```bash
# Development setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the app in different modes
voxd-plus --gui          # GUI mode
voxd-plus --tray         # System tray mode (background)
voxd-plus --cli          # CLI mode (default)
voxd-plus --flux         # VAD-triggered mode

# Trigger recording (for hotkey integration)
voxd-plus --trigger-record

# Run tests
pytest
pytest tests/test_foo.py -v        # Single test file
pytest -k "test_name" -v           # Single test by name

# Diagnostics
voxd-plus --diagnose     # Check audio devices, ydotool status
voxd-plus --verbose      # Enable debug logging
```

## Architecture

### Core Processing Pipeline

1. **Recording** (`core/recorder.py`) - Captures audio via sounddevice
2. **Transcription** (`core/transcriber.py`, `core/streaming_transcriber.py`) - Sends audio to whisper.cpp CLI
3. **AIPP** (`core/aipp.py`) - Optional AI post-processing via LLM providers
4. **Output** (`core/typer.py`, `core/clipboard.py`) - Types text via ydotool/xdotool or copies to clipboard

### Two Transcription Modes

- **Standard mode** (`core/voxd_core.py`) - Records full audio, transcribes after recording stops
- **Streaming mode** (`core/streaming_core.py`, `core/streaming_transcriber.py`) - Transcribes in chunks during recording, types incrementally

### UI Surfaces

All UI modes share the same core processing:
- `gui/gui_main.py` - PyQt6 GUI with settings dialog
- `tray/tray_main.py` - System tray icon
- `cli/cli_main.py` - Command-line interface
- `flux/flux_main.py` - Voice activity detection triggered mode

### Overlay System

The recording overlay (`overlay/`) runs independently and monitors the microphone:
- `recording_overlay.py` - Main overlay window with timer, status
- `waveform_widget.py` - Animated FFT-based frequency visualizer
- `audio_cues.py` - Synthesized tones or custom WAV files

### IPC

Hotkey triggering uses Unix socket IPC:
- `utils/ipc_server.py` - Listens for trigger signals in GUI/tray modes
- `utils/ipc_client.py` - Sends trigger signal (`voxd-plus --trigger-record`)

## Configuration

Config file: `~/.config/voxd-plus/config.yaml`

Key settings in `core/config.py`:
- `typing_method`: "clipboard" (default) or "direct" - clipboard paste avoids keyboard layout issues
- `gpu_enabled`, `gpu_device`: GPU acceleration settings
- `streaming_mode`: Enable real-time incremental transcription
- `aipp_provider`: LLM provider for post-processing (ollama, openai, anthropic, gemini, groq, openrouter, lmstudio)

## LLM Providers

AIPP (AI Post-Processing) supports multiple providers configured via environment variables:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `GROQ_API_KEY`, `OPENROUTER_API_KEY`
- `LMSTUDIO_API_BASE` (defaults to `http://localhost:1234/v1`)

## GPU Support

whisper.cpp is built with CUDA support when available. GPU detection in `utils/gpu_detect.py`:
- Checks for nvidia-smi and CUDA toolkit
- Falls back to CPU if GPU fails
- Config: `gpu_device` can be "auto", "cuda", or "cpu"

## Testing Notes

Tests use pytest with fixtures that:
- Isolate XDG directories to temp paths
- Stub sounddevice for headless testing
- Set `QT_QPA_PLATFORM=offscreen` for Qt

## Hardware Context

Development machine: TUXEDO InfinityBook Pro AMD Gen10 laptop, AMD Ryzen AI 9 365 (20 threads), 64 GiB RAM, AMD Radeon 890M integrated GPU (no discrete GPU, no CUDA). KDE Plasma 6.5.2 on Wayland.
