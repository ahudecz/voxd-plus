# VOXD Fork Analysis & Enhanced Roadmap

## Overview

Based on analysis of voxd's 11 open issues and 9 open PRs (as of Dec 30, 2025), this document outlines what we should incorporate, fix, and enhance in our fork.

---

## Open Issues Analysis

### Critical Bugs to Fix

| Issue # | Title | Priority | Our Action |
|---------|-------|----------|------------|
| **#20** | Typing responses limited to 285 characters | 🔴 HIGH | **Fix** - ydotool has buffer limits; need chunked typing |
| **#24** | Whisper.cpp fails because GPU is enabled | 🔴 HIGH | **Fix** - Add GPU detection/fallback logic |
| **#19** | Keyboard layout issue | 🟡 MEDIUM | **Fix** - ydotool doesn't respect non-US layouts |
| **#29** | Unsupported Python but bigger than v3.9 | 🟡 MEDIUM | **Fix** - Relax Python version constraints |

### Feature Requests to Implement

| Issue # | Title | Priority | Our Action |
|---------|-------|----------|------------|
| **#21** | GUI settings to modify llama-server/llama-cli paths | 🟡 MEDIUM | **Implement** - Essential for custom setups |
| **#17** | Flatpak packaging | 🟢 LOW | **Consider** - Good for sandboxed installs |
| **#28** | AppImage | 🟢 LOW | **Consider** - Portable distribution |
| **#11** | Multi language/models support on trigger | 🟡 MEDIUM | **Implement** - Switch models via hotkey |
| **#6** | Manually add flags to models | 🟢 LOW | **Implement** - Advanced whisper.cpp flags |
| **#12** | Version in pyproject | 🟢 LOW | **Fix** - Proper semver versioning |

### Community Notes

| Issue # | Title | Notes |
|---------|-------|-------|
| **#13** | "Thank you for building this" | Shows community appreciation; keep UX friendly |

---

## Open PRs Analysis

### Ready to Merge (High Value)

| PR # | Title | Status | Our Action |
|------|-------|--------|------------|
| **#22** | Fix 285-character truncation in ydotool typing | Ready | **Merge** - Fixes critical bug #20 |
| **#25** | Dark mode settings background fix (Ubuntu 25.04) | Ready | **Merge** - UI consistency |
| **#26** | Make copy to clipboard configurable | Ready | **Merge** - User preference |
| **#15** | Remove hard coded python versions | Ready | **Merge** - Fixes #29 |
| **#18** | Fix invalid version string with valid semver | Ready | **Merge** - Fixes #12 |

### Feature PRs to Evaluate

| PR # | Title | Status | Our Action |
|------|-------|--------|------------|
| **#31** | Add streaming transcription + Python version handling | New (Dec 30) | **Evaluate** - Real-time transcription! |
| **#27** | Add OPENAI_API_BASE environment variable | Ready | **Merge** - Enables custom endpoints (LM Studio, etc) |
| **#23** | Add GUI settings for llama.cpp paths | Ready | **Merge** - Fixes #21 |
| **#16** | GPU Acceleration Support | In Review | **Evaluate & Merge** - Major performance boost |

---

## Enhanced Fork Roadmap

### Phase 1: Foundation (Week 1)
**Goal: Stable, bug-free base with all community fixes**

```
□ Merge all ready PRs (#15, #18, #22, #25, #26, #27)
□ Evaluate and merge #23 (llama.cpp path settings)
□ Evaluate and merge #16 (GPU acceleration)
□ Fix keyboard layout issues (#19)
□ Add proper error handling for GPU fallback (#24)
□ Update pyproject.toml with proper versioning
□ Test on Ubuntu 24.04, Fedora 42, Arch
```

### Phase 2: UX Enhancements (Week 2)
**Goal: Audio/visual feedback that voxd lacks**

```
□ Audio cues module (start/stop/error tones) ✅ DONE
□ Recording overlay with waveform ✅ DONE  
□ Recording time indicator ✅ DONE
□ Level meter/clipping indicator
□ Overlay position persistence (save to config)
□ Customizable cue sounds (or wav file support)
□ Visual recording state in tray icon
```

### Phase 3: LLM Provider Expansion (Week 3)
**Goal: More AI post-processing options**

```
Current voxd providers:
  ✓ llama.cpp (local)
  ✓ Ollama (local)
  ✓ OpenAI
  ✓ Anthropic
  ✓ xAI

Add:
  □ Google Gemini API
  □ Groq (ultra-fast inference)
  □ OpenRouter (access to many models)
  □ LM Studio API (local, OpenAI-compatible) - PR #27 helps!
  □ Together.ai
  □ Local via text-generation-webui API
```

### Phase 4: Settings Overhaul (Week 4)
**Goal: Comprehensive, user-friendly configuration**

```
□ Unified settings UI (not scattered across menus)
□ Audio device selection with live preview
□ Microphone input level visualization
□ Whisper model comparison (size/speed/accuracy)
□ Per-provider prompt templates
  - Grammar cleanup
  - Professional tone
  - Code formatting
  - Custom user prompts
□ Hotkey configuration with conflict detection
□ VAD sensitivity tuning with visual feedback
□ Export/import settings
```

### Phase 5: Advanced Features (Week 5+)
**Goal: Power user capabilities**

```
□ Streaming transcription (evaluate PR #31)
□ Multi-language quick-switch (hotkey per language)
□ Transcription history with search
□ Continuous dictation mode (like --flux but stable)
□ Wake word activation (optional)
□ Punctuation/formatting commands ("period", "new paragraph")
□ Clipboard history integration
□ Export transcripts (txt, srt, json)
```

### Phase 6: Distribution (Ongoing)
**Goal: Easy installation everywhere**

```
□ AppImage build (portable)
□ Flatpak manifest (sandboxed)
□ AUR package (Arch)
□ PPA for Ubuntu/Debian
□ Copr for Fedora
□ Nix package
□ Auto-update mechanism
```

---

## Technical Debt to Address

### Code Quality
- [ ] Add type hints throughout codebase
- [ ] Increase test coverage
- [ ] Add pre-commit hooks (black, isort, mypy)
- [ ] Document all config options
- [ ] Create developer setup guide

### Architecture Improvements
- [ ] Separate core engine from UI (enable headless mode)
- [ ] Plugin architecture for LLM providers
- [ ] Abstract audio backend (support PulseAudio, PipeWire, ALSA)
- [ ] IPC mechanism for external control (what I built with sockets)

### Performance
- [ ] Lazy load models
- [ ] Memory usage optimization
- [ ] Startup time improvement
- [ ] GPU memory management

---

## Naming Considerations

If we fork significantly, consider renaming to avoid confusion:
- **voxd-enhanced** (maintains lineage)
- **voxd-plus**
- **whispertype** 
- **voxtend**
- **dictate** (simple, memorable)

---

## Immediate Next Steps

1. **Fork voxd** to your GitHub
2. **Create enhancement branch** 
3. **Cherry-pick/merge the ready PRs**:
   - #15 (Python versions)
   - #18 (semver fix)
   - #22 (285 char fix)
   - #25 (dark mode)
   - #26 (clipboard toggle)
   - #27 (OPENAI_API_BASE)
   - #23 (llama.cpp paths)
4. **Evaluate #16** (GPU) and **#31** (streaming)
5. **Integrate our overlay module**
6. **Test thoroughly**
7. **Document changes**

---

## Resources

- voxd repo: https://github.com/jakovius/voxd
- whisper.cpp: https://github.com/ggml-org/whisper.cpp
- llama.cpp: https://github.com/ggml-org/llama.cpp
- PyQt6 docs: https://doc.qt.io/qtforpython-6/
- sounddevice: https://python-sounddevice.readthedocs.io/

---

*Document created: Dec 30, 2025*
*For: Ryan @ Yeeboo Digital*
