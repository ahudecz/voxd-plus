# Recording Overlay for voxd

A non-invasive recording popup with live waveform visualization and audio cues, designed to enhance [voxd](https://github.com/jakovius/voxd) or work standalone.

![Demo concept](demo-concept.png)

## Features

- **Audio Cues**: Ascending tones on start, descending on stop
- **Live Waveform**: Real-time visualization of audio input
- **Timer Display**: Precise recording duration in `MM:SS.ss` format
- **Level Indicator**: Color-coded bar showing audio input level
- **Non-invasive**: Small, draggable, always-on-top window
- **Easy Integration**: Works via socket IPC or can be embedded directly

## Installation

```bash
# Install dependencies (same as voxd uses)
pip install PyQt6 numpy sounddevice

# Optional: for the demo to save wav files
pip install wave  # Usually included with Python
```

## Quick Start

### Demo Mode

Run the overlay directly to test it:

```bash
python recording_overlay.py
```

Controls:
- **R** - Toggle recording
- **Q** - Quit
- **Double-click** - Toggle recording
- **Drag** - Reposition window
- **×** - Close

### Daemon Mode (for hotkey integration)

Start the overlay daemon:

```bash
python trigger_integration.py --daemon
```

Then configure your system hotkey to run:

```bash
python trigger_integration.py --trigger
```

This sends a toggle command to the running daemon.

## Integration with voxd

### Option 1: Subprocess Mode (No voxd changes)

1. Run the overlay daemon alongside voxd:
   ```bash
   python trigger_integration.py --daemon &
   ```

2. Modify your hotkey to trigger both:
   ```bash
   bash -c 'python trigger_integration.py --trigger; voxd --trigger-record'
   ```

### Option 2: Embedded Mode (Modify voxd)

Add the overlay directly to voxd's GUI. See:

```bash
python trigger_integration.py --show-integration
```

This prints code examples for:
- Subprocess integration
- Embedded integration
- Audio-cues-only integration

### Option 3: Audio Cues Only

If you just want the audio feedback without the visual overlay, copy the `AudioCue` class from `recording_overlay.py` into voxd's recorder.py.

## Configuration

### Sample Rate

Match voxd's Whisper model sample rate (default 16000):

```bash
python trigger_integration.py --daemon --sample-rate 16000
```

### Disable Audio Cues

```bash
python trigger_integration.py --daemon --no-audio-cues
```

### Positioning

The overlay defaults to bottom-right corner. Drag it to your preferred location—the position persists during the session.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  recording_overlay.py                           │
│  ├── AudioCue - Sound feedback                  │
│  ├── AudioRecorder - sounddevice wrapper        │
│  ├── WaveformWidget - PyQt6 custom painter      │
│  └── RecordingOverlay - Main UI widget          │
└─────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  trigger_integration.py                         │
│  ├── TriggerServer - Unix socket IPC           │
│  ├── send_trigger_command() - Client function   │
│  └── VoxdOverlayIntegration - Code examples     │
└─────────────────────────────────────────────────┘
```

## API Reference

### RecordingOverlay Class

```python
from recording_overlay import RecordingOverlay

overlay = RecordingOverlay(
    sample_rate=16000,      # Audio sample rate
    play_audio_cues=True,   # Enable/disable sounds
    always_on_top=True      # Keep window above others
)

# Connect to receive audio data
overlay.recording_complete.connect(my_handler)

# Control recording
overlay.start_recording()
overlay.stop_recording()
overlay.toggle_recording()
overlay.is_recording()

# Show/position
overlay.show()
overlay.move(x, y)
```

### RecordingOverlayManager Class

Convenient wrapper for external integration:

```python
from recording_overlay import RecordingOverlayManager

manager = RecordingOverlayManager(
    sample_rate=16000,
    play_audio_cues=True,
    on_recording_complete=my_transcription_function
)

manager.show()
manager.toggle_recording()
manager.hide()
```

## Customization

### Audio Cues

Modify `AudioCue` class in `recording_overlay.py`:

```python
# Change frequencies (Hz) and durations (seconds)
@staticmethod
def play_start_cue():
    tone1 = AudioCue.generate_tone(880, 0.08)   # A5
    tone2 = AudioCue.generate_tone(1320, 0.12)  # E6
    # ...
```

Or replace with WAV files:

```python
import sounddevice as sd
import soundfile as sf

def play_start_cue():
    data, samplerate = sf.read('start.wav')
    sd.play(data, samplerate)
```

### Visual Style

Modify colors in `WaveformWidget` and CSS in `RecordingOverlay._setup_ui()`:

```python
# WaveformWidget colors
self.bg_color = QColor(20, 22, 30)
self.wave_color = QColor(0, 200, 255)
self.wave_color_hot = QColor(255, 100, 100)

# Container CSS
self.container.setStyleSheet("""
    #container {
        background-color: rgba(25, 28, 38, 245);
        border: 1px solid rgba(80, 90, 120, 150);
        border-radius: 12px;
    }
""")
```

## License

MIT - Feel free to integrate into voxd or use independently.

## Contributing

This is a proof-of-concept. Ideas for improvement:
- FFT frequency spectrum option
- Configurable window size
- Position persistence across sessions
- Wayland-native positioning hints
- Direct voxd plugin architecture
