"""
Audio cues for recording start/stop feedback
"""

import threading
import numpy as np
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioCue:
    """Generate simple audio cues using numpy + sounddevice"""

    SAMPLE_RATE = 44100
    _default_volume = 0.3

    @staticmethod
    def is_available() -> bool:
        """Check if audio cues can be played"""
        return sd is not None

    @staticmethod
    def _get_volume(cfg=None) -> float:
        """Get volume from config or use default"""
        if cfg:
            try:
                return float(cfg.data.get("audio_cue_volume", AudioCue._default_volume))
            except (TypeError, ValueError):
                pass
        return AudioCue._default_volume

    @staticmethod
    def generate_tone(frequency: float, duration: float, volume: float = 0.3,
                      fade_ms: float = 10) -> np.ndarray:
        """Generate a sine wave tone with fade in/out"""
        samples = int(AudioCue.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        tone = np.sin(2 * np.pi * frequency * t) * volume

        # Apply fade in/out to avoid clicks
        fade_samples = int(AudioCue.SAMPLE_RATE * fade_ms / 1000)
        if fade_samples > 0 and fade_samples < samples // 2:
            fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        return tone

    @staticmethod
    def _play_async(audio: np.ndarray):
        """Play audio in a background thread"""
        if sd is None:
            return
        threading.Thread(
            target=lambda: sd.play(audio, AudioCue.SAMPLE_RATE),
            daemon=True
        ).start()

    @staticmethod
    def _play_file(file_path: str, volume: float = 0.3) -> bool:
        """
        Play a WAV file with volume control.
        Returns True if file was played, False otherwise.
        """
        if sd is None:
            return False

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return False

        try:
            # Try scipy first for WAV files
            try:
                from scipy.io import wavfile
                rate, data = wavfile.read(str(path))
            except ImportError:
                # Fallback to wave module
                import wave
                with wave.open(str(path), 'rb') as wf:
                    rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    if wf.getsampwidth() == 2:
                        data = np.frombuffer(frames, dtype=np.int16)
                    elif wf.getsampwidth() == 4:
                        data = np.frombuffer(frames, dtype=np.int32)
                    else:
                        data = np.frombuffer(frames, dtype=np.uint8)

            # Convert to float32 and apply volume
            if data.dtype == np.int16:
                audio = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                audio = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                audio = (data.astype(np.float32) - 128) / 128.0
            else:
                audio = data.astype(np.float32)

            audio *= volume

            # Play in background thread
            threading.Thread(
                target=lambda: sd.play(audio, rate),
                daemon=True
            ).start()
            return True

        except Exception:
            return False

    @staticmethod
    def _try_custom_file(cfg, file_key: str, volume: float) -> bool:
        """Try to play custom file if enabled and configured"""
        if cfg is None:
            return False

        if not cfg.data.get("audio_cue_use_custom", False):
            return False

        file_path = cfg.data.get(file_key, "")
        if not file_path:
            return False

        return AudioCue._play_file(file_path, volume)

    @staticmethod
    def play_start(cfg=None):
        """Play ascending two-tone cue for recording start"""
        if sd is None:
            return

        volume = AudioCue._get_volume(cfg)

        # Try custom file first
        if AudioCue._try_custom_file(cfg, "audio_cue_start_file", volume):
            return

        # Fallback to synthesized tone
        tone1 = AudioCue.generate_tone(880, 0.08, volume * 0.83)   # A5
        tone2 = AudioCue.generate_tone(1320, 0.12, volume * 0.83)  # E6
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        AudioCue._play_async(cue)

    @staticmethod
    def play_stop(cfg=None):
        """Play descending two-tone cue for recording stop"""
        if sd is None:
            return

        volume = AudioCue._get_volume(cfg)

        # Try custom file first
        if AudioCue._try_custom_file(cfg, "audio_cue_stop_file", volume):
            return

        # Fallback to synthesized tone
        tone1 = AudioCue.generate_tone(1320, 0.08, volume * 0.83)  # E6
        tone2 = AudioCue.generate_tone(880, 0.12, volume * 0.83)   # A5
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        AudioCue._play_async(cue)

    @staticmethod
    def play_error(cfg=None):
        """Play low buzz for errors"""
        if sd is None:
            return

        volume = AudioCue._get_volume(cfg)

        # Try custom file first
        if AudioCue._try_custom_file(cfg, "audio_cue_error_file", volume):
            return

        # Fallback to synthesized tone
        tone = AudioCue.generate_tone(220, 0.2, volume * 0.67)
        AudioCue._play_async(tone)

    @staticmethod
    def play_success(cfg=None):
        """Play pleasant chime for successful transcription"""
        if sd is None:
            return

        volume = AudioCue._get_volume(cfg)

        # Try custom file first
        if AudioCue._try_custom_file(cfg, "audio_cue_success_file", volume):
            return

        # Fallback to synthesized tone
        tone1 = AudioCue.generate_tone(523, 0.08, volume * 0.67)   # C5
        tone2 = AudioCue.generate_tone(659, 0.08, volume * 0.67)   # E5
        tone3 = AudioCue.generate_tone(784, 0.12, volume * 0.67)   # G5
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.015), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2, gap, tone3])
        AudioCue._play_async(cue)

    @staticmethod
    def test_cue(cue_type: str, cfg=None):
        """Test a specific audio cue by type name"""
        if cue_type == "start":
            AudioCue.play_start(cfg)
        elif cue_type == "stop":
            AudioCue.play_stop(cfg)
        elif cue_type == "success":
            AudioCue.play_success(cfg)
        elif cue_type == "error":
            AudioCue.play_error(cfg)
