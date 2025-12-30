"""
Audio cues for recording start/stop feedback
"""

import threading
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioCue:
    """Generate simple audio cues using numpy + sounddevice"""

    SAMPLE_RATE = 44100

    @staticmethod
    def is_available() -> bool:
        """Check if audio cues can be played"""
        return sd is not None

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
    def play_start():
        """Play ascending two-tone cue for recording start"""
        if sd is None:
            return
        tone1 = AudioCue.generate_tone(880, 0.08, 0.25)   # A5
        tone2 = AudioCue.generate_tone(1320, 0.12, 0.25)  # E6
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        AudioCue._play_async(cue)

    @staticmethod
    def play_stop():
        """Play descending two-tone cue for recording stop"""
        if sd is None:
            return
        tone1 = AudioCue.generate_tone(1320, 0.08, 0.25)  # E6
        tone2 = AudioCue.generate_tone(880, 0.12, 0.25)   # A5
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        AudioCue._play_async(cue)

    @staticmethod
    def play_error():
        """Play low buzz for errors"""
        if sd is None:
            return
        tone = AudioCue.generate_tone(220, 0.2, 0.2)
        AudioCue._play_async(tone)

    @staticmethod
    def play_success():
        """Play pleasant chime for successful transcription"""
        if sd is None:
            return
        tone1 = AudioCue.generate_tone(523, 0.08, 0.2)   # C5
        tone2 = AudioCue.generate_tone(659, 0.08, 0.2)   # E5
        tone3 = AudioCue.generate_tone(784, 0.12, 0.2)   # G5
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.015), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2, gap, tone3])
        AudioCue._play_async(cue)
