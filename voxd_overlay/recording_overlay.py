#!/usr/bin/env python3
"""
Recording Overlay with Live Waveform Visualization
===================================================
A non-invasive popup window showing:
- Real-time audio waveform
- Recording time indicator
- Audio cues for start/stop

Designed to complement voxd or work standalone.
Dependencies: PyQt6, numpy, sounddevice

Install: pip install PyQt6 numpy sounddevice
"""

import sys
import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Callable

try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QPushButton, QSystemTrayIcon
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient
except ImportError:
    print("PyQt6 not found. Install with: pip install PyQt6")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not found. Install with: pip install sounddevice")
    sys.exit(1)


class AudioCue:
    """Generate simple audio cues using numpy + sounddevice"""
    
    SAMPLE_RATE = 44100
    
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
    def play_start_cue():
        """Play ascending two-tone cue for recording start"""
        tone1 = AudioCue.generate_tone(880, 0.08, 0.25)   # A5
        tone2 = AudioCue.generate_tone(1320, 0.12, 0.25)  # E6
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        threading.Thread(target=lambda: sd.play(cue, AudioCue.SAMPLE_RATE), daemon=True).start()
    
    @staticmethod
    def play_stop_cue():
        """Play descending two-tone cue for recording stop"""
        tone1 = AudioCue.generate_tone(1320, 0.08, 0.25)  # E6
        tone2 = AudioCue.generate_tone(880, 0.12, 0.25)   # A5
        gap = np.zeros(int(AudioCue.SAMPLE_RATE * 0.02), dtype=np.float32)
        cue = np.concatenate([tone1, gap, tone2])
        threading.Thread(target=lambda: sd.play(cue, AudioCue.SAMPLE_RATE), daemon=True).start()
    
    @staticmethod
    def play_error_cue():
        """Play low buzz for errors"""
        tone = AudioCue.generate_tone(220, 0.2, 0.2)
        threading.Thread(target=lambda: sd.play(tone, AudioCue.SAMPLE_RATE), daemon=True).start()


class AudioSignals(QObject):
    """Qt signals for thread-safe audio data updates"""
    audio_data = pyqtSignal(np.ndarray)
    level_data = pyqtSignal(float)


class AudioRecorder:
    """
    Non-blocking audio recorder with callback for visualization data
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_duration = buffer_duration
        
        self.recording = False
        self.stream: Optional[sd.InputStream] = None
        self.signals = AudioSignals()
        
        # Rolling buffer for waveform display
        buffer_size = int(sample_rate * buffer_duration)
        self.waveform_buffer = deque(maxlen=buffer_size)
        
        # Full recording buffer
        self.recorded_frames: list = []
        
        self.start_time: Optional[float] = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info, status):
        """Called by sounddevice for each audio chunk"""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to mono float32 if needed
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio = audio.astype(np.float32)
        
        # Store for final output
        self.recorded_frames.append(audio.copy())
        
        # Update rolling buffer for visualization
        self.waveform_buffer.extend(audio)
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Emit signals for UI update
        self.signals.audio_data.emit(np.array(self.waveform_buffer))
        self.signals.level_data.emit(float(rms))
    
    def start(self) -> bool:
        """Start recording"""
        if self.recording:
            return False
        
        try:
            self.recorded_frames = []
            self.waveform_buffer.clear()
            self.start_time = time.time()
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=1024,
                callback=self._audio_callback
            )
            self.stream.start()
            self.recording = True
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            return False
    
    def stop(self) -> Optional[np.ndarray]:
        """Stop recording and return audio data"""
        if not self.recording:
            return None
        
        self.recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.recorded_frames:
            return np.concatenate(self.recorded_frames)
        return None
    
    def get_elapsed_time(self) -> float:
        """Get recording duration in seconds"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class WaveformWidget(QWidget):
    """
    Custom widget for drawing real-time audio waveform
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = np.zeros(1000)
        self.level = 0.0
        
        # Visual settings
        self.bg_color = QColor(20, 22, 30)
        self.wave_color = QColor(0, 200, 255)
        self.wave_color_hot = QColor(255, 100, 100)
        self.grid_color = QColor(40, 45, 60)
        self.center_line_color = QColor(60, 70, 90)
        
        self.setMinimumSize(300, 80)
    
    def update_data(self, data: np.ndarray):
        """Update waveform data"""
        self.audio_data = data
        self.update()
    
    def update_level(self, level: float):
        """Update audio level"""
        self.level = min(level * 10, 1.0)  # Scale for visibility
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        center_y = h // 2
        
        # Background
        painter.fillRect(0, 0, w, h, self.bg_color)
        
        # Grid lines
        painter.setPen(QPen(self.grid_color, 1))
        for i in range(1, 4):
            y = int(h * i / 4)
            painter.drawLine(0, y, w, y)
        
        # Center line
        painter.setPen(QPen(self.center_line_color, 1))
        painter.drawLine(0, center_y, w, center_y)
        
        # Waveform
        if len(self.audio_data) > 1:
            # Downsample for display
            display_samples = min(len(self.audio_data), w * 2)
            step = max(1, len(self.audio_data) // display_samples)
            data = self.audio_data[::step]
            
            # Normalize
            max_val = max(np.max(np.abs(data)), 0.001)
            normalized = data / max_val
            
            # Color based on level
            if self.level > 0.8:
                color = self.wave_color_hot
            else:
                color = self.wave_color
            
            # Draw waveform
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            
            points_per_pixel = len(data) / w
            prev_x, prev_y = 0, center_y
            
            for i, sample in enumerate(data):
                x = int(i / len(data) * w)
                y = int(center_y - sample * (h // 2 - 5))
                y = max(2, min(h - 2, y))
                
                if i > 0:
                    painter.drawLine(prev_x, prev_y, x, y)
                
                prev_x, prev_y = x, y
        
        # Level indicator bar at bottom
        bar_height = 3
        bar_width = int(w * self.level)
        gradient = QLinearGradient(0, h - bar_height, bar_width, h - bar_height)
        gradient.setColorAt(0, QColor(0, 200, 100))
        gradient.setColorAt(0.7, QColor(255, 200, 0))
        gradient.setColorAt(1, QColor(255, 50, 50))
        painter.fillRect(0, h - bar_height, bar_width, bar_height, gradient)


class RecordingOverlay(QWidget):
    """
    Main overlay window with waveform, timer, and controls
    """
    
    # Signal emitted when recording completes with audio data
    recording_complete = pyqtSignal(np.ndarray)
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 play_audio_cues: bool = True,
                 always_on_top: bool = True,
                 parent=None):
        super().__init__(parent)
        
        self.play_audio_cues = play_audio_cues
        self.recorder = AudioRecorder(sample_rate=sample_rate)
        
        self._setup_ui()
        self._connect_signals()
        
        # Window properties
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_timer_display)
        
        # Dragging support
        self._drag_pos = None
    
    def _setup_ui(self):
        """Build the UI"""
        self.setFixedSize(320, 130)
        
        # Main container with rounded corners
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Container widget for styling
        self.container = QWidget()
        self.container.setObjectName("container")
        self.container.setStyleSheet("""
            #container {
                background-color: rgba(25, 28, 38, 245);
                border: 1px solid rgba(80, 90, 120, 150);
                border-radius: 12px;
            }
        """)
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(12, 10, 12, 10)
        container_layout.setSpacing(8)
        
        # Top row: status and time
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        
        # Recording indicator
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #666; font-size: 14px;")
        top_row.addWidget(self.status_dot)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            color: #aaa;
            font-size: 12px;
            font-weight: 500;
        """)
        top_row.addWidget(self.status_label)
        
        top_row.addStretch()
        
        # Timer display
        self.timer_label = QLabel("00:00.0")
        self.timer_label.setStyleSheet("""
            color: #0cf;
            font-size: 18px;
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
            font-weight: 600;
        """)
        top_row.addWidget(self.timer_label)
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #f55;
            }
        """)
        close_btn.clicked.connect(self.close)
        top_row.addWidget(close_btn)
        
        container_layout.addLayout(top_row)
        
        # Waveform display
        self.waveform = WaveformWidget()
        self.waveform.setFixedHeight(60)
        container_layout.addWidget(self.waveform)
        
        # Bottom row: hint
        self.hint_label = QLabel("Press hotkey to start • Drag to move")
        self.hint_label.setStyleSheet("""
            color: #555;
            font-size: 10px;
        """)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.hint_label)
        
        main_layout.addWidget(self.container)
    
    def _connect_signals(self):
        """Connect audio signals to UI"""
        self.recorder.signals.audio_data.connect(self.waveform.update_data)
        self.recorder.signals.level_data.connect(self.waveform.update_level)
    
    def _update_timer_display(self):
        """Update the recording time display"""
        elapsed = self.recorder.get_elapsed_time()
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:05.2f}")
    
    def toggle_recording(self) -> bool:
        """Toggle recording state. Returns True if now recording."""
        if self.recorder.recording:
            return self.stop_recording()
        else:
            return self.start_recording()
    
    def start_recording(self) -> bool:
        """Start recording"""
        if self.recorder.recording:
            return False
        
        if self.play_audio_cues:
            AudioCue.play_start_cue()
        
        # Small delay after cue before starting
        if self.play_audio_cues:
            time.sleep(0.15)
        
        success = self.recorder.start()
        
        if success:
            self.status_dot.setStyleSheet("color: #f44; font-size: 14px;")
            self.status_label.setText("Recording")
            self.status_label.setStyleSheet("color: #f88; font-size: 12px; font-weight: 500;")
            self.hint_label.setText("Press hotkey to stop")
            self.update_timer.start(50)  # 20 fps timer update
            return True
        else:
            if self.play_audio_cues:
                AudioCue.play_error_cue()
            return False
    
    def stop_recording(self) -> bool:
        """Stop recording"""
        if not self.recorder.recording:
            return False
        
        self.update_timer.stop()
        audio_data = self.recorder.stop()
        
        if self.play_audio_cues:
            AudioCue.play_stop_cue()
        
        self.status_dot.setStyleSheet("color: #666; font-size: 14px;")
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: #aaa; font-size: 12px; font-weight: 500;")
        self.hint_label.setText("Press hotkey to start • Drag to move")
        
        if audio_data is not None and len(audio_data) > 0:
            self.recording_complete.emit(audio_data)
        
        return True
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recorder.recording
    
    # Dragging support
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None


class RecordingOverlayManager:
    """
    Manager class for integrating the overlay with external hotkey systems
    (like voxd's trigger-record mechanism)
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 play_audio_cues: bool = True,
                 on_recording_complete: Optional[Callable[[np.ndarray], None]] = None):
        self.app: Optional[QApplication] = None
        self.overlay: Optional[RecordingOverlay] = None
        self.sample_rate = sample_rate
        self.play_audio_cues = play_audio_cues
        self.on_recording_complete = on_recording_complete
        self._initialized = False
    
    def _ensure_app(self):
        """Ensure Qt application exists"""
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        self._initialized = True
    
    def show(self, x: Optional[int] = None, y: Optional[int] = None):
        """Show the overlay window"""
        self._ensure_app()
        
        if self.overlay is None:
            self.overlay = RecordingOverlay(
                sample_rate=self.sample_rate,
                play_audio_cues=self.play_audio_cues
            )
            if self.on_recording_complete:
                self.overlay.recording_complete.connect(self.on_recording_complete)
        
        # Position
        if x is not None and y is not None:
            self.overlay.move(x, y)
        else:
            # Default: bottom right corner
            screen = QApplication.primaryScreen().geometry()
            self.overlay.move(
                screen.width() - self.overlay.width() - 20,
                screen.height() - self.overlay.height() - 80
            )
        
        self.overlay.show()
    
    def hide(self):
        """Hide the overlay window"""
        if self.overlay:
            self.overlay.hide()
    
    def toggle_recording(self) -> bool:
        """Toggle recording state"""
        if self.overlay:
            return self.overlay.toggle_recording()
        return False
    
    def is_recording(self) -> bool:
        """Check if recording"""
        if self.overlay:
            return self.overlay.is_recording()
        return False


def demo():
    """Demo/test function"""
    app = QApplication(sys.argv)
    
    def on_complete(audio: np.ndarray):
        duration = len(audio) / 16000
        print(f"\n✓ Recording complete: {duration:.2f}s, {len(audio)} samples")
        print("  (In real use, this would be sent to Whisper for transcription)")
    
    overlay = RecordingOverlay(
        sample_rate=16000,
        play_audio_cues=True
    )
    overlay.recording_complete.connect(on_complete)
    
    # Position bottom-right
    screen = app.primaryScreen().geometry()
    overlay.move(
        screen.width() - overlay.width() - 20,
        screen.height() - overlay.height() - 80
    )
    
    overlay.show()
    
    print("=" * 50)
    print("Recording Overlay Demo")
    print("=" * 50)
    print("\nClick on the overlay and:")
    print("  - Double-click to toggle recording")
    print("  - Drag to reposition")
    print("  - Click × to close")
    print("\nOr use keyboard: R to toggle, Q to quit")
    print()
    
    # Simple keyboard handling for demo
    from PyQt6.QtCore import QObject
    from PyQt6.QtGui import QKeyEvent
    
    class KeyHandler(QObject):
        def __init__(self, overlay):
            super().__init__()
            self.overlay = overlay
        
        def eventFilter(self, obj, event):
            if isinstance(event, QKeyEvent):
                key = event.key()
                if key == Qt.Key.Key_R:
                    self.overlay.toggle_recording()
                    return True
                elif key == Qt.Key.Key_Q:
                    app.quit()
                    return True
            return False
    
    handler = KeyHandler(overlay)
    app.installEventFilter(handler)
    
    # Double-click to toggle
    original_mouse_double = overlay.mouseDoubleClickEvent
    def double_click_handler(event):
        overlay.toggle_recording()
    overlay.mouseDoubleClickEvent = double_click_handler
    
    sys.exit(app.exec())


if __name__ == "__main__":
    demo()
