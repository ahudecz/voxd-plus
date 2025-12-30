"""
Recording overlay window with waveform visualization
"""

import time
from collections import deque
from typing import Optional, Callable

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

from voxd.overlay.waveform_widget import WaveformWidget
from voxd.overlay.audio_cues import AudioCue

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioSignals(QObject):
    """Qt signals for thread-safe audio data updates"""
    audio_data = pyqtSignal(np.ndarray)
    level_data = pyqtSignal(float)


class OverlayAudioMonitor:
    """
    Audio monitor that captures mic input for waveform visualization only.
    Does not replace voxd's main recorder - just provides visualization data.
    """

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.monitoring = False
        self.stream: Optional[object] = None
        self.signals = AudioSignals()

        # Rolling buffer for waveform display
        buffer_size = int(sample_rate * buffer_duration)
        self.waveform_buffer = deque(maxlen=buffer_size)

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Called by sounddevice for each audio chunk"""
        if not self.monitoring:
            return

        # Convert to mono float32 if needed
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio = audio.astype(np.float32)

        # Update rolling buffer for visualization
        self.waveform_buffer.extend(audio)

        # Calculate RMS level
        rms = np.sqrt(np.mean(audio ** 2))

        # Emit signals for UI update
        self.signals.audio_data.emit(np.array(self.waveform_buffer))
        self.signals.level_data.emit(float(rms))

    def start(self) -> bool:
        """Start monitoring audio for visualization"""
        if sd is None or self.monitoring:
            return False

        try:
            self.waveform_buffer.clear()

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
                callback=self._audio_callback
            )
            self.stream.start()
            self.monitoring = True
            return True
        except Exception as e:
            print(f"[overlay] Failed to start audio monitor: {e}")
            return False

    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class RecordingOverlay(QWidget):
    """
    Overlay window showing waveform visualization during recording
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 always_on_top: bool = True,
                 parent=None):
        super().__init__(parent)

        self.monitor = OverlayAudioMonitor(sample_rate=sample_rate)
        self.start_time: Optional[float] = None

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
        self.setFixedSize(640, 320)

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
        self.status_dot = QLabel("\u25cf")  # Unicode filled circle
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
        close_btn = QPushButton("\u00d7")  # Unicode multiplication sign
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
        close_btn.clicked.connect(self.hide)
        top_row.addWidget(close_btn)

        container_layout.addLayout(top_row)

        # Waveform display
        self.waveform = WaveformWidget()
        self.waveform.setFixedHeight(240)
        container_layout.addWidget(self.waveform)

        # Bottom row: hint
        self.hint_label = QLabel("Recording... Press hotkey to stop")
        self.hint_label.setStyleSheet("""
            color: #555;
            font-size: 10px;
        """)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.hint_label)

        main_layout.addWidget(self.container)

    def _connect_signals(self):
        """Connect audio signals to UI"""
        self.monitor.signals.audio_data.connect(self.waveform.update_data)
        self.monitor.signals.level_data.connect(self.waveform.update_level)

    def _update_timer_display(self):
        """Update the recording time display"""
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:04.1f}")

    def start_recording_display(self):
        """Start the overlay display for recording"""
        self.start_time = time.time()
        self.monitor.start()

        self.status_dot.setStyleSheet("color: #f44; font-size: 14px;")
        self.status_label.setText("Recording")
        self.status_label.setStyleSheet("color: #f88; font-size: 12px; font-weight: 500;")
        self.hint_label.setText("Recording... Press hotkey to stop")
        self.timer_label.setText("00:00.0")

        self.update_timer.start(100)  # 10 fps timer update
        self.show()

    def stop_recording_display(self):
        """Stop the overlay display"""
        self.update_timer.stop()
        self.monitor.stop()

        self.status_dot.setStyleSheet("color: #666; font-size: 14px;")
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet("color: #aaa; font-size: 12px; font-weight: 500;")
        self.hint_label.setText("Transcribing...")

    def finish_display(self):
        """Called when transcription is complete"""
        self.hide()
        self.status_label.setText("Ready")
        self.hint_label.setText("Recording... Press hotkey to stop")

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
    Singleton manager for the recording overlay.
    Use this to show/hide the overlay from anywhere in voxd.
    """

    _instance: Optional['RecordingOverlayManager'] = None
    _overlay: Optional[RecordingOverlay] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._position = None  # Remember position between sessions

    def _ensure_overlay(self):
        """Create overlay if needed"""
        if self._overlay is None:
            self._overlay = RecordingOverlay()

    def show_recording(self):
        """Show overlay and start recording display"""
        self._ensure_overlay()

        # Position overlay
        if self._position:
            self._overlay.move(*self._position)
        else:
            # Default: bottom right corner
            screen = QApplication.primaryScreen()
            if screen:
                geom = screen.geometry()
                self._overlay.move(
                    geom.width() - self._overlay.width() - 20,
                    geom.height() - self._overlay.height() - 100
                )

        self._overlay.start_recording_display()

    def show_processing(self):
        """Update overlay to show processing state"""
        if self._overlay:
            self._overlay.stop_recording_display()

    def hide(self):
        """Hide the overlay"""
        if self._overlay:
            # Save position for next time
            self._position = (self._overlay.x(), self._overlay.y())
            self._overlay.finish_display()

    def is_visible(self) -> bool:
        """Check if overlay is visible"""
        return self._overlay is not None and self._overlay.isVisible()


# Global convenience function
def get_overlay_manager() -> RecordingOverlayManager:
    """Get the singleton overlay manager"""
    return RecordingOverlayManager()
