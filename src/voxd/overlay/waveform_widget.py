"""
Real-time waveform visualization widget
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient


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
