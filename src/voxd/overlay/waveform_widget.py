"""
Real-time audio visualization widget with animated bars
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient, QBrush


class WaveformWidget(QWidget):
    """
    Animated audio visualizer with bouncy bars that respond to audio levels
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Number of bars in the visualization
        self.num_bars = 32

        # Current bar heights (0.0 to 1.0)
        self.bar_heights = np.zeros(self.num_bars)
        # Peak heights for decay effect
        self.peak_heights = np.zeros(self.num_bars)
        # Velocity for smooth animation
        self.velocities = np.zeros(self.num_bars)

        # Audio data buffer
        self.audio_data = np.zeros(2048)
        self.level = 0.0

        # Animation settings
        self.gravity = 0.015  # How fast bars fall
        self.bounce = 0.3    # Bounce factor
        self.peak_decay = 0.02  # How fast peaks fall
        self.smoothing = 0.3  # Smoothing between frames

        # Visual settings
        self.bg_color = QColor(20, 22, 30)
        self.bar_spacing = 2
        self.bar_radius = 2

        # Colors
        self.color_low = QColor(0, 200, 255)      # Cyan for low levels
        self.color_mid = QColor(100, 255, 150)    # Green for mid levels
        self.color_high = QColor(255, 100, 100)   # Red for high levels
        self.peak_color = QColor(255, 255, 255, 200)  # White peaks
        self.glow_color = QColor(0, 200, 255, 40)  # Subtle glow

        self.setMinimumSize(300, 80)

        # Animation timer for smooth updates
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(16)  # ~60fps

    def update_data(self, data: np.ndarray):
        """Update with new audio data"""
        self.audio_data = data
        self._process_audio()

    def update_level(self, level: float):
        """Update audio level"""
        self.level = min(level * 10, 1.0)

    def _process_audio(self):
        """Process audio data into bar heights using FFT"""
        if len(self.audio_data) < 256:
            return

        # Take recent samples for responsiveness
        samples = self.audio_data[-2048:] if len(self.audio_data) >= 2048 else self.audio_data

        # Apply window function to reduce spectral leakage
        window = np.hanning(len(samples))
        windowed = samples * window

        # FFT
        fft = np.abs(np.fft.rfft(windowed))

        # Only use lower frequencies (more musically relevant)
        useful_bins = len(fft) // 4
        fft = fft[:useful_bins]

        if len(fft) < self.num_bars:
            return

        # Group FFT bins into bars with logarithmic scaling
        # This gives more resolution to lower frequencies
        bar_targets = np.zeros(self.num_bars)

        for i in range(self.num_bars):
            # Logarithmic distribution of frequency bins
            start = int((i / self.num_bars) ** 1.5 * len(fft))
            end = int(((i + 1) / self.num_bars) ** 1.5 * len(fft))
            end = max(end, start + 1)

            # Average the bins for this bar
            bar_targets[i] = np.mean(fft[start:end])

        # Normalize
        max_val = np.max(bar_targets)
        if max_val > 0.001:
            bar_targets = bar_targets / max_val

        # Apply some gain and compression for visual appeal
        bar_targets = np.power(bar_targets, 0.7) * 1.2
        bar_targets = np.clip(bar_targets, 0, 1)

        # Update bar heights with physics
        for i in range(self.num_bars):
            target = bar_targets[i]
            current = self.bar_heights[i]

            if target > current:
                # Jump up quickly
                self.bar_heights[i] = current + (target - current) * 0.6
                self.velocities[i] = 0.02  # Small upward velocity
            else:
                # Apply gravity for falling
                self.velocities[i] -= self.gravity
                self.bar_heights[i] += self.velocities[i]

                # Bounce at bottom
                if self.bar_heights[i] < 0:
                    self.bar_heights[i] = 0
                    self.velocities[i] = -self.velocities[i] * self.bounce

            # Update peak
            if self.bar_heights[i] > self.peak_heights[i]:
                self.peak_heights[i] = self.bar_heights[i]

    def _animate(self):
        """Animation frame - decay peaks and repaint"""
        # Decay peaks slowly
        self.peak_heights = np.maximum(
            self.peak_heights - self.peak_decay,
            self.bar_heights
        )

        # Continue gravity on bars even without new audio
        for i in range(self.num_bars):
            if self.bar_heights[i] > 0:
                self.velocities[i] -= self.gravity * 0.5
                self.bar_heights[i] += self.velocities[i]
                if self.bar_heights[i] < 0:
                    self.bar_heights[i] = 0
                    self.velocities[i] = 0

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, self.bg_color)

        # Calculate bar dimensions
        total_spacing = self.bar_spacing * (self.num_bars - 1)
        bar_width = max(2, (w - total_spacing) / self.num_bars)
        max_bar_height = h - 10  # Leave some padding

        center_y = h // 2

        # Draw bars (mirrored - both up and down from center)
        for i in range(self.num_bars):
            x = int(i * (bar_width + self.bar_spacing))
            height = int(self.bar_heights[i] * max_bar_height / 2)

            if height < 1:
                # Draw minimum indicator line
                painter.setPen(QPen(QColor(60, 70, 90), 1))
                painter.drawLine(int(x), center_y, int(x + bar_width), center_y)
                continue

            # Create gradient based on height
            bar_top = center_y - height
            bar_bottom = center_y + height

            gradient = QLinearGradient(0, bar_top, 0, center_y)

            # Color based on height
            intensity = self.bar_heights[i]
            if intensity > 0.7:
                gradient.setColorAt(0, self.color_high)
                gradient.setColorAt(0.5, self.color_mid)
                gradient.setColorAt(1, self.color_low)
            elif intensity > 0.4:
                gradient.setColorAt(0, self.color_mid)
                gradient.setColorAt(1, self.color_low)
            else:
                gradient.setColorAt(0, self.color_low)
                gradient.setColorAt(1, QColor(0, 150, 200))

            # Draw glow effect for active bars
            if height > 5:
                glow_rect = (
                    int(x - 2), center_y - height - 2,
                    int(bar_width + 4), height * 2 + 4
                )
                glow_gradient = QLinearGradient(0, glow_rect[1], 0, glow_rect[1] + glow_rect[3])
                glow_gradient.setColorAt(0, QColor(self.color_low.red(), self.color_low.green(), self.color_low.blue(), 30))
                glow_gradient.setColorAt(0.5, QColor(self.color_low.red(), self.color_low.green(), self.color_low.blue(), 50))
                glow_gradient.setColorAt(1, QColor(self.color_low.red(), self.color_low.green(), self.color_low.blue(), 30))
                painter.fillRect(*glow_rect, glow_gradient)

            # Draw upper bar (above center)
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                int(x), bar_top,
                int(bar_width), height,
                self.bar_radius, self.bar_radius
            )

            # Draw lower bar (below center) - mirrored gradient
            gradient_lower = QLinearGradient(0, center_y, 0, bar_bottom)
            if intensity > 0.7:
                gradient_lower.setColorAt(0, self.color_low)
                gradient_lower.setColorAt(0.5, self.color_mid)
                gradient_lower.setColorAt(1, self.color_high)
            elif intensity > 0.4:
                gradient_lower.setColorAt(0, self.color_low)
                gradient_lower.setColorAt(1, self.color_mid)
            else:
                gradient_lower.setColorAt(0, QColor(0, 150, 200))
                gradient_lower.setColorAt(1, self.color_low)

            painter.setBrush(QBrush(gradient_lower))
            painter.drawRoundedRect(
                int(x), center_y,
                int(bar_width), height,
                self.bar_radius, self.bar_radius
            )

            # Draw peak indicators
            peak_height = int(self.peak_heights[i] * max_bar_height / 2)
            if peak_height > height + 2:
                painter.setPen(QPen(self.peak_color, 2))
                # Upper peak
                painter.drawLine(
                    int(x), center_y - peak_height,
                    int(x + bar_width), center_y - peak_height
                )
                # Lower peak
                painter.drawLine(
                    int(x), center_y + peak_height,
                    int(x + bar_width), center_y + peak_height
                )

        # Level indicator bar at bottom
        bar_height = 3
        bar_width_level = int(w * self.level)
        gradient = QLinearGradient(0, h - bar_height, bar_width_level, h - bar_height)
        gradient.setColorAt(0, QColor(0, 200, 100))
        gradient.setColorAt(0.7, QColor(255, 200, 0))
        gradient.setColorAt(1, QColor(255, 50, 50))
        painter.fillRect(0, h - bar_height, bar_width_level, bar_height, gradient)
