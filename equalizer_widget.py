# equalizer_widget.py
# Contains the EqualizerGLWidget class for ChronoMIDI.

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import (
    glEnable, glBlendFunc, glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    glViewport, glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
    glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
)

class EqualizerGLWidget(QOpenGLWidget):
    """
    An OpenGL widget for displaying a real-time audio equalizer visualization.
    It processes incoming PCM audio data to calculate frequency band levels
    and renders them as vertical bars using OpenGL immediate mode.
    """
    def __init__(self, sr=44100, bands=128, decay=0.92, parent=None):
        """
        Initializes the EqualizerGLWidget.

        Args:
            sr (int): Sample rate of the audio.
            bands (int): Number of frequency bands to display.
            decay (float): Decay rate for the bar levels (how fast they fall).
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.sr = sr          # Audio sample rate
        self.bands = bands    # Number of equalizer bars
        self.decay = decay    # How fast the bars decay after a peak
        self.levels = [0.0] * bands # Current amplitude levels for each band, initialized to zero

        # QTimer to trigger updates for smooth animation
        # The update method will call paintGL at ~60 FPS
        QTimer(self, timeout=self.update, interval=1000 // 60).start()

    def push_audio(self, pcm: np.ndarray):
        """
        Processes a block of PCM audio data to update the equalizer bar levels.
        Performs FFT to get frequency spectrum and averages into bands.

        Args:
            pcm (np.ndarray): Stereo PCM audio data (e.g., int16 or float).
        """
        mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
        # Perform Real FFT (rfft) to get the frequency spectrum
        spec = np.abs(np.fft.rfft(mono))

        # Calculate chunk size for averaging frequencies into bands
        # Ensure chunk is at least 1 to prevent division by zero or empty slices
        chunk = max(1, len(spec) // self.bands)

        # Calculate magnitude for each band by averaging spectrum chunks
        mags = [spec[i * chunk:(i + 1) * chunk].mean() for i in range(self.bands)]

        # Find the peak magnitude across all bands to normalize levels
        # If all mags are 0 (silence), use 1.0 to avoid division by zero
        peak = max(mags) or 1.0

        # Update each band's level
        for i, m in enumerate(mags):
            # Normalize the current magnitude relative to the peak
            val = m / peak
            # Update the level: if current value is higher, take it; otherwise, apply decay
            self.levels[i] = val if val > self.levels[i] else self.levels[i] * self.decay

    def clear(self):
        """
        Resets all equalizer bar levels to zero.
        """
        self.levels = [0.0] * self.bands

    def initializeGL(self):
        """
        Initializes OpenGL states for the widget.
        Called once by PyQt/OpenGL context.
        """
        glEnable(GL_BLEND) # Enable blending for transparent effects
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Standard alpha blending
        glClearColor(0, 0, 0, 1) # Set clear color to black (RGBA)

    def resizeGL(self, w: int, h: int):
        """
        Resizes the OpenGL viewport and projection matrix when the widget size changes.

        Args:
            w (int): New width of the widget.
            h (int): New height of the widget.
        """
        glViewport(0, 0, w, h) # Set the viewport to cover the entire widget
        glMatrixMode(GL_PROJECTION); glLoadIdentity() # Switch to projection matrix mode and reset
        # Set up an orthographic projection: (left, right, bottom, top, near, far)
        # Maps screen coordinates directly to OpenGL coordinates (0,0 is bottom-left)
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Switch back to modelview matrix mode and reset

    def paintGL(self):
        """
        Renders the equalizer bars using OpenGL.
        Called repeatedly by the QTimer via `update()`.
        """
        glClear(GL_COLOR_BUFFER_BIT) # Clear the color buffer with the clear color (black)

        w, h = self.width(), self.height() # Get current widget dimensions
        slot = w / self.bands # Calculate width for each bar slot (bar + gap)
        barw = slot * 0.9 # Calculate actual bar width (90% of slot, 10% for gap)

        # Iterate through each equalizer band level
        for i, lvl in enumerate(self.levels):
            # Set bar color (blueish-white based on level, with a minimum blue component)
            glColor4f(lvl, lvl, lvl * 0.8 + 0.2, 1) # R, G, B, Alpha (full opacity)

            x = i * slot # X position of the current bar
            barh = lvl * h # Height of the bar, proportional to its level and widget height

            # Draw the bar as a filled rectangle (GL_QUADS requires 4 vertices)
            glBegin(GL_QUADS)
            glVertex2f(x, 0)       # Bottom-left
            glVertex2f(x + barw, 0)    # Bottom-right
            glVertex2f(x + barw, barh) # Top-right
            glVertex2f(x, barh)    # Top-left
            glEnd()
