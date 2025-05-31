# kaleidoscope_widget.py
# Contains the KaleidoscopeVisualizerGLWidget class for ChronoMIDI.

import numpy as np
import math
import random
from collections import deque

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    glEnable, glBlendFunc, glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    glViewport, glMatrixMode, glLoadIdentity, GL_PROJECTION, GL_MODELVIEW,
    glPushMatrix, glTranslatef, glRotatef, glPopMatrix, glLineWidth,
    glBegin, glEnd, glVertex2f, GL_QUADS,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_STRIP,
    GL_LINE_SMOOTH, GL_NICEST, glHint, GL_LINE_SMOOTH_HINT,
    glGenBuffers, glBindBuffer, glBufferData, glDrawArrays,
    glEnableClientState, glDisableClientState, glVertexPointer, glColorPointer,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT
)
from OpenGL.GLU import gluPerspective

try:
    import kaleidoscope_computations
except ImportError:
    print("Warning: kaleidoscope_computations Cython module not found. Kaleidoscope will not work.")
    class _dummy_kaleidoscope_computations:
        def fill_kaleidoscope_data_cython(*args, **kwargs):
            # Dummy function, returns 0 vertices added and an empty list of sub_commands
            return 0, []
    kaleidoscope_computations = _dummy_kaleidoscope_computations()


class KaleidoscopeVisualizerGLWidget(QOpenGLWidget):
    """
    An OpenGL widget for displaying a real-time, audio-reactive kaleidoscope visualization.
    Generates procedural patterns, applies audio-driven transformations (rotation, oscillation),
    and features a basic particle system.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_amplitude = 0.0 # Current audio amplitude (normalized)
        self.rotation_angle = 0.0    # Current rotation angle for the kaleidoscope
        self.hue_offset = 0.0        # Current hue for color cycling
        self.particles = []          # List of active particles
        self.oscillation_mode = 0    # 0: Linear, 1: Circular (for line oscillation)
        self.history = deque(maxlen=6) # Reduced maxlen for fewer ghosts
        self.focal_length = 500.0 # For perspective projection
        self.frame_count = 0 # Initialize frame counter for oscillation phase

        # VBOs for kaleidoscope lines
        self.vbo_kaleidoscope_vertex = None
        self.vbo_kaleidoscope_color = None

        # Pre-allocate max possible NumPy arrays for VBO data
        self.max_kaleidoscope_vertices = 20480
        self.kaleidoscope_vertices_buffer = np.zeros((self.max_kaleidoscope_vertices, 2), dtype=np.float32) # (x, y)
        self.kaleidoscope_colors_buffer = np.zeros((self.max_kaleidoscope_vertices, 4), dtype=np.float32)   # (r, g, b, a)

        QTimer(self, timeout=self.update, interval=1000 // 60).start() # ~60 FPS

    def push_audio(self, pcm: np.ndarray):
        """
        Processes a block of PCM audio data to update visualizer parameters.
        Calculates amplitude and updates internal state.

        Args:
            pcm (np.ndarray): Stereo PCM audio data (e.g., int16 or float).
        """
        mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
        rms_amplitude = np.sqrt(np.mean(mono**2))
        self.current_amplitude = rms_amplitude * 5.0
        self.current_amplitude = max(0.0, min(1.0, self.current_amplitude))
        self.rotation_angle += self.current_amplitude * 1.0
        self.rotation_angle %= 360
        self.hue_offset = (self.hue_offset + self.current_amplitude * 1.0) % 360

        if self.current_amplitude > 0.3 and random.random() < self.current_amplitude * 0.7:
            num_new_particles = int(self.current_amplitude * 20)
            for _ in range(num_new_particles):
                x = random.uniform(-20, 20)
                y = random.uniform(-20, 20)
                vx = random.uniform(-4, 4) * self.current_amplitude
                vy = random.uniform(-4, 4) * self.current_amplitude
                particle_color = QColor.fromHsv(int(self.hue_offset), 150, 255).getRgbF()
                self.particles.append({
                    'x': x, 'y': y, 'vx': vx, 'vy': vy,
                    'lifetime': 90, 'initial_lifetime': 90, 'color': particle_color,
                    'initial_size': 1.0
                })
        self.history.append((self.rotation_angle, self.hue_offset, self.current_amplitude))

    def clear_particles(self):
        """
        Clears all active particles from the visualizer.
        """
        self.particles = []

    def initializeGL(self):
        """
        Initializes OpenGL states for the widget.
        """
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 1) # Black background
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        self.vbo_kaleidoscope_vertex = glGenBuffers(1)
        self.vbo_kaleidoscope_color = glGenBuffers(1)

    def resizeGL(self, w: int, h: int):
        """
        Resizes the OpenGL viewport and sets up the perspective projection.
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(1,h), 0.1, 1000.0) # Added max(1,h) to avoid division by zero if h is 0
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousePressEvent(self, event):
        """
        Toggles the oscillation mode on mouse click.
        """
        if event.button() == Qt.LeftButton: # Use Qt.LeftButton
            self.oscillation_mode = (self.oscillation_mode + 1) % 2
            self.update()

    def paintGL(self):
        """
        Renders the kaleidoscope pattern and particles.
        """
        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self.frame_count += 1
        current_kaleidoscope_vertex_offset = 0
        kaleidoscope_draw_commands = []
        history_copy = list(self.history)

        for i, (hist_rot_angle, hist_hue_offset, hist_amplitude) in enumerate(history_copy):
            z_offset = -50.0 - (len(history_copy) - 1 - i) * 10.0
            normalized_age = i / max(1, len(history_copy) - 1) if len(history_copy) > 1 else 0 # Avoid division by zero for single item
            alpha_fade = normalized_age ** 2 * 0.7 + 0.05
            alpha_fade = min(1.0, max(0.0, alpha_fade))
            strobe_val_for_ghost = hist_amplitude * 0.9 + 0.1

            total_vertices_added, sub_commands = kaleidoscope_computations.fill_kaleidoscope_data_cython(
                self.kaleidoscope_vertices_buffer, self.kaleidoscope_colors_buffer,
                current_kaleidoscope_vertex_offset,
                hist_rot_angle, hist_hue_offset, hist_amplitude,
                False, strobe_val_for_ghost, alpha_fade,
                self.frame_count, self.oscillation_mode
            )
            for rel_start_idx, num_pts, line_w in sub_commands:
                kaleidoscope_draw_commands.append((current_kaleidoscope_vertex_offset + rel_start_idx, num_pts, line_w, z_offset, hist_rot_angle))
            current_kaleidoscope_vertex_offset += total_vertices_added

        strobe_val_for_current = self.current_amplitude * 0.9 + 0.1
        total_vertices_added, sub_commands = kaleidoscope_computations.fill_kaleidoscope_data_cython(
            self.kaleidoscope_vertices_buffer, self.kaleidoscope_colors_buffer,
            current_kaleidoscope_vertex_offset,
            self.rotation_angle, self.hue_offset, self.current_amplitude,
            True, strobe_val_for_current, 1.0,
            self.frame_count, self.oscillation_mode
        )
        for rel_start_idx, num_pts, line_w in sub_commands:
            kaleidoscope_draw_commands.append((current_kaleidoscope_vertex_offset + rel_start_idx, num_pts, line_w, 0.0, self.rotation_angle))
        current_kaleidoscope_vertex_offset += total_vertices_added

        if current_kaleidoscope_vertex_offset > 0: # Only buffer data if there's something to draw
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_vertex)
            glBufferData(GL_ARRAY_BUFFER, self.kaleidoscope_vertices_buffer[:current_kaleidoscope_vertex_offset].nbytes, self.kaleidoscope_vertices_buffer[:current_kaleidoscope_vertex_offset], GL_DYNAMIC_DRAW)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_color)
            glBufferData(GL_ARRAY_BUFFER, self.kaleidoscope_colors_buffer[:current_kaleidoscope_vertex_offset].nbytes, self.kaleidoscope_colors_buffer[:current_kaleidoscope_vertex_offset], GL_DYNAMIC_DRAW)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_vertex)
            glVertexPointer(2, GL_FLOAT, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_color)
            glColorPointer(4, GL_FLOAT, 0, None)

            for start_idx, num_pts, line_w, z_offset_for_draw, rotation_angle_for_draw in kaleidoscope_draw_commands:
                if num_pts > 0: # Ensure we only draw if there are points
                    glPushMatrix()
                    glTranslatef(0, 0, z_offset_for_draw)
                    glRotatef(rotation_angle_for_draw, 0, 0, 1)
                    glLineWidth(line_w)
                    glDrawArrays(GL_LINE_STRIP, start_idx, num_pts)
                    glPopMatrix()

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)

        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                new_particles.append(p)
                normalized_lifetime = p['lifetime'] / p['initial_lifetime']
                alpha = math.sin(normalized_lifetime * math.pi) * 1.0
                alpha = max(0.0, min(1.0, alpha))

                glPushMatrix()
                glTranslatef(p['x'], p['y'], -200.0)
                glRotatef(self.rotation_angle, 0, 0, 1)

                strobe_pow_particle = self.current_amplitude ** 6.0
                desaturation_factor_particle = strobe_pow_particle
                target_saturation_hsv_particle = int((1.0 - desaturation_factor_particle) * 255)
                target_saturation_hsv_particle = max(0, min(255, target_saturation_hsv_particle))
                target_value_hsv_particle = int(150 + strobe_pow_particle * 105.0)
                target_value_hsv_particle = max(150, min(255, target_value_hsv_particle))

                original_particle_qcolor = QColor.fromRgbF(*p['color'])
                original_h = original_particle_qcolor.hue()

                particle_strobe_qcolor = QColor.fromHsv(original_h if original_h != -1 else 0, target_saturation_hsv_particle, target_value_hsv_particle) # Ensure hue is valid
                strobe_r, strobe_g, strobe_b, _ = particle_strobe_qcolor.getRgbF()

                glColor4f(strobe_r, strobe_g, strobe_b, alpha * 0.7)
                glBegin(GL_QUADS)
                aura_size = p['initial_size'] * 3.0
                glVertex2f(-aura_size, -aura_size)
                glVertex2f(aura_size, -aura_size)
                glVertex2f(aura_size, aura_size)
                glVertex2f(-aura_size, aura_size)
                glEnd()

                glColor4f(strobe_r, strobe_g, strobe_b, alpha)
                glBegin(GL_QUADS)
                size = p['initial_size'] * (normalized_lifetime * 0.8 + 0.2)
                glVertex2f(-size, -size)
                glVertex2f(size, -size)
                glVertex2f(size, size)
                glVertex2f(-size, size)
                glEnd()
                glPopMatrix()
        self.particles = new_particles
