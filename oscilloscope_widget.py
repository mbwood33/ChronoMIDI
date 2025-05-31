# oscilloscope_widget.py
# Contains the Oscilloscope class for ChronoMIDI.

import numpy as np
import math
from collections import deque

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QOpenGLWidget

from OpenGL.GL import (
    glEnable, glBlendFunc, glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    glViewport, glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
    glGenBuffers, glBindBuffer, glBufferData, glDrawArrays,
    glEnableClientState, glDisableClientState, glVertexPointer, glColorPointer,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT, GL_LINE_STRIP,
    glLineWidth
)

# Try to import the Cython module. If it's not built yet or not in the path,
# this will raise an ImportError, which might be handled by the main script
# or build process.
try:
    import oscilloscope_computations
except ImportError:
    # This is a fallback or placeholder. Ideally, the main application
    # should ensure Cython modules are built and importable.
    # For now, we can define a dummy/stub if needed for basic loading,
    # or simply let the ImportError propagate if the main script handles it.
    print("Warning: oscilloscope_computations Cython module not found. Oscilloscope will not work.")
    # As a simple fallback, let's create a dummy module so the class can be defined
    # without crashing outright if the main program doesn't handle the import error.
    # This dummy won't do the actual computations, so the oscilloscope would be non-functional.
    class _dummy_oscilloscope_computations:
        def fill_trace_data_cython(*args, **kwargs):
            # Dummy function, does nothing
            pass
    oscilloscope_computations = _dummy_oscilloscope_computations()


class Oscilloscope(QOpenGLWidget):
    """
    A real-time oscilloscope visualizer with linear and circular modes.
    - Displays the latest block of audio as a time-domain line (white).
    - Ghosts of the waveform trace back with a rainbow gradient.
    - Modes: Linear (scrolling diagonally) and Circular (spiraling outwards).
    - Toggle mode with left-click.
    - Includes an "Edge Glow" effect for waveform emphasis.
    - Optimized with OpenGL Vertex Buffer Objects (VBOs) and pre-allocated NumPy arrays,
      with core computations offloaded to Cython for consistent 60 FPS.
    """
    LINEAR_MODE = 0    # Constant for linear display mode
    CIRCULAR_MODE = 1  # Constant for circular display mode

    def __init__(self, width=512, height=512, parent=None):
        """
        Initializes the Oscilloscope widget.

        Args:
            width (int): Fixed width of the oscilloscope.
            height (int): Fixed height of the oscilloscope.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setFixedSize(width, height) # Set a fixed size for the widget

        self.hue_offset = 0  # Starting hue for the rainbow gradient of ghosts
        self.trace_history = deque(maxlen=100) # Stores past audio traces for ghosting effect (max 100)

        self.audio_queue = deque() # Queue to hold incoming audio PCM blocks

        self.timer = QTimer(self) # Timer for updating the OpenGL view
        self.timer.timeout.connect(self.update) # Connect timeout to update (triggers paintGL)
        self.timer.start(1000 // 60) # Start timer to update at ~60 FPS (1000ms / 60 frames)

        self.current_mode = self.LINEAR_MODE # Initial display mode is linear

        self.vbo_vertex = None # OpenGL Vertex Buffer Object for vertex coordinates
        self.vbo_color = None  # OpenGL Vertex Buffer Object for vertex colors

        # --- Pre-allocate max possible NumPy arrays for VBO data ---
        # This avoids reallocations during runtime for performance
        max_lines_to_draw = (self.trace_history.maxlen + 1) * 2 # 100 ghosts * 2 (glow+core) + current trace * 2 (glow+core)
        max_points_per_line = width // 2 # Rough estimate of points per trace based on width

        self.max_total_vertices = max_lines_to_draw * max_points_per_line

        # Pre-allocate the NumPy arrays that Cython will write into
        # These arrays will directly back the OpenGL VBOs
        self.all_vertices_buffer = np.zeros((self.max_total_vertices, 2), dtype=np.float32) # (x, y) coordinates
        self.all_colors_buffer = np.zeros((self.max_total_vertices, 4), dtype=np.float32)   # (r, g, b, a) colors

    def initializeGL(self):
        """
        Initializes OpenGL states and generates VBOs.
        Called once by PyQt/OpenGL context when the widget is created.
        """
        glEnable(GL_BLEND) # Enable blending for transparency
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Standard alpha blending
        glClearColor(0, 0, 0, 1) # Set clear color to black (RGBA)

        # Generate OpenGL Vertex Buffer Objects
        self.vbo_vertex = glGenBuffers(1)
        self.vbo_color = glGenBuffers(1)

    def resizeGL(self, w: int, h: int):
        """
        Resizes the OpenGL viewport and sets up the orthographic projection.

        Args:
            w (int): New width of the widget.
            h (int): New height of the widget.
        """
        glViewport(0, 0, w, h) # Set the viewport to the new widget dimensions
        glMatrixMode(GL_PROJECTION); glLoadIdentity() # Switch to projection matrix mode and reset
        # Set up an orthographic projection where (0,0) is bottom-left and (w,h) is top-right
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Switch back to modelview matrix mode and reset

    def mousePressEvent(self, event):
        """
        Handles mouse press events, specifically toggling the display mode
        on a left-click.

        Args:
            event (QMouseEvent): The mouse event object.
        """
        if event.button() == Qt.LeftButton:
            # Toggle between LINEAR_MODE (0) and CIRCULAR_MODE (1)
            self.current_mode = (self.current_mode + 1) % 2
            self.update() # Request a repaint to immediately show the new mode

    def paintGL(self):
        """
        Renders the oscilloscope waveform(s) using OpenGL and VBOs.
        This method isalled by the QTimer via `update()`.
        """
        glClear(GL_COLOR_BUFFER_BIT) # Clear the screen with the background color (black)

        w, h = self.width(), self.height() # Get current widget width and height
        center_x, center_y = w / 2.0, h / 2.0 # Calculate the center of the widget

        # Process all pending audio data from the queue
        while self.audio_queue:
            pcm = self.audio_queue.popleft() # Get the next audio block
            # Convert stereo PCM to mono and normalize to -1.0 to 1.0 range
            mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
            self.trace_history.append(mono) # Add the processed mono data to history

        num_traces = len(self.trace_history) # Get the current number of traces (including ghosts)
        if num_traces == 0:
            return # Nothing to draw if no audio data has been received yet

        # Reference length for calculating points to draw and sample step
        # Uses the latest trace as the reference
        mono_data_len_ref = len(self.trace_history[-1])

        # Calculate hue step for rainbow gradient of ghosts
        hue_step_per_ghost = 360 / max(1, num_traces + 10) # Added +10 to ensure smoother spread even with few ghosts

        # Parameters for linear mode scrolling effect
        max_linear_scroll_dist = 100

        # Parameters for circular mode spiraling effect
        max_spiral_radius_offset = min(w, h) * 0.45 # Max radius for spiral, 45% of min dimension
        spiral_angle_offset_per_ghost = 0.05 * math.pi # Angle offset for each ghost in circular mode

        # Determine how many points to draw per line based on widget width
        # This downsamples the audio data to fit the screen
        points_to_draw = w // 2
        # Calculate the step size for sampling points from the mono audio data
        sample_step = max(1, mono_data_len_ref // points_to_draw)

        # --- GLOW PARAMETERS ---
        glow_color_base = QColor(200, 200, 255) # Base color for the glow (light bluish-white)
        glow_alpha_factor = 0.25 # Alpha multiplier for glow (makes it semi-transparent)
        glow_width_linear = 3.0 # Line width for glow in linear mode
        glow_width_circular = 4.0 # Line width for glow in circular mode

        # Pixel offsets for glow effect (slight shift to create a "halo")
        glow_offset_x = 0.5
        glow_offset_y = 0.5
        glow_radius_offset_amount = 2.0 # Radius increase for glow in circular mode

        draw_commands = [] # List to store (start_index, num_points, line_width) for glDrawArrays calls
        current_vertex_offset = 0 # Tracks current position in the pre-allocated VBO buffers

        # --- Populate Data for Ghosts (Glow + Core) using Cython ---
        # Iterate through the history of traces to draw older "ghost" waveforms
        for i, mono_data in enumerate(self.trace_history):
            # Calculate hue for the current ghost (rainbow effect)
            ghost_hue = (self.hue_offset + i * hue_step_per_ghost) % 360
            # Calculate alpha value for the ghost (fades out older ghosts)
            # Alpha increases quadratically with age to make newer ghosts more visible
            alpha_val = int(255 * (i / max(1, num_traces))**2 * 0.6 + 5)
            alpha_val = min(255, max(0, alpha_val)) # Clamp alpha between 0 and 255

            qt_color = QColor.fromHsv(int(ghost_hue), 220, 255, alpha_val) # Create QColor from HSV

            # Pre-calculate float color components for Cython function
            glow_r, glow_g, glow_b = glow_color_base.redF(), glow_color_base.greenF(), glow_color_base.blueF()
            glow_a_current = qt_color.alphaF() * glow_alpha_factor # Apply glow specific alpha factor

            core_r, core_g, core_b, core_a = qt_color.redF(), qt_color.greenF(), qt_color.blueF(), qt_color.alphaF()

            # --- GLOW Pass for the current ghost ---
            start_index_glow = current_vertex_offset
            # Call Cython function to fill vertex and color data for the glow trace
            oscilloscope_computations.fill_trace_data_cython(
                mono_data, self.all_vertices_buffer, self.all_colors_buffer,
                start_index_glow,
                float(w), float(h), float(center_x), float(center_y),
                points_to_draw, sample_step,
                self.current_mode,
                i, num_traces, # `i` and `num_traces` determine ghosting offset/fade
                max_linear_scroll_dist,
                max_spiral_radius_offset, spiral_angle_offset_per_ghost,
                glow_offset_x, glow_offset_y, glow_radius_offset_amount,
                glow_r, glow_g, glow_b, glow_a_current,
                True, # is_glow_pass = True
                False # is_current_trace = False (this is a ghost)
            )
            # Add draw command for this glow trace
            draw_commands.append((start_index_glow, points_to_draw, glow_width_linear if self.current_mode == self.LINEAR_MODE else glow_width_circular))
            current_vertex_offset += points_to_draw # Advance offset for next trace

            # --- CORE Pass for the current ghost ---
            start_index_core = current_vertex_offset
            # Call Cython function to fill vertex and color data for the core trace
            oscilloscope_computations.fill_trace_data_cython(
                mono_data, self.all_vertices_buffer, self.all_colors_buffer,
                start_index_core,
                float(w), float(h), float(center_x), float(center_y),
                points_to_draw, sample_step,
                self.current_mode,
                i, num_traces,
                max_linear_scroll_dist,
                max_spiral_radius_offset, spiral_angle_offset_per_ghost,
                glow_offset_x, glow_offset_y, glow_radius_offset_amount, # Not used for core, but passed
                core_r, core_g, core_b, core_a,
                False, # is_glow_pass = False
                False # is_current_trace = False (this is a ghost)
            )
            # Add draw command for this core trace
            draw_commands.append((start_index_core, points_to_draw, 1.0)) # Core width is typically 1.0
            current_vertex_offset += points_to_draw # Advance offset

        # Update base hue for the next frame's "newest" ghost, creating a continuous shift
        self.hue_offset = (self.hue_offset + 5) % 360

        # --- Populate Data for CURRENT (Newest) Trace (Glow + Core) using Cython ---
        # The most recent waveform is drawn distinctly (pure white core, brighter glow)
        current_mono_data = self.trace_history[-1] # Get the latest audio trace

        # --- CURRENT GLOW Pass (always full alpha, white-ish glow) ---
        start_index_current_glow = current_vertex_offset
        oscilloscope_computations.fill_trace_data_cython(
            current_mono_data, self.all_vertices_buffer, self.all_colors_buffer,
            start_index_current_glow,
            float(w), float(h), float(center_x), float(center_y),
            points_to_draw, sample_step,
            self.current_mode,
            0, 1, # i, num_traces - these values are ignored by Cython when is_current_trace is True
            max_linear_scroll_dist, # Ignored for current trace
            max_spiral_radius_offset, spiral_angle_offset_per_ghost,
            glow_offset_x, glow_offset_y, glow_radius_offset_amount,
            glow_color_base.redF(), glow_color_base.greenF(), glow_color_base.blueF(), 1.0, # Full alpha for current glow
            True, # is_glow_pass = True
            True # is_current_trace = True (this is the live trace)
        )
        # Add draw command for the current glow trace, slightly wider than ghosts
        draw_commands.append((start_index_current_glow, points_to_draw, glow_width_linear + 1.0))
        current_vertex_offset += points_to_draw

        # --- CURRENT CORE Pass (pure white core) ---
        start_index_current_core = current_vertex_offset
        oscilloscope_computations.fill_trace_data_cython(
            current_mono_data, self.all_vertices_buffer, self.all_colors_buffer,
            start_index_current_core,
            float(w), float(h), float(center_x), float(center_y),
            points_to_draw, sample_step,
            self.current_mode,
            0, 1, # Ignored
            max_linear_scroll_dist, # Ignored
            max_spiral_radius_offset, spiral_angle_offset_per_ghost, # Ignored
            glow_offset_x, glow_offset_y, glow_radius_offset_amount, # Ignored
            1.0, 1.0, 1.0, 1.0, # Pure white, full opacity core
            False, # is_glow_pass = False
            True # is_current_trace = True (this is the live trace)
        )
        # Add draw command for this core trace, slightly wider than ghosts
        draw_commands.append((start_index_current_core, points_to_draw, 1.5))
        current_vertex_offset += points_to_draw


        # --- Send ALL data to GPU (only two glBufferData calls!) ---
        # This is a key optimization: transfer all vertex and color data in one go.
        # This reduces CPU-GPU communication overhead significantly.

        # Bind the vertex VBO and transfer all vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
        glBufferData(GL_ARRAY_BUFFER, self.all_vertices_buffer.nbytes, self.all_vertices_buffer, GL_DYNAMIC_DRAW)

        # Bind the color VBO and transfer all color data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
        glBufferData(GL_ARRAY_BUFFER, self.all_colors_buffer.nbytes, self.all_colors_buffer, GL_DYNAMIC_DRAW)

        # --- Enable Client States and Set Pointers (once per paintGL) ---
        # These operations configure OpenGL to use the data from the bound VBOs.
        glEnableClientState(GL_VERTEX_ARRAY) # Enable vertex array processing
        glEnableClientState(GL_COLOR_ARRAY)  # Enable color array processing

        # Point OpenGL to the vertex data in the VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
        glVertexPointer(2, GL_FLOAT, 0, None) # 2 components (x,y), float type, tightly packed, no offset

        # Point OpenGL to the color data in the VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
        glColorPointer(4, GL_FLOAT, 0, None) # 4 components (r,g,b,a), float type, tightly packed, no offset

        # --- Execute ALL Draw Calls (only glDrawArrays calls) ---
        # Iterate through the list of draw commands generated earlier.
        # Each command specifies a segment of the VBOs to draw.
        for start_idx, num_pts, line_w in draw_commands:
            glLineWidth(line_w) # Set the line thickness for the current trace
            glDrawArrays(GL_LINE_STRIP, start_idx, num_pts)

        # --- Disable Client States ---
        # Clean up OpenGL states to avoid interfering with other drawing operations (if any).
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
