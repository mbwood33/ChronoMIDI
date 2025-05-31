#!/usr/bin/env python3
# chronomidi_gui.py – ChronoMIDI GUI/playback/visualizer

"""
ChronoMIDI v0.2.2 (5/31/2025) - A real-time MIDI playback and visualization application
Started 5/19/2025
Developed by Michael Wood (github.com/mbwood33) (with coding assistance from Google Gemini + OpenAI ChatGPT)
Kaleidoscope Visualizor idea contribution by Rayce Hinkle (github.com/Wrhinkle)

This script implements a PyQt5-based graphical user interface for playing MIDI files,
visualizing audio output with an equalizer and oscilloscope, and analyzing MIDI event data.
It integrates with external libraries like mido for MIDI parsing, numpy and sounddevice
for audio processing, and pyfluidsynth for MIDI synthesis. OpenGL is used for
high-performance visualizations.

Features include:
🎼 MIDI playback and MIDI event analysis/display
📊 Real-time Equalizer visualization
🌀 Real-time Oscilloscope and Kaleidoscope visualizations
💻 Real-time MIDI event analysis and display
💾 MIDI-to-MP3 audio export using currently loaded Soundfont
"""

import sys
import os
import subprocess
import random
from collections import deque
import math

# MIDI and Audio Libraries
import mido  # For MIDI file parsing and event handling
import numpy as np  # For numerical operations, especially with audio data
import sounddevice as sd  # For audio output to speakers
from fluidsynth import Synth  # The Python binding for FluidSynth, a software synthesizer

# Custom modules for optimized computations (assumed to be Cython modules)
import oscilloscope_computations    # For Oscilloscope Visualizer computations
import kaleidoscope_computations    # For Kaleidoscope Visualizer computations

# PyQt5 Core Modules
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal,  # Core utilities, timers, and signals for event handling
    QAbstractTableModel, QModelIndex, QVariant, QPointF  # Data models for table views
)
# PyQt5 GUI Modules
from PyQt5.QtGui import (
    QColor, QFont, QPalette, QFontDatabase,  # Styling: colors, fonts, palettes
    QImage, QPainter, QPen, QPolygonF, QIcon # Graphics: images, drawing, icon support
)
# PyQt5 Widget Modules
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, # Main application, windows, dialogs
    QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QTableView, # Layouts, labels, tabs, table views
    QPushButton, QGroupBox, QFormLayout, QMessageBox, # Buttons, grouping, form layouts, message boxes
    QAbstractItemView, QHeaderView, QStyledItemDelegate, QOpenGLWidget # Table view components, OpenGL widget
)
# OpenGL Bindings (PyOpenGL)
from OpenGL.GL import (
    glViewport, glMatrixMode, glLoadIdentity, glOrtho, # Basic OpenGL matrix and viewport setup
    glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_PROJECTION, GL_MODELVIEW, # Clearing and matrix modes
    glEnable, glBlendFunc, # Enabling capabilities like blending
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_STRIP, # Blending functions and primitive types
    glLineWidth, # Setting line thickness

    # VBO related imports (used by Oscilloscope for performance)
    glGenBuffers, glBindBuffer, glBufferData, glDrawArrays, # Buffer generation, binding, data transfer, drawing
    glEnableClientState, glDisableClientState, # Enabling/disabling client-side capabilities
    glVertexPointer, glColorPointer, # Setting pointers to vertex and color data in buffers
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, # Buffer types and usage patterns
    GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT, # Array types and data types

    # Added for EqualizerGLWidget (using immediate mode, though VBOs are generally preferred)
    glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS,    # Immediate mode commands for drawing colored quads

    glPushMatrix, glTranslatef, glRotatef, glPopMatrix,  # Matrix stack operations for transformations
    glScalef, # Matrix operations for scaling and loading matrices
    GL_LINE_SMOOTH, GL_NICEST, glHint, GL_LINE_SMOOTH_HINT # For antialiasing lines and hint
)
# OpenGL Utility Library (for gluPerspective)
from OpenGL.GLU import gluPerspective


# ─── Helpers ───────────────────────────────────────────────────────────────

# List of standard MIDI note names for display purposes
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_note_to_name(n: int) -> str:
    """
    Converts a MIDI note number (0-127) to its musical name (e.g., C4, A#3).

    Args:
        n (int): The MIDI note number.

    Returns:
        str: The musical name of the note.
    """
    # Calculate note name (C, C#, D, etc.) using modulo 12
    # Calculate octave number: MIDI note 0 is C-1, so (n // 12) - 1
    return NOTE_NAMES[n % 12] + str((n // 12) - 1)

# Dictionary mapping MIDI Control Change (CC) numbers to human-readable names
CONTROL_CHANGE_NAMES = {
    0: 'Bank Select', 1: 'Modulation Wheel', 2: 'Breath Controller',
    4: 'Foot Controller', 5: 'Portamento Time', 6: 'Data Entry MSB',
    7: 'Channel Volume', 8: 'Balance', 10: 'Pan',
    11: 'Expression Controller', 12: 'Effect Control 1', 13: 'Effect Control 2',
    64: 'Sustain Pedal', 65: 'Portamento On/Off', 66: 'Sostenuto Pedal',
    67: 'Soft Pedal', 68: 'Legato Footswitch', 69: 'Hold 2',
    70: 'Sound Controller 1', 71: 'Sound Controller 2',
    72: 'Sound Controller 3', 73: 'Sound Controller 4',
    74: 'Sound Controller 5', 75: 'Sound Controller 6',
    76: 'Sound Controller 7', 77: 'Sound Controller 8',
    78: 'Sound Controller 9', 79: 'Sound Controller 10',
    80: 'General Purpose Controller 1', 81: 'General Purpose Controller 2',
    82: 'General Purpose Controller 3', 83: 'General Purpose Controller 4',
    84: 'Portamento Control',
    91: 'Effects 1 Depth (Reverb)', 92: 'Effects 2 Depth (Tremolo)',
    93: 'Effects 3 Depth (Chorus)', 94: 'Effects 4 Depth (Celeste)',
    95: 'Effects 5 Depth (Phaser)',
    96: 'Data Increment', 97: 'Data Decrement', 98: 'NRPN LSB',
    99: 'NRPN MSB', 100: 'RPN LSB', 101: 'RPN MSB',
    121: 'Reset All Controllers', 122: 'Local Control On/Off',
    123: 'All Notes Off', 124: 'Omni Mode Off', 125: 'Omni Mode On',
    126: 'Mono Mode On', 127: 'Poly Mode On',
}

def get_time_signature_at_tick(absolute_tick: int, time_signature_changes: list) -> tuple:
    """
    Determines the active time signature at a given absolute MIDI tick.

    Args:
        absolute_tick (int): The absolute tick position in the MIDI file.
        time_signature_changes (list): A sorted list of (tick, numerator, denominator, cumulative_measures) tuples.

    Returns:
        tuple: A (numerator, denominator) tuple representing the active time signature.
    """
    active_ts = (4, 4) # Default time signature
    for ts_tick, num, den, _ in time_signature_changes:
        if absolute_tick >= ts_tick:
            active_ts = (num, den)
        else:
            break # List is sorted, so we've passed the relevant changes
    return active_ts

def calculate_beat_measure(absolute_tick: int, ticks_per_beat: int, time_signature_changes: list) -> tuple:
    """
    Calculates the cumulative measure and beat number for a given absolute tick,
    using pre-calculated cumulative measures from time_signature_changes.

    Args:
        absolute_tick (int): The absolute tick position.
        ticks_per_beat (int): The MIDI file's ticks per beat (ticks per quarter note).
        time_signature_changes (list): A sorted list of (tick, numerator, denominator, cumulative_measures_at_start_of_this_ts_block) tuples.
            This list is expected to be sorted by tick and include a (0, 4, 4, 0) entry if no
            time signature is explicitly set at the beginning.

    Returns:
        tuple: A (measure, beat_in_measure) tuple, where measure is the cumulative measure
               from the beginning of the MIDI file.
    """
    # Find the most recent time signature change that applies to absolute_tick
    active_ts_info = time_signature_changes[0] # Default to the first entry (tick 0)
    for ts_info in time_signature_changes:
        ts_change_tick = ts_info[0]
        if absolute_tick >= ts_change_tick:
            active_ts_info = ts_info
        else:
            break # List is sorted, so we've passed the relevant changes

    active_ts_tick = active_ts_info[0]
    active_ts_numerator = active_ts_info[1]
    active_ts_denominator = active_ts_info[2]
    cumulative_measures_at_start_of_block = active_ts_info[3] # This is the key: full measures before this block

    # Calculate ticks within the current time signature block
    ticks_in_current_block = absolute_tick - active_ts_tick

    # Convert ticks in current block to quarter notes
    quarter_notes_in_current_block = ticks_in_current_block / ticks_per_beat

    # Convert quarter notes to the actual beat unit of the current time signature
    # The '4' in (active_ts_denominator / 4.0) represents a quarter note as the base unit.
    # For example, if active_ts_denominator is 8 (for 7/8), conversion_factor is 2.0.
    # This means 1 quarter note = 2 eighth notes.
    conversion_factor = active_ts_denominator / 4.0
    beats_in_current_block_in_target_unit = quarter_notes_in_current_block * conversion_factor

    # Calculate measure and beat within this current time signature block
    beats_per_measure_in_target_unit = active_ts_numerator
    
    if beats_per_measure_in_target_unit == 0: # Avoid division by zero
        measure_offset_in_block = 0
        beat_in_measure = 0.0
    else:
        measure_offset_in_block = int(beats_in_current_block_in_target_unit // beats_per_measure_in_target_unit)
        beat_in_measure = beats_in_current_block_in_target_unit % beats_per_measure_in_target_unit

    # Total cumulative measures = measures accumulated before this block + measures within this block
    # We add 1 for 1-indexing of the measure number.
    total_measure = cumulative_measures_at_start_of_block + measure_offset_in_block + 1
    
    return total_measure, beat_in_measure


class EditDelegate(QStyledItemDelegate):
    """
    A custom item delegate for QTableView that applies specific styling to
    QLineEdit editors when a cell is being edited.
    """
    def createEditor(self, parent, option, index):
        """
        Creates and returns a QLineEdit editor for the specified index,
        applying custom dark theme styling.

        Args:
            parent (QWidget): The parent widget (the table view).
            option (QStyleOptionViewItem): Styling options.
            index (QModelIndex): The model index of the item being edited.

        Returns:
            QLineEdit: The created editor with custom stylesheet.
        """
        # Call the base class to create the default editor (a QLineEdit for strings)
        e = super().createEditor(parent, option, index)
        # Apply custom CSS styling for dark theme consistency
        e.setStyleSheet(
            "QLineEdit{background:#444;color:white;}"
            "QLineEdit{selection-background-color:#666;"
            " selection-color:white;}")
        return e

class PandasModel(QAbstractTableModel):
    """
    A custom table model that wraps a pandas DataFrame, allowing it to be
    displayed in a PyQt QTableView. This class is for general DataFrame display,
    though `EventsModel` is used specifically for MIDI events.
    """
    def __init__(self, data):
        """
        Initializes the model with a pandas DataFrame.

        Args:
            data (pandas.DataFrame): The DataFrame to display.
        """
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        """
        Returns the number of rows in the table.
        """
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        """
        Returns the number of columns in the table.
        """
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """
        Returns the data for a given index and role.

        Args:
            index (QModelIndex): The index of the cell.
            role (Qt.ItemDataRole): The role of the data requested.

        Returns:
            QVariant: The data for the specified role.
        """
        if not index.isValid():
            return QVariant() # Return invalid variant for invalid indices
        if role == Qt.DisplayRole:
            # Return the string representation of the data at the given row and column
            return str(self._data.iloc[index.row(), index.column()])
        return QVariant() # Return invalid variant for unsupported roles

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Returns the header data for rows or columns.
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole: # Fixed condition
            # Return column names for horizontal headers
            return str(self._data.columns[section])
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            # Return row index for vertical headers
            return str(self._data.index[section])
        return QVariant()

class ReadOnlyDelegate(QStyledItemDelegate):
    """
    A custom item delegate that prevents editing of cells in a QTableView.
    """
    def createEditor(self, parent, option, index):
        """
        Overrides the createEditor method to always return None, effectively
        making cells non-editable.

        Args:
            parent (QWidget): The parent widget.
            option (QStyleOptionViewItem): Styling options.
            index (QModelIndex): The model index.

        Returns:
            None: Always returns None to prevent editor creation.
        """
        return None # Prevents editing

# ─── OpenGL Equalizer Widget ──────────────────────────────────────────

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


# ─── OpenGL Oscilloscope Widget (REWRITTEN) ──────────────────────────────

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
        for start_idx, num_pts, line_width in draw_commands:
            glLineWidth(line_width) # Set the line thickness for the current trace
            glDrawArrays(GL_LINE_STRIP, start_idx, num_pts)

        # --- Disable Client States ---
        # Clean up OpenGL states to avoid interfering with other drawing operations (if any).
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# ─── Visualizer Window ──────────────────────────────────────────────────

class VisualizerWindow(QMainWindow):
    """
    A separate QMainWindow that hosts the Oscilloscope visualization.
    It provides a dedicated window for the oscilloscope.
    """
    def __init__(self, sr):
        """
        Initializes the VisualizerWindow.

        Args:
            sr (int): Sample rate of the audio, passed to the Oscilloscope.
        """
        super().__init__()
        self.setWindowTitle("Oscilloscope Visualizer")
        
        # Create an instance of the Oscilloscope widget
        self.osc = Oscilloscope(width=512, height=512)
        
        # Expose the oscilloscope's audio queue to the main ChronoMIDI window
        # This allows ChronoMIDI to push audio data directly to the visualizer
        self.audio_queue = self.osc.audio_queue

        # Set up the central widget and layout
        cw = QWidget() # Create a central widget
        layout = QVBoxLayout(cw) # Create a vertical layout for the central widget
        layout.addWidget(self.osc) # Add the oscilloscope to the layout
        self.setCentralWidget(cw) # Set the central widget of the QMainWindow
        self.resize(532, 550) # Set a fixed size for the visualizer window (slightly larger than oscilloscope)

# ─── Kaleidoscope Visualizer Widget ───────────────────────────────────────

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
        # Max vertices calculation: 12 segments * 2 (horiz/vert lines) * 11 points/line * 11 points/line * 7 total patterns (6 ghosts + 1 current)
        # Each segment draws (num_lines+1) horizontal lines and (num_lines+1) vertical lines.
        # Each line has (num_lines+1) points.
        # So, 12 * 2 * (10+1) * (10+1) * (6+1) = 12 * 2 * 11 * 11 * 7 = 2904 * 7 = 20328 vertices.
        # Using a slightly larger power of 2 for safety.
        self.max_kaleidoscope_vertices = 20480
        self.kaleidoscope_vertices_buffer = np.zeros((self.max_kaleidoscope_vertices, 2), dtype=np.float32) # (x, y)
        self.kaleidoscope_colors_buffer = np.zeros((self.max_kaleidoscope_vertices, 4), dtype=np.float32)   # (r, g, b, a)


        # QTimer to trigger updates for smooth animation
        QTimer(self, timeout=self.update, interval=1000 // 60).start() # ~60 FPS

    def push_audio(self, pcm: np.ndarray):
        """
        Processes a block of PCM audio data to update visualizer parameters.
        Calculates amplitude and updates internal state.

        Args:
            pcm (np.ndarray): Stereo PCM audio data (e.g., int16 or float).
        """
        mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
        # Calculate RMS amplitude for a smoother response
        rms_amplitude = np.sqrt(np.mean(mono**2))
        self.current_amplitude = rms_amplitude * 5.0 # Amplify for stronger visual effect

        # Clamp amplitude to a reasonable range (0.0 to 1.0)
        self.current_amplitude = max(0.0, min(1.0, self.current_amplitude))

        # Update rotation angle based on amplitude
        self.rotation_angle += self.current_amplitude * 1.0 # Faster rotation with louder audio
        self.rotation_angle %= 360 # Keep angle within 0-360

        # Update hue offset for color cycling
        self.hue_offset = (self.hue_offset + self.current_amplitude * 1.0) % 360 # Increased speed

        # Add particles based on amplitude
        if self.current_amplitude > 0.3 and random.random() < self.current_amplitude * 0.7: # More frequent particles
            num_new_particles = int(self.current_amplitude * 20) # More particles
            for _ in range(num_new_particles):
                # Create particles near the center, with random velocities
                x = random.uniform(-20, 20) # Generate relative to 0,0
                y = random.uniform(-20, 20) # Generate relative to 0,0
                vx = random.uniform(-4, 4) * self.current_amplitude # Faster particles
                vy = random.uniform(-4, 4) * self.current_amplitude
                
                # Pastel-ish colors for particles (lower saturation)
                particle_color = QColor.fromHsv(int(self.hue_offset), 150, 255).getRgbF() # S=150 for pastel
                
                self.particles.append({
                    'x': x, 'y': y, 'vx': vx, 'vy': vy,
                    'lifetime': 90, 'initial_lifetime': 90, 'color': particle_color, # Longer lifetime, store initial
                    'initial_size': 1.0 # Smaller initial size for particles
                })
        
        # Store current state for ghosting
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
        glEnable(GL_LINE_SMOOTH) # Enable line antialiasing
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST) # Hint for best quality antialiasing

        # Generate VBOs for kaleidoscope lines
        self.vbo_kaleidoscope_vertex = glGenBuffers(1)
        self.vbo_kaleidoscope_color = glGenBuffers(1)


    def resizeGL(self, w: int, h: int):
        """
        Resizes the OpenGL viewport and sets up the perspective projection.
        """
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Set up a perspective projection
        gluPerspective(45.0, w / h, 0.1, 1000.0) # FOV, Aspect, Near, Far
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousePressEvent(self, event):
        """
        Toggles the oscillation mode on mouse click.
        """
        if event.button() == Qt.LeftButton:
            self.oscillation_mode = (self.oscillation_mode + 1) % 2
            self.update()

    def paintGL(self):
        """
        Renders the kaleidoscope pattern and particles.
        """
        # Background should remain black
        glClearColor(0, 0, 0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        w, h = self.width(), self.height()
        
        # Increment frame count for oscillation phase
        self.frame_count += 1

        # Reset current vertex offset for kaleidoscope VBOs
        current_kaleidoscope_vertex_offset = 0
        # List of (start_idx, num_pts, line_width, z_offset, rotation_angle)
        kaleidoscope_draw_commands = [] 

        # Draw kaleidoscope pattern and its ghosts
        history_copy = list(self.history)
        
        # Draw historical ghosts first
        for i, (hist_rot_angle, hist_hue_offset, hist_amplitude) in enumerate(history_copy):
            # Calculate Z-offset for "zoom through" effect - increased multiplier
            z_offset = -50.0 - (len(history_copy) - 1 - i) * 10.0 # Older ghosts are further back

            # Normalized age (0 for oldest, 1 for newest ghost)
            normalized_age = i / max(1, len(history_copy) - 1)
            # Apply a quadratic fade and scale to a desired alpha range (e.g., 0.05 to 0.75)
            alpha_fade = normalized_age ** 2 * 0.7 + 0.05
            alpha_fade = min(1.0, max(0.0, alpha_fade)) # Clamp alpha between 0 and 1

            # Calculate strobe_val for this ghost based on its amplitude
            strobe_val_for_ghost = hist_amplitude * 0.9 + 0.1

            total_vertices_added, sub_commands = kaleidoscope_computations.fill_kaleidoscope_data_cython( # CALLING CYTHON
                self.kaleidoscope_vertices_buffer, self.kaleidoscope_colors_buffer,
                current_kaleidoscope_vertex_offset,
                hist_rot_angle, hist_hue_offset, hist_amplitude,
                False, strobe_val_for_ghost, alpha_fade, # Pass strobe_val for lines
                self.frame_count, self.oscillation_mode # Pass frame_count and oscillation_mode
            )
            # Add each sub-command with its line width, z_offset, and rotation_angle
            for rel_start_idx, num_pts, line_w in sub_commands: # Unpack line_w
                kaleidoscope_draw_commands.append((current_kaleidoscope_vertex_offset + rel_start_idx, num_pts, line_w, z_offset, hist_rot_angle)) # Use line_w
            current_kaleidoscope_vertex_offset += total_vertices_added


        # Draw the current, most prominent kaleidoscope pattern
        strobe_val_for_current = self.current_amplitude * 0.9 + 0.1
        total_vertices_added, sub_commands = kaleidoscope_computations.fill_kaleidoscope_data_cython( # CALLING CYTHON
            self.kaleidoscope_vertices_buffer, self.kaleidoscope_colors_buffer,
            current_kaleidoscope_vertex_offset,
            self.rotation_angle, self.hue_offset, self.current_amplitude,
            True, strobe_val_for_current, 1.0, # Current pattern at Z=0, full opacity
            self.frame_count, self.oscillation_mode
        )
        for rel_start_idx, num_pts, line_w in sub_commands: # Unpack line_w
            kaleidoscope_draw_commands.append((current_kaleidoscope_vertex_offset + rel_start_idx, num_pts, line_w, 0.0, self.rotation_angle)) # Use line_w
        current_kaleidoscope_vertex_offset += total_vertices_added

        # --- Send ALL kaleidoscope data to GPU ---
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_vertex)
        glBufferData(GL_ARRAY_BUFFER, self.kaleidoscope_vertices_buffer.nbytes, self.kaleidoscope_vertices_buffer, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_color)
        glBufferData(GL_ARRAY_BUFFER, self.kaleidoscope_colors_buffer.nbytes, self.kaleidoscope_colors_buffer, GL_DYNAMIC_DRAW)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_vertex)
        glVertexPointer(2, GL_FLOAT, 0, None) 

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_kaleidoscope_color)
        glColorPointer(4, GL_FLOAT, 0, None)

        # Execute ALL kaleidoscope draw calls
        for start_idx, num_pts, line_w, z_offset_for_draw, rotation_angle_for_draw in kaleidoscope_draw_commands:
            glPushMatrix() # Push current modelview matrix
            glTranslatef(0, 0, z_offset_for_draw) # Apply Z-translation
            glRotatef(rotation_angle_for_draw, 0, 0, 1) # Apply rotation
            
            glLineWidth(line_w) # <-- use the passed-in line width
            glDrawArrays(GL_LINE_STRIP, start_idx, num_pts)
            glPopMatrix() # Pop matrix to restore previous state

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


        # Update and draw particles
        new_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                new_particles.append(p)
                # Fade out particles based on initial lifetime, starting more transparent
                # Alpha curve for orb-like fade: starts lower, peaks, then fades out
                normalized_lifetime = p['lifetime'] / p['initial_lifetime']
                alpha = math.sin(normalized_lifetime * math.pi) * 1.0 # Max 100% opacity, starts/ends at 0
                alpha = max(0.0, min(1.0, alpha)) # Clamp alpha

                glPushMatrix() # Save current matrix state for particle
                # Translate particle into view along Z, and to its relative X, Y position
                # Apply current rotation of the kaleidoscope to particles as well
                glTranslatef(p['x'], p['y'], -200.0) # Z-offset to make particles visible in perspective
                glRotatef(self.rotation_angle, 0, 0, 1) # Rotate particles with kaleidoscope

                # Apply strobing to particle color using the same logic as in Cython
                # Re-calculate strobe_pow_particle based on current_amplitude
                strobe_pow_particle = self.current_amplitude ** 6.0 # Use the same power as in Cython
                
                # Desaturation for particles
                desaturation_factor_particle = strobe_pow_particle
                target_saturation_hsv_particle = int((1.0 - desaturation_factor_particle) * 255)
                target_saturation_hsv_particle = max(0, min(255, target_saturation_hsv_particle))

                # Brightness for particles
                # MODIFIED: Increased the base value for particle brightness
                target_value_hsv_particle = int(150 + strobe_pow_particle * 105.0) # Range from 150 to 255
                target_value_hsv_particle = max(150, min(255, target_value_hsv_particle)) # Ensure it's clamped correctly

                # Convert original particle color (r,g,b) to a QColor to use its HSV conversion
                original_particle_qcolor = QColor.fromRgbF(*p['color']) # Unpack original color
                original_h = original_particle_qcolor.hue()
                
                # Apply the same HSV logic as in Cython for particle color
                particle_strobe_qcolor = QColor.fromHsv(original_h, target_saturation_hsv_particle, target_value_hsv_particle)
                strobe_r, strobe_g, strobe_b, _ = particle_strobe_qcolor.getRgbF()

                # Draw aura (larger, more transparent)
                # MODIFIED: Increased aura alpha multiplier
                glColor4f(strobe_r, strobe_g, strobe_b, alpha * 0.7) # More opaque aura
                glBegin(GL_QUADS)
                aura_size = p['initial_size'] * 3.0 # Larger for aura
                glVertex2f(-aura_size, -aura_size)
                glVertex2f(aura_size, -aura_size)
                glVertex2f(aura_size, aura_size)
                glVertex2f(-aura_size, aura_size)
                glEnd()
                
                # Draw core particle (smaller, less transparent)
                glColor4f(strobe_r, strobe_g, strobe_b, alpha) # Full alpha for core
                glBegin(GL_QUADS) # Draw as a small square
                size = p['initial_size'] * (normalized_lifetime * 0.8 + 0.2) # Shrink slightly as they fade
                glVertex2f(-size, -size)
                glVertex2f(size, -size)
                glVertex2f(size, size)
                glVertex2f(-size, size)
                glEnd()
                glPopMatrix() # Restore matrix state
        self.particles = new_particles


class KaleidoscopeVisualizerWindow(QMainWindow):
    """
    A separate QMainWindow that hosts the KaleidoscopeVisualizerGLWidget.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaleidoscope Visualizer")
        self.kv = KaleidoscopeVisualizerGLWidget()
        # Removed: self.audio_queue = self.kv.audio_queue  # Expose the audio queue
        cw = QWidget()
        layout = QVBoxLayout(cw)
        layout.addWidget(self.kv)
        self.setCentralWidget(cw)
        self.resize(600, 600)   # Set a reasonable default size
    
    def showEvent(self, event):
        """
        Overrides showEvent to clear particles when the window is shown.
        """
        super().showEvent(event)
        self.kv.clear_particles() # Clear particles when the window becomes visible
            

# ─── Event Table Model ──────────────────────────────────────────────────

# Color map for different MIDI message types in the event table
COLOR_MAP = {
    'note_on': '#8BE9FD',       # Light blue
    'note_off': '#6272A4',      # Dark blue-gray
    'control_change': '#FFB86C',# Orange
    'program_change': '#50FA7B',# Green
    'pitchwheel': '#FF79C6'     # Pink
}

class EventsModel(QAbstractTableModel):
    """
    A custom table model for displaying parsed MIDI events in a QTableView.
    It formats MIDI event data (time, type, parameters) for presentation.
    """
    HEAD = ['Measure', 'Beat', 'Dur', 'Time(s)', 'Ch', 'Type', 'Param'] # Column headers

    def __init__(self, events: list, ticks_per_beat: int, time_signature_changes: list):
        """
        Initializes the model with a list of parsed MIDI event dictionaries.

        Args:
            events (list): A list of dictionaries, where each dict represents a MIDI event.
            ticks_per_beat (int): The MIDI file's ticks per beat.
            time_signature_changes (list): A sorted list of (tick, numerator, denominator, cumulative_measures) tuples.
        """
        super().__init__()
        self.ev = events # Store the list of event dictionaries
        self.ticks_per_beat = ticks_per_beat
        self.time_signature_changes = time_signature_changes

    def rowCount(self, parent=QModelIndex()):
        """
        Returns the number of rows in the table.
        """
        return len(self.ev)

    def columnCount(self, parent=QModelIndex()):
        """
        Returns the number of columns in the table.
        """
        return len(self.HEAD)

    def data(self, idx, role=Qt.DisplayRole):
        """
        Returns the data for a given index and role.

        Args:
            idx (QModelIndex): The index of the cell.
            role (Qt.ItemDataRole): The role of the data requested.

        Returns:
            QVariant: The data for the specified role, formatted for display.
        """
        if not idx.isValid():
            return QVariant() # Return invalid variant for invalid indices
        
        e = self.ev[idx.row()] # Get the event dictionary for the current row
        c = idx.column()      # Get the column index

        if role == Qt.DisplayRole:
            # Format data based on column index
            if c == 0: # Measure
                measure, _ = calculate_beat_measure(e['abs'], self.ticks_per_beat, self.time_signature_changes)
                return measure
            if c == 1: # Beat
                _, beat = calculate_beat_measure(e['abs'], self.ticks_per_beat, self.time_signature_changes)
                return f"{beat+1:.2f}" # Beat number (1-indexed, 2 decimal places)
            if c == 2: # Duration
                # Duration calculation already done in _load_midi, assuming it's accurate
                return f"{e['duration_beats']:.2f}"
            if c == 3: return f"{e['time_s']:.3f}" # Time in seconds (3 decimal places)
            if c == 4: 
                # Safely get channel, default to N/A if not present (e.g., for meta messages)
                return e['channel']+1 if e['channel'] is not None else "N/A" 
            if c == 5: return e['type'] # MIDI message type (e.g., 'note_on')
            if c == 6: # Parameters column
                parts = [] # List to build the parameter string
                # Add note name and number if available
                if e['note'] is not None:
                    parts.append(f"{midi_note_to_name(e['note'])}({e['note']})")
                # Add velocity if available
                if e['velocity'] is not None:
                    parts.append(f"vel={e['velocity']}")
                # Add control change name and value if available
                if e['control'] is not None:
                    cc_name = CONTROL_CHANGE_NAMES.get(e['control'], f"CC{e['control']}")
                    parts.append(f"{cc_name}={e['value']}")
                # Add pitch wheel value if available
                if e['pitch'] is not None:
                    parts.append(f"pitch={e['pitch']}")
                # Add program change value if available
                if e['program'] is not None:
                    parts.append(f"prg={e['program']}")
                return ', '.join(parts) # Join all parts with a comma

        elif role == Qt.ForegroundRole and c == 5:
            # Apply color based on message type for the 'Type' column
            return QColor(COLOR_MAP.get(e['type'], '#F8F8F2')) # Default to light gray if type not in map
        
        return QVariant() # Return invalid variant for unsupported roles

    def headerData(self, s, o, r):
        """
        Returns the header data for rows or columns.

        Args:
            s (int): Section index.
            o (Qt.Orientation): Orientation (Horizontal or Vertical).
            r (Qt.ItemDataRole): Role of the data.

        Returns:
            QVariant: The header text.
        """
        if o == Qt.Horizontal and r == Qt.DisplayRole:
            return self.HEAD[s] # Return column names for horizontal headers
        return QVariant()


# ─── Main Window ──────────────────────────────────────────────────────────

class ChronoMIDI(QMainWindow):
    """
    The main application window for ChronoMIDI.
    Manages the GUI layout, MIDI file loading, SoundFont management,
    audio playback, and integration with visualizers.
    """
    # Signal emitted when a MIDI event is processed during playback,
    # used to highlight the corresponding row in the event table.
    event_signal = pyqtSignal(int)
    # Signal emitted when tempo or time signature changes, to update GUI labels
    midi_meta_update_signal = pyqtSignal(float, tuple) # (tempo_bpm, (ts_num, ts_den))

    def __init__(self):
        """
        Initializes the ChronoMIDI main window and its components.
        Sets up the UI, audio stream, and initial state variables.
        """
        super().__init__(); self.setWindowTitle("ChronoMIDI"); self.resize(1000,800)

        # --- Application State Variables ---
        self.midi_path = None               # Path to the currently loaded MIDI file
        self.midi_file = None               # mido.MidiFile instance
        self.sf2_path = None                # Path to the currently loaded SoundFont file
        self.sr = 44100                     # Sample rate for audio playback (44.1 kHz)
        self.eq_queue = deque()             # Queue for audio data to be processed by the equalizer
        self.events = []                    # Parsed MIDI events (list of dictionaries)
        self.channels = []                  # List of active MIDI channels found in the file
        self.sample_events = []             # MIDI events sorted by sample time for real-time dispatch
        self.cur_sample = 0                 # Current audio sample position in playback
        self.is_playing = False             # Playback state flag
        self.synth = None                   # FluidSynth synthesizer instance
        self.vis_win = None                 # VisualizerWindow instance (for oscilloscope), created on demand
        self.kaleidoscope_vis_win = None    # Kaleidoscope Visualizer Window

        # For dynamic tempo and time signature tracking
        self.tempo_changes = []                     # Stores (absolute_tick, tempo_in_bpm) tuples for beat/measure calculation
        self.time_signature_changes = []            # Stores (absolute_tick, numerator, denominator, cumulative_measures_at_this_tick) tuples for beat/measure calculation
        self.tempo_changes_by_time_s = []           # Stores (time_s, tempo_in_bpm) for GUI updates
        self.time_signature_changes_by_time_s = []  # Stores (time_s, numerator, denominator) for GUI updates

        self.current_tempo_bpm = 120.0          # Current active tempo
        self.current_time_signature = (4, 4)    # Current active time signature
        self.ticks_per_beat = 480               # Default mido ticks_per_beat, updated on MIDI load


        # --- Audio Stream Setup ---
        # Uses sounddevice to open an audio output stream
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2, # Stereo output
            dtype='int16', # 16-bit integer PCM
            callback=self._audio_cb, # Callback function for audio buffer requests
            blocksize=1024 # Size of audio blocks requested by the callback
        )

        # --- GUI Layout Setup ---
        central = QWidget() # Create a central widget for the QMainWindow
        self.setCentralWidget(central)  # Set it as the main window's central widget
        v = QVBoxLayout(central)    # Use a vertical layout for the central widget

        # File Label (displays current MIDI file name)
        self.lbl_file=QLabel("No file loaded")
        self.lbl_file.setStyleSheet("color:white;")
        v.addWidget(self.lbl_file)

        # Metadata Group Box
        meta = QGroupBox("File Metadata")
        # Labels for MIDI metadata
        self.lbl_tempo = QLabel(f"Tempo: {self.current_tempo_bpm} BPM") # Tempo
        self.lbl_ts = QLabel(f"Time Sig: {self.current_time_signature[0]}/{self.current_time_signature[1]}")    # Time Signature
        self.lbl_key = QLabel()     # Key Signature
        self.lbl_meta = QLabel()    # Other metadata (currently commented out in form)
        # Apply white color style to metadata labels
        for l in (self.lbl_tempo,self.lbl_ts,self.lbl_key,self.lbl_meta): 
            l.setStyleSheet("color:white;")
        # Form layout for metadata labels
        f = QFormLayout(meta)
        f.addRow("Tempo:",self.lbl_tempo)
        f.addRow("Time Sig:",self.lbl_ts)
        f.addRow("Key Sig:",self.lbl_key);
        # f.addRow("Other:",self.lbl_meta) # Example of other metadata, currently unused in UI
        v.addWidget(meta)

        # Tab Widget for MIDI Event Tables
        self.tabs=QTabWidget(); self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;} "   # No border around the tab pane
            "QTabBar::tab{background:#222;color:white;padding:5px;} "   # Styling for unselected tabs
            "QTabBar::tab:selected{background:#555;}")  # Styling for selected tab
        v.addWidget(self.tabs)

        # Equalizer Widget (OpenGL)
        self.eq = EqualizerGLWidget(sr = self.sr, bands = 256)
        self.eq.setFixedHeight(200) # Height
        v.addWidget(self.eq)
        # Timer to regularly drain the equalizer queue and update the display
        QTimer(self, timeout = self._drain_eq, interval = 1000 // 60).start()

        # Playback Control Buttons
        h = QHBoxLayout()   # Horizontal layout for buttons
        def btn(t, cb):
            """
            Helper function to create a styled QPushButton.
            """
            b = QPushButton(t, clicked=cb)
            b.setStyleSheet("background:#333;color:white;padding:6px;")
            h.addWidget(b)
        
        # Add buttons with their respective callback functions
        btn("Open MIDI…", self.open_midi)
        btn("Load SF2…", self.open_sf2)
        btn("Play", self.play)
        btn("Pause", self.pause)
        btn("Stop", self.stop)
        btn("Visualizer…", self.show_vis)
        btn("Kaleidoscope…", self.show_kaleidoscope_vis)
        btn("Export MP3…", self.export_mp3)
        
        h.addStretch()  # Adds a stretchable space to push buttons to the left
        v.addLayout(h)  # Add the button layout to the main vertical layout

        # Connect custom signals to their slots
        self.event_signal.connect(self._hilite)
        self.midi_meta_update_signal.connect(self._update_meta_display) # NEW: Connect for meta updates

    def _update_meta_display(self, tempo_bpm: float, time_signature: tuple):
        """
        Slot to update the tempo and time signature labels in the GUI.
        Connected to midi_meta_update_signal.

        Args:
            tempo_bpm (float): The current tempo in BPM.
            time_signature (tuple): The current time signature (numerator, denominator).
        """
        self.lbl_tempo.setText(f"Tempo: {tempo_bpm:.2f} BPM")
        self.lbl_ts.setText(f"Time: {time_signature[0]}/{time_signature[1]}")

    def _drain_eq(self):
        """
        Drains the audio queue for the equalizer, pushing PCM data to it for visualization.
        Called by a QTimer to update the equalizer at a consistent rate.
        """
        while self.eq_queue:
            self.eq.push_audio(self.eq_queue.popleft()) # Pop audio block and push to equalizer

    def _hilite(self, idx: int):
        """
        Highlights a specific row in the currently active MIDI event table and
        scrolls to ensure it's visible.

        Args:
            idx (int): The index of the event (row) to highlight. This is the index
                       in the `self.events` list, not necessarily in the current tab's model.
        """
        # Get the current tab index and the corresponding QTableView
        tab = self.tabs.currentIndex()
        tbl = self.tables[tab]
        
        # Select the row in the table view
        tbl.selectRow(idx)
        # Scroll the table view to center the selected row
        tbl.scrollTo(tbl.model().index(idx, 0), QAbstractItemView.PositionAtCenter)

    def open_midi(self):
        """
        Opens a file dialog to select a MIDI file, loads it, and updates the UI.
        Stops any ongoing playback before loading.
        """
        # Open file dialog for MIDI files (.mid, .midi)
        p, _ = QFileDialog.getOpenFileName(self, "Open MIDI", "", "MIDI Files (*.mid *.midi)")
        if not p:
            return # User cancelled dialog
        
        self.stop() # Stop any current playback
        self.midi_path = p # Store the path to the loaded MIDI file
        self.lbl_file.setText(os.path.basename(p)) # Display file name in UI
        self._load_midi(p) # Parse and load the MIDI file data

    def open_sf2(self):
        """
        Opens a file dialog to select a SoundFont (.sf2) file and loads it into FluidSynth.
        Initializes FluidSynth if not already done.
        """
        # Open file dialog for SoundFont files (.sf2)
        p, _ = QFileDialog.getOpenFileName(self, "Load SoundFont", "", "SF2 Files (*.sf2)")
        if not p:
            return # User cancelled dialog
        
        self.sf2_path = p # Store the path to the loaded SoundFont
        
        # Initialize FluidSynth if it's not already running
        if not self.synth:
            self.synth = Synth()
        
        # Load the SoundFont into FluidSynth and select the default program (preset 0, bank 0)
        sfid = self.synth.sfload(p) # Load SoundFont and get its ID
        self.synth.program_select(0, sfid, 0, 0) # Select program 0 on channel 0

    def _load_midi(self, path: str):
        """
        Parses the specified MIDI file, extracts metadata, and processes MIDI events.
        Populates metadata labels and prepares event data for display and playback.
        This now includes pre-processing for all tempo and time signature changes.

        Args:
            path (str): The path to the MIDI file to load.
        """
        mid = mido.MidiFile(path) # Load the MIDI file using mido
        self.midi_file = mid # Store the mido.MidiFile instance
        self.ticks_per_beat = mid.ticks_per_beat # Store ticks per beat

        # Initialize metadata variables with default values for the file's start
        tempo_us_per_beat = mido.bpm2tempo(120) # Default MIDI tempo (500,000 microseconds per beat = 120 BPM)
        time_sig_num = 4    # Default Time Signature (4/4) numerator
        time_sig_den = 4    # Default Time Signature (4/4) denominator
        key_signature = None     # Key Signature
        other_meta_types = []     # List for other meta message types

        # NEW: Clear and populate tempo and time signature change lists
        self.tempo_changes = []         # Stores (absolute_tick, tempo_in_bpm) tuples for beat/measure calculation
        # time_signature_changes will now store (absolute_tick, numerator, denominator, cumulative_measures)
        self.time_signature_changes = []
        self.tempo_changes_by_time_s = [] # Stores (time_s, tempo_in_bpm) for GUI updates
        self.time_signature_changes_by_time_s = [] # Stores (time_s, numerator, denominator) for GUI updates

        # Add initial values at tick 0.
        self.tempo_changes.append((0, mido.bpm2tempo(120))) # Store tempo in microseconds per beat
        # At tick 0, 4/4, 0 cumulative measures
        self.time_signature_changes.append((0, 4, 4, 0)) # Cumulative measures start at 0 (integer)
        self.tempo_changes_by_time_s.append((0.0, 120.0)) # Store BPM for display
        self.time_signature_changes_by_time_s.append((0.0, 4, 4))

        # --- Collect all messages with their absolute times for chronological processing ---
        all_messages_with_abs_time = []
        for track in self.midi_file.tracks:
            absolute_tick = 0
            for msg in track:
                absolute_tick += msg.time # msg.time is delta time
                all_messages_with_abs_time.append((absolute_tick, msg))

        # Sort all messages by absolute tick to process them chronologically
        all_messages_with_abs_time.sort(key=lambda x: x[0])

        current_tempo_for_time_calc = mido.bpm2tempo(120) # Initial tempo for time_s calculation
        current_time_s = 0.0
        last_event_tick = 0

        # Variables for cumulative measure calculation
        cumulative_measures_total = 0 # This will be an integer count of full measures completed
        last_ts_change_abs_tick = 0
        current_ts_numerator_calc = 4
        current_ts_denominator_calc = 4

        ev = [] # Initialize the list to store processed event dictionaries

        # Process meta messages to build tempo_changes and time_signature_changes lists
        # Also extract initial metadata for display
        for absolute_tick, msg in all_messages_with_abs_time:
            # Calculate time_s up to this message using the tempo active *before* this message
            # This ensures time_s values are accurate even with tempo changes
            current_time_s += mido.tick2second(absolute_tick - last_event_tick, self.ticks_per_beat, current_tempo_for_time_calc)
            last_event_tick = absolute_tick # Update last_event_tick for the next segment

            if msg.is_meta:
                if msg.type == 'set_tempo':
                    new_bpm = mido.tempo2bpm(msg.tempo)
                    # Only add if it's a new tempo or the very first one at this tick
                    if not self.tempo_changes or self.tempo_changes[-1][0] != absolute_tick or self.tempo_changes[-1][1] != msg.tempo:
                        self.tempo_changes.append((absolute_tick, msg.tempo)) # Store tempo in microseconds per beat
                        self.tempo_changes_by_time_s.append((current_time_s, new_bpm)) # Store BPM for display
                    current_tempo_for_time_calc = msg.tempo # Update tempo for subsequent time_s calculations
                elif msg.type == 'time_signature':
                    new_num, new_den = msg.numerator, msg.denominator
                    
                    # Calculate measures passed in the segment *before* this time signature change
                    ticks_in_segment = absolute_tick - last_ts_change_abs_tick
                    
                    if ticks_in_segment > 0:
                        quarter_notes_in_segment = ticks_in_segment / self.ticks_per_beat
                        # Conversion factor from quarter notes to the beat unit of the *previous* time signature
                        conversion_factor = current_ts_denominator_calc / 4.0
                        beats_in_segment_in_target_unit = quarter_notes_in_segment * conversion_factor
                        
                        beats_per_measure_for_segment = current_ts_numerator_calc
                        
                        if beats_per_measure_for_segment > 0:
                            # Add the full measures from this segment to the cumulative total
                            cumulative_measures_total += int(beats_in_segment_in_target_unit / beats_per_measure_for_segment)
                    
                    # Only add if it's a new time signature or the very first one at this tick
                    # The cumulative_measures_total at this point represents measures *completed before* this new TS block starts.
                    if not self.time_signature_changes or \
                        self.time_signature_changes[-1][0] != absolute_tick or \
                        self.time_signature_changes[-1][1] != new_num or \
                        self.time_signature_changes[-1][2] != new_den:
                            self.time_signature_changes.append((absolute_tick, new_num, new_den, cumulative_measures_total))
                            self.time_signature_changes_by_time_s.append((current_time_s, new_num, new_den))
                    
                    # Update for the new time signature for subsequent calculations
                    current_ts_numerator_calc = new_num
                    current_ts_denominator_calc = new_den
                    last_ts_change_abs_tick = absolute_tick
                elif msg.type == 'key_signature':
                    if key_signature is None: # Only take the first key signature found
                        key_signature = msg.key
                else:
                    other_meta_types.append(msg.type)
                continue # Skip non-meta processing for meta messages
            
            # For non-meta messages, add to 'ev' list
            # (This implicitly handles the 'AttributeError: 'Message' object has no attribute 'channel')
            # Measure and beat will be calculated in the EventsModel's data method using the pre-calculated cumulative measures.
            event_dict = dict(
                time_s=current_time_s,     # Time in seconds
                abs=absolute_tick,         # Absolute ticks from start
                channel=getattr(msg, 'channel', None), # Safely get channel, default to None
                type=msg.type,             # MIDI message type (e.g., 'note_on')
                note=getattr(msg, 'note', None),           # Note number (for note_on/off)
                velocity=getattr(msg, 'velocity', None),   # Velocity (for note_on/off)
                control=getattr(msg, 'control', None),     # Control number (for control_change)
                value=getattr(msg, 'value', None),         # Value (for control_change/pitchwheel)
                pitch=getattr(msg, 'pitch', None),         # Pitch bend value (for pitchwheel)
                program=getattr(msg, 'program', None)      # Program number (for program_change)
            )
            ev.append(event_dict)

        # Ensure lists are sorted (they should be from the initial sort, but good practice)
        self.tempo_changes.sort(key=lambda x: x[0])
        self.time_signature_changes.sort(key=lambda x: x[0])
        self.tempo_changes_by_time_s.sort(key=lambda x: x[0])
        self.time_signature_changes_by_time_s.sort(key=lambda x: x[0])

        # After processing all messages, calculate measures for the final segment
        # up to the end of the MIDI file's perceived length or the last event's tick.
        final_tick = self.midi_file.length * self.ticks_per_beat # This length is in seconds, need to convert
        
        # Calculate the end tick based on the total time length of the MIDI file
        # This is more robust than just using the last event's tick, as some files might have trailing silence.
        # Use the tempo active at the very end of the file for this conversion
        last_tempo_us_per_beat_for_final_calc = self.tempo_changes[-1][1] if self.tempo_changes else mido.bpm2tempo(120)
        midi_file_end_tick = mido.second2tick(self.midi_file.length, self.ticks_per_beat, last_tempo_us_per_beat_for_final_calc)

        # Ensure final_tick is at least as large as the last event's absolute tick
        if ev:
            midi_file_end_tick = max(midi_file_end_tick, ev[-1]['abs'])
        
        # Calculate measures for the very last segment of the MIDI file
        ticks_in_final_segment = midi_file_end_tick - last_ts_change_abs_tick
        if ticks_in_final_segment > 0:
            quarter_notes_in_segment = ticks_in_final_segment / self.ticks_per_beat
            conversion_factor = current_ts_denominator_calc / 4.0
            beats_in_segment_in_target_unit = quarter_notes_in_segment * conversion_factor
            
            beats_per_measure_for_segment = current_ts_numerator_calc
            if beats_per_measure_for_segment > 0:
                cumulative_measures_total += int(beats_in_segment_in_target_unit / beats_per_measure_for_segment)


        # Update initial display based on the first entries in the sorted lists
        # or the defaults if no meta messages were found
        self.current_tempo_bpm = self.tempo_changes_by_time_s[0][1] if self.tempo_changes_by_time_s else 120.0
        self.current_time_signature = (self.time_signature_changes[0][1], self.time_signature_changes[0][2]) if self.time_signature_changes else (4, 4)

        self.lbl_tempo.setText(f"Tempo: {self.current_tempo_bpm:.2f} BPM")
        self.lbl_ts.setText(f"Time: {self.current_time_signature[0]}/{self.current_time_signature[1]}")
        self.lbl_key.setText(key_signature or "N/A")
        self.lbl_meta.setText(', '.join(other_meta_types) or "N/A")

        # --- Calculate Note Durations ---
        active = {} # Dictionary to track active notes: {(channel, note): index_of_note_on_event}
        for i, e in enumerate(ev):
            # Check for note_on with velocity > 0
            if e['type'] == 'note_on' and e['velocity'] > 0:
                # If there's an active note with the same channel and note, it means the previous one
                # was not explicitly turned off. We'll "turn it off" at the current event's time.
                if (e['channel'], e['note']) in active:
                    prev_note_on_idx = active.pop((e['channel'], e['note']))
                    # Calculate duration for the previous note
                    if ev[prev_note_on_idx]['abs'] < e['abs']: # Ensure start tick is before end tick
                        ev[prev_note_on_idx]['duration_beats'] = (e['abs'] - ev[prev_note_on_idx]['abs']) / self.ticks_per_beat
                
                # Store the current note_on event
                active[(e['channel'], e['note'])] = i
                e['duration_beats'] = 0.0 # Initialize duration to 0, will be updated by note_off or end of file
            
            # Check for note_off or note_on with velocity == 0
            elif (e['type'] == 'note_off' or (e['type'] == 'note_on' and e['velocity'] == 0)) \
                 and (e['channel'], e['note']) in active:
                
                # Get the index of the corresponding note_on event
                note_on_idx = active.pop((e['channel'], e['note']))
                
                # Calculate duration in beats and store it in the original note_on event
                if ev[note_on_idx]['abs'] < e['abs']: # Ensure start tick is before end tick
                    # Corrected line: Use ev[note_on_idx]['abs'] instead of e[note_on_idx]['abs']
                    ev[note_on_idx]['duration_beats'] = (e['abs'] - ev[note_on_idx]['abs']) / self.ticks_per_beat
        
        # After iterating through all events, handle any notes that are still "active"
        # (i.e., they had a note_on but no corresponding note_off/velocity 0 note_on)
        # Assign them a duration until the end of the MIDI file.
        
        # Determine the effective end tick of the MIDI file for duration calculation.
        # This uses the already calculated midi_file_end_tick from the meta-message processing.
        
        for (channel, note), note_on_idx in active.items():
            # If the note_on event's absolute tick is before the end of the file
            if ev[note_on_idx]['abs'] < midi_file_end_tick:
                ev[note_on_idx]['duration_beats'] = (midi_file_end_tick - ev[note_on_idx]['abs']) / self.ticks_per_beat
            else:
                # If the note_on is at or after the end of the file, assign a small default duration
                ev[note_on_idx]['duration_beats'] = 0.01 # A very short duration if it's at the very end

        # Ensure all events have a 'duration_beats' key (default to 0.0 if not set for other event types)
        for e in ev:
            e.setdefault('duration_beats', 0.0)


        # Update application state with processed data
        self.events = ev # Store the full list of parsed events
        # Get unique channels and sort them for tab creation
        # Filter out None channels in case of meta messages getting through (though they shouldn't now)
        self.channels = sorted(list({e['channel'] for e in ev if 'channel' in e and e['channel'] is not None}))
        # Prepare events sorted by sample time for efficient playback processing
        self.sample_events = sorted((int(e['time_s'] * self.sr), i) for i, e in enumerate(self.events))
        
        self.cur_sample = 0 # Reset current sample position to start
        self.eq.clear() # Clear the equalizer visualization
        self._build_tables() # Rebuild event tables based on new data

    def _build_tables(self):
        """
        Builds and populates the QTabWidget with QTableView instances for
        displaying MIDI events, including an "All Events" tab and
        separate tabs for each MIDI channel.
        """
        self.tabs.clear() # Clear any existing tabs
        self.tables = [] # Reset the list of table views
        
        # Add a tab for all MIDI events
        self._add_table("All", self.events)
        
        # Add a separate tab for events on each distinct MIDI channel
        for ch in self.channels:
            # Filter events for the current channel
            channel_events = [e for e in self.events if e['channel'] == ch]
            self._add_table(f"Ch{ch+1}", channel_events) # Channel numbers are 1-indexed for display

    def _add_table(self, title: str, evts: list):
        """
        Helper function to create a new QTableView tab with the given events.

        Args:
            title (str): The title for the new tab.
            evts (list): A list of MIDI event dictionaries to display in this table.
        """
        # Pass ticks_per_beat and time_signature_changes to the EventsModel
        model = EventsModel(evts, self.ticks_per_beat, self.time_signature_changes)
        view = QTableView() # Create a new table view
        view.setModel(model) # Set the model for the table view
        view.setItemDelegate(EditDelegate(view)) # Apply custom delegate for styling editors
        
        # Apply dark theme styling to the table view
        view.setStyleSheet(
            "QTableView{background:black;color:white;gridline-color:gray; font-size: 8pt;}" # Added font-size here
            "QHeaderView::section{background:#444;color:white;}"
            "QTableView::item:selected{background:#444;color:white;}"
            "QTableView::item{padding: 0px;}" # Added to reduce padding
        )
        
        view.verticalHeader().setDefaultSectionSize(14) # Reduced from 16 to 14 for less padding
        
        # Resize columns to fit content initially
        for c in range(model.columnCount()):
            view.resizeColumnToContents(c)
        
        self.tabs.addTab(view, title) # Add the new table view as a tab
        self.tables.append(view) # Store the table view instance

    def play(self):
        """
        Starts MIDI playback. Resets the playback position, clears visualizers,
        and starts the audio stream.
        """
        # Do not start if no MIDI events or no SoundFont loaded
        if not (self.events and self.synth):
            return
        
        # Re-sort sample_events to ensure correct playback from current position
        # (Important if _load_midi wasn't called or if events were modified)
        self.sample_events = sorted((int(e['time_s'] * self.sr), i) for i, e in enumerate(self.events))
        
        self.cur_sample = 0 # Reset playback to the beginning
        self.eq.clear()     # Clear equalizer state
        
        # Start the audio output stream, which will trigger _audio_cb
        if not self.stream.active:
            self.stream.start()
        
        self.is_playing = True # Set playback state to true

    def pause(self):
        """
        Pauses MIDI playback by stopping the audio stream.
        """
        if self.stream.active: # Check if stream is active before trying to stop
            self.stream.stop()
        self.is_playing = False # Set playback state to false

    def stop(self):
        """
        Stops MIDI playback, resets playback position, clears visualizers,
        and sends 'All Notes Off' messages to FluidSynth.
        """
        if self.stream.active: # Check if stream is active before trying to stop
            self.stream.stop()
        
        self.is_playing = False # Set playback state to false
        self.eq.clear() # Clear equalizer state
        
        # If FluidSynth instance exists, reset it or send 'All Notes Off'
        if self.synth:
            try:
                # Attempt a full system reset
                self.synth.system_reset()
            except Exception:
                # Fallback: send All Notes Off (CC 123) to all 16 MIDI channels
                # This ensures any hanging notes are turned off.
                [self.synth.cc(ch, 123, 0) for ch in range(16)]

    def _audio_cb(self, out: np.ndarray, frames: int, time, status):
        """
        Audio callback function for sounddevice. This function is called
          periodically by the audio hardware to fill a buffer with audio samples.
        It dispatches MIDI events, renders audio via FluidSynth, and updates
        the GUI with current tempo and time signature.

        Args:
            out (np.ndarray): The output buffer to fill with audio samples.
            frames (int): The number of frames (samples per channel) requested.
            time (object): A Cffi_audiodevice._time_info struct.
            status (sd.CallbackFlags): Status flags (e.g., xrun, input_overflow).
        """
        # Calculate the start and end sample for the current audio block
        s0, s1 = self.cur_sample, self.cur_sample + frames

        # Process MIDI events that fall within the current audio block's time range
        # self.sample_events is sorted by sample time
        while self.sample_events and self.sample_events[0][0] < s1:
            _, idx = self.sample_events.pop(0) # Get the index of the event
            e = self.events[idx] # Retrieve the actual event dictionary
            ch = e['channel'] # Get the MIDI channel

            # Dispatch MIDI message to FluidSynth based on event type
            if e['type'] == 'note_on':
                self.synth.noteon(ch, e['note'], e['velocity'])
            elif e['type'] == 'note_off':
                self.synth.noteoff(ch, e['note'])
            elif e['type'] == 'control_change':
                self.synth.cc(ch, e['control'], e['value'])
            elif e['type'] == 'program_change':
                self.synth.program_change(ch, e['program'])
            elif e['type'] == 'pitchwheel':
                self.synth.pitch_bend(ch, e['pitch'])
            # Note: FluidSynth handles 'set_tempo' messages internally if they are part of the MIDI stream
            # that it processes, but we are managing tempo changes for display separately.

            # Emit signal to highlight the event in the GUI table
            self.event_signal.emit(idx)

        # Update current sample position
        self.cur_sample = s1
        
        # NEW: Update current tempo and time signature for GUI display
        # Use the current playback time in seconds for accurate lookup
        current_time_s_playback = self.cur_sample / self.sr
        
        new_tempo_bpm = self.current_tempo_bpm # Start with current display value
        # Find the latest tempo change by time_s that has occurred
        for time_s_change, bpm in self.tempo_changes_by_time_s:
            if current_time_s_playback >= time_s_change:
                new_tempo_bpm = bpm
            else:
                break # List is sorted by time_s, so no more changes apply yet
        
        new_time_sig = self.current_time_signature
        # Find the latest time signature change by time_s that has occurred
        for time_s_change, num, den in self.time_signature_changes_by_time_s:
            if current_time_s_playback >= time_s_change:
                new_time_sig = (num, den)
            else:
                break # List is sorted by time_s, so no more changes apply yet

        # Emit signal ONLY if the tempo or time signature has actually changed
        if new_tempo_bpm != self.current_tempo_bpm or new_time_sig != self.current_time_signature:
            self.current_tempo_bpm = new_tempo_bpm
            self.current_time_signature = new_time_sig
            self.midi_meta_update_signal.emit(self.current_tempo_bpm, self.current_time_signature)


        # Get audio samples from FluidSynth for the requested number of frames
        # Reshape to (frames, 2) for stereo output
        pcm = np.frombuffer(self.synth.get_samples(frames), dtype=np.int16).reshape(-1, 2)
        
        # Copy the generated PCM data to the output buffer for sounddevice
        out[:] = pcm
        
        # If playback is active, push audio data to visualizer queues
        if self.is_playing:
            self.eq_queue.append(pcm.copy()) # For the equalizer
            if self.vis_win: # Only push to oscilloscope if its window is open
                self.vis_win.audio_queue.append(pcm.copy()) # For the oscilloscope visualizer
            if self.kaleidoscope_vis_win and self.kaleidoscope_vis_win.isVisible(): # NEW: Only push to kaleidoscope visualizer if its window is open AND visible
                self.kaleidoscope_vis_win.kv.push_audio(pcm.copy()) # Direct call to push_audio


    def show_vis(self):
        """
        Displays the oscilloscope visualizer window.
        Creates the VisualizerWindow instance if it doesn't already exist.
        """
        if self.vis_win is None:
            self.vis_win = VisualizerWindow(sr=self.sr) # Create window if not yet created
        self.vis_win.show() # Show the visualizer window

    def show_kaleidoscope_vis(self):
        """
        Displays the kaleidoscope visualizer window.
        Creates the KaleidoscopeVisualizerWindow instance if it doesn't already exist.
        """
        if self.kaleidoscope_vis_win is None:
            self.kaleidoscope_vis_win = KaleidoscopeVisualizerWindow()
        self.kaleidoscope_vis_win.show()

    def export_mp3(self):
        """
        Exports the loaded MIDI file as an MP3 audio file.
        Requires both a MIDI file and a SoundFont to be loaded.
        Uses FluidSynth (command-line) and FFmpeg (command-line) for the conversion.
        """
        # Check if both MIDI and SoundFont are loaded
        if not (self.midi_path and self.sf2_path):
            QMessageBox.warning(self, "Export", "Please load a MIDI file and a SoundFont first.")
            return

        # Open file dialog to choose output MP3 file name
        fn, _ = QFileDialog.getSaveFileName(self, "Save MP3", "", "MP3 Files (*.mp3)")
        if not fn:
            return # User cancelled dialog
        
        # Ensure the filename has a .mp3 extension
        if not fn.lower().endswith(".mp3"):
            fn += ".mp3"
        
        # Define a temporary WAV file path for intermediate audio
        wav = fn[:-4] + "_tmp.wav"

        try:
            # Step 1: Render MIDI to WAV using FluidSynth command-line tool
            subprocess.run([
                "fluidsynth",
                "-q", # Quiet mode
                "-i", # Interactive shell (useful for scripting, though not strictly needed here)
                "-F", wav, # Output to WAV file
                "-r", str(self.sr), # Sample rate
                self.sf2_path, # SoundFont file
                self.midi_path # MIDI file
            ], check=True, # Raise an exception if the command returns a non-zero exit code
            capture_output=True, text=True) # Capture output for debugging (optional)

            # Step 2: Convert WAV to MP3 using FFmpeg command-line tool
            subprocess.run([
                "ffmpeg",
                "-y", # Overwrite output file if it exists
                "-i", wav, # Input WAV file
                "-codec:a", "libmp3lame", # Use LAME MP3 encoder
                "-qscale:a", "2", # Quality scale (2 is usually very good quality)
                fn # Output MP3 file
            ], check=True,
            capture_output=True, text=True) # Capture output for debugging (optional)

        except FileNotFoundError as e:
            # Handle case where fluidsynth or ffmpeg executables are not found
            QMessageBox.critical(self, "Export Error",
                                 f"External tool not found. Please ensure FluidSynth and FFmpeg are installed and in your system's PATH.\nError: {e}")
            return
        except subprocess.CalledProcessError as e:
            # Handle errors during the subprocess calls (e.g., invalid MIDI/SF2)
            error_message = f"Export failed. Please check your MIDI and SoundFont files.\nError: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            QMessageBox.critical(self, "Export Error", error_message)
            return
        except Exception as e:
            # Catch any other unexpected errors
            QMessageBox.critical(self, "Export Error", f"An unexpected error occurred during export:\n{e}")
            return
        finally:
            # Clean up: remove the temporary WAV file if it exists
            if os.path.exists(wav):
                os.remove(wav)
        
        QMessageBox.information(self, "Export Complete", f"Successfully saved to:\n{fn}")


# ─── Application bootstrap ───────────────────────────────────────────────

if __name__ == "__main__":
    # Create the QApplication instance. This is essential for any PyQt GUI application.
    app = QApplication(sys.argv)

    # --- Set the application icon ---
    # Construct the path to the custom font file.
    # This ensures the icon is found regardless of the current working directory.
    # Replace 'chronomidi_icon.png' with 'chronomidi_icon.ico' if you're using an .ico file.
    icon_path = os.path.join(os.path.dirname(__file__), "chronomidi.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path)) # Set the application-wide icon
    else:
        print(f"Warning: Application icon not found at {icon_path}")


    # --- Load Custom Font ---
    # Construct the path to the custom font file.
    font_fp = os.path.join(os.path.dirname(__file__), "fonts", "PixelCode.ttf")
    if os.path.exists(font_fp):
        # Add the font to the application's font database
        fid = QFontDatabase.addApplicationFont(font_fp)
        # Get the font family names from the loaded font
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam:
            # Set the application's default font to the loaded custom font
            app.setFont(QFont(fam[0], 9))
        else:
            # Fallback if font families couldn't be retrieved
            print(f"Warning: Could not retrieve font family from {font_fp}. Falling back to system font.")
            app.setFont(QFont("Courier New", 9))
    else:
        print(f"Warning: Custom font not found at {font_fp}. Falling back to system font.")
        app.setFont(QFont("Courier New", 9)) # Default to Courier New if custom font not found


    # --- Set Application Palette (Dark Theme) ---
    pal = QPalette() # Create a new color palette
    # Set various color roles for a dark theme
    pal.setColor(QPalette.Window, QColor('black')) # Background color of windows
    pal.setColor(QPalette.Base, QColor('black'))   # Background color for widgets (e.g., QLineEdit, QTableView)
    pal.setColor(QPalette.WindowText, QColor('white')) # Default text color for windows
    pal.setColor(QPalette.Text, QColor('white'))       # Default text color for editable text
    pal.setColor(QPalette.Button, QColor('#333'))      # Button background color
    pal.setColor(QPalette.ButtonText, QColor('white')) # Button text color
    pal.setColor(QPalette.Highlight, QColor('#444'))   # Selection highlight background color
    pal.setColor(QPalette.HighlightedText, QColor('white')) # Selection highlight text color
    app.setPalette(pal) # Apply the custom palette to the application

    # Create and show the main ChronoMIDI window
    main_window = ChronoMIDI()
    main_window.show()

    # Start the PyQt event loop. This line transfers control to Qt,
    # and the application will wait for user interactions.
    # sys.exit() ensures a clean exit when the application closes.
    sys.exit(app.exec_())
