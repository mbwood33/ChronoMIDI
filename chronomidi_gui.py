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

# Add cython_modules directory to sys.path to find compiled Cython extensions
current_dir = os.path.dirname(os.path.abspath(__file__))
cython_modules_path = os.path.join(current_dir, "cython_modules")
if cython_modules_path not in sys.path:
    sys.path.insert(0, cython_modules_path)

# Custom application modules
from utils import NOTE_NAMES, CONTROL_CHANGE_NAMES, midi_note_to_name, get_time_signature_at_tick, calculate_beat_measure
from models import EventsModel, EditDelegate # COLOR_MAP is used by EventsModel, PandasModel, ReadOnlyDelegate not used directly by ChronoMIDI
from equalizer_widget import EqualizerGLWidget
# Oscilloscope and KaleidoscopeVisualizerGLWidget are imported by windows.py
# from oscilloscope_widget import Oscilloscope
# from kaleidoscope_widget import KaleidoscopeVisualizerGLWidget
from windows import VisualizerWindow, KaleidoscopeVisualizerWindow


# PyQt5 Core Modules
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QVariant # QAbstractTableModel, QModelIndex, QPointF (no longer directly used here)
)
# PyQt5 GUI Modules
from PyQt5.QtGui import (
    QColor, QFont, QPalette, QFontDatabase, QIcon # QImage, QPainter, QPen, QPolygonF (no longer directly used here)
)
# PyQt5 Widget Modules
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, # Main application, windows, dialogs
    QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QTableView, # Layouts, labels, tabs, table views
    QPushButton, QGroupBox, QFormLayout, QMessageBox, # Buttons, grouping, form layouts, message boxes
    QAbstractItemView # QHeaderView, QStyledItemDelegate, QOpenGLWidget (no longer directly used here)
)
# OpenGL Bindings (PyOpenGL) - These are now primarily used within the widget files themselves
# However, some basic OpenGL imports might still be used if any GL calls remained in ChronoMIDI,
# but it seems they have all been encapsulated. For safety, we can keep common ones or remove if truly unused.
# from OpenGL.GL import (
#     glViewport, glMatrixMode, glLoadIdentity, glOrtho, # Basic OpenGL matrix and viewport setup
#     glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_PROJECTION, GL_MODELVIEW, # Clearing and matrix modes
#     glEnable, glBlendFunc, # Enabling capabilities like blending
#     GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_STRIP, # Blending functions and primitive types
#     glLineWidth, # Setting line thickness
#
#     # VBO related imports (used by Oscilloscope for performance)
#     glGenBuffers, glBindBuffer, glBufferData, glDrawArrays, # Buffer generation, binding, data transfer, drawing
#     glEnableClientState, glDisableClientState, # Enabling/disabling client-side capabilities
#     glVertexPointer, glColorPointer, # Setting pointers to vertex and color data in buffers
#     GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, # Buffer types and usage patterns
#     GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT, # Array types and data types
#
#     # Added for EqualizerGLWidget (using immediate mode, though VBOs are generally preferred)
#     glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS,    # Immediate mode commands for drawing colored quads
#
#     glPushMatrix, glTranslatef, glRotatef, glPopMatrix,  # Matrix stack operations for transformations
#     glScalef, # Matrix operations for scaling and loading matrices
#     GL_LINE_SMOOTH, GL_NICEST, glHint, GL_LINE_SMOOTH_HINT # For antialiasing lines and hint
# )
# OpenGL Utility Library (for gluPerspective)
# from OpenGL.GLU import gluPerspective # Used by Kaleidoscope, now in its own file.


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
        self.tempo_changes = []                     # Stores (absolute_tick, tempo_in_microseconds_per_beat) tuples
        self.time_signature_changes = []            # Stores (absolute_tick, numerator, denominator, cumulative_measures_at_this_tick) tuples
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
        self.lbl_tempo = QLabel(f"Tempo: {self.current_tempo_bpm:.2f} BPM") # Corrected formatting
        self.lbl_ts = QLabel(f"Time Sig: {self.current_time_signature[0]}/{self.current_time_signature[1]}")    # Time Signature
        self.lbl_key = QLabel("N/A")     # Key Signature default
        self.lbl_meta = QLabel("N/A")    # Other metadata default
        # Apply white color style to metadata labels
        for l in (self.lbl_tempo,self.lbl_ts,self.lbl_key,self.lbl_meta): 
            l.setStyleSheet("color:white;")
        # Form layout for metadata labels
        f = QFormLayout(meta)
        f.addRow("Tempo:",self.lbl_tempo)
        f.addRow("Time Sig:",self.lbl_ts)
        f.addRow("Key Sig:",self.lbl_key);
        f.addRow("Other:",self.lbl_meta)
        v.addWidget(meta)

        # Tab Widget for MIDI Event Tables
        self.tabs=QTabWidget(); self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;} "   # No border around the tab pane
            "QTabBar::tab{background:#222;color:white;padding:5px;} "   # Styling for unselected tabs
            "QTabBar::tab:selected{background:#555;}")  # Styling for selected tab
        v.addWidget(self.tabs)
        self.tables = [] # Initialize list of tables

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
        self.midi_meta_update_signal.connect(self._update_meta_display)

    def _update_meta_display(self, tempo_bpm: float, time_signature: tuple):
        """
        Slot to update the tempo and time signature labels in the GUI.
        Connected to midi_meta_update_signal.

        Args:
            tempo_bpm (float): The current tempo in BPM.
            time_signature (tuple): The current time signature (numerator, denominator).
        """
        self.lbl_tempo.setText(f"Tempo: {tempo_bpm:.2f} BPM")
        self.lbl_ts.setText(f"Time Sig: {time_signature[0]}/{time_signature[1]}") # Corrected label

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
    # Construct the path to the icon file.
    # This ensures the icon is found regardless of the current working directory.
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chronomidi.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path)) # Set the application-wide icon
    else:
        print(f"Warning: Application icon not found at {icon_path}")


    # --- Load Custom Font ---
    # Construct the path to the custom font file.
    font_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "PixelCode.ttf")
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
            app.setFont(QFont("Courier New", 9)) # Default fallback
    else:
        print(f"Warning: Custom font not found at {font_fp}. Falling back to system font.")
        app.setFont(QFont("Courier New", 9)) # Default fallback


    # --- Set Application Palette (Dark Theme) ---
    pal = QPalette() # Create a new color palette
    # Set various color roles for a dark theme
    pal.setColor(QPalette.Window, QColor('black'))
    pal.setColor(QPalette.Base, QColor('black'))
    pal.setColor(QPalette.WindowText, QColor('white'))
    pal.setColor(QPalette.Text, QColor('white'))
    pal.setColor(QPalette.Button, QColor('#333'))
    pal.setColor(QPalette.ButtonText, QColor('white'))
    pal.setColor(QPalette.Highlight, QColor('#444'))
    pal.setColor(QPalette.HighlightedText, QColor('white'))
    app.setPalette(pal) # Apply the custom palette to the application

    # Create and show the main ChronoMIDI window
    main_window = ChronoMIDI()
    main_window.show()

    # Start the PyQt event loop. This line transfers control to Qt,
    # and the application will wait for user interactions.
    # sys.exit() ensures a clean exit when the application closes.
    sys.exit(app.exec_())
