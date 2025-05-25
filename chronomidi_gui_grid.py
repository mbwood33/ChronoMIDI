#!/usr/bin/env python3
# chronomidi_tracker_gui.py
# ChronoMIDI Tracker-style GUI using PyQt5, dynamic subdivision, and sounddevice scheduling

import sys
import os
import math
import mido
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTableView
)
from PyQt5.QtGui import QColor, QFont, QPalette, QFontDatabase
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, pyqtSignal
from fluidsynth import Synth

# Note name conversion
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_name(note):
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f"{name}{octave}"

# Partial CC mapping
CONTROL_CHANGE_NAMES = {1: 'Modulation', 7: 'Volume', 10: 'Pan', 11: 'Expression',
                        64: 'Sustain', 91: 'Reverb', 93: 'Chorus'}

class TrackerModel(QAbstractTableModel):
    def __init__(self, grid, subdivisions, channels, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.subdiv = subdivisions
        self.channels = channels
        # We will have: 1 Pos col + for each channel 4 subcols [Note,Prg,CC,Pitch]
        self.cols = 1 + channels * 4

    def rowCount(self, parent=QModelIndex()):
        return len(self.grid)

    def columnCount(self, parent=QModelIndex()):
        return self.cols

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row, col = index.row(), index.column()
        if col == 0:
            # Position: measure.subdivision (1-based)
            m = row // self.subdiv + 1
            s = row % self.subdiv + 1
            return f"{m:03d}.{s:02d}"
        # Determine channel and subcol
        ch = (col - 1) // 4
        sub = (col - 1) % 4
        events = self.grid[row][ch]
        # Gather specific field
        if sub == 0:  # Note
            for e in events:
                if e['type'] == 'note_on' and e['velocity'] > 0:
                    return midi_note_to_name(e['note'])
            return ''
        elif sub == 1:  # Program
            for e in events:
                if e['type'] == 'program_change' and e['program'] is not None:
                    return str(e['program'])
            return ''
        elif sub == 2:  # CC
            parts = []
            for e in events:
                if e['type'] == 'control_change' and e['control'] is not None:
                    name = CONTROL_CHANGE_NAMES.get(e['control'], f"CC{e['control']}")
                    parts.append(f"{name}={e['value']}")
            return ','.join(parts)
        else:  # sub == 3, Pitch
            for e in events:
                if e['type'] == 'pitchwheel' and e['pitch'] is not None:
                    return str(e['pitch'])
            return ''

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        if section == 0:
            return 'Pos'
        ch = (section - 1) // 4 + 1
        sub = (section - 1) % 4
        labels = ['Note', 'Prg', 'CC', 'PW']
        return f"Ch{ch} {labels[sub]}"

class ChronoMIDITrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI Tracker")
        self.resize(1000, 600)

        # Playback
        self.sr = 44100
        self.synth = None
        self.stream = None
        self.events = []

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # File label
        self.label_file = QLabel("No file loaded")
        layout.addWidget(self.label_file)

        # Tracker view
        self.view = QTableView()
        self.view.setFont(QFont('Courier', 10))
        layout.addWidget(self.view)

        # Controls
        ctrl = QHBoxLayout()
        for text, slot in [("Open MIDI", self.open_midi),
                           ("Load SF2", self.open_sf2),
                           ("Play", self.on_play),
                           ("Stop", self.on_stop)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            ctrl.addWidget(b)
        layout.addLayout(ctrl)

        # Dark palette
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor('black'))
        pal.setColor(QPalette.Base, QColor('black'))
        pal.setColor(QPalette.Text, QColor('white'))
        self.view.setPalette(pal)

    def open_midi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MIDI", "", "MIDI Files (*.mid *.midi)")
        if not path: return
        self.on_stop()
        self.label_file.setText(os.path.basename(path))
        self.load_midi(path)

    def open_sf2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load SF2", "", "SF2 (*.sf2)")
        if not path: return
        if not self.synth: self.synth = Synth()
        sfid = self.synth.sfload(path)
        self.synth.program_select(0, sfid, 0, 0)

    def load_midi(self, path):
        mid = mido.MidiFile(path)
        # Determine dynamic subdivisions per beat
        tpb = mid.ticks_per_beat
        tempo = 500000
        curr_t = tempo
        abs_ticks = 0
        beat_counts = {}
        raw_events = []
        for msg in mido.merge_tracks(mid.tracks):
            abs_ticks += msg.time
            if msg.is_meta and msg.type == 'set_tempo': curr_t = msg.tempo
            if not msg.is_meta and hasattr(msg, 'channel'):
                beat_idx = abs_ticks / tpb
                beat_floor = math.floor(beat_idx)
                beat_counts[beat_floor] = beat_counts.get(beat_floor, 0) + 1
                raw_events.append((abs_ticks, beat_idx, msg))
        # subdivisions = max events in any beat (at least 1)
        subdivisions = max(beat_counts.values(), default=1)
        # rebuild with time_s and tick resolution
        curr_t = tempo; abs_time = 0.0; abs_ticks = 0
        events = []
        for abs_tk, beat_idx, msg in raw_events:
            delta = msg.time
            abs_ticks += delta
            abs_time += mido.tick2second(delta, tpb, curr_t)
            if msg.is_meta and msg.type == 'set_tempo': curr_t = msg.tempo; continue
            e = {'abs_ticks': abs_ticks, 'time_s': abs_time,
                 'type': msg.type, 'channel': msg.channel,
                 'note': getattr(msg, 'note', None),
                 'velocity': getattr(msg, 'velocity', None),
                 'control': getattr(msg, 'control', None),
                 'value': getattr(msg, 'value', None),
                 'pitch': getattr(msg, 'pitch', None),
                 'program': getattr(msg, 'program', None)}
            events.append(e)
        # Determine grid size
        total_beats = max((e['abs_ticks'] / tpb for e in events), default=0)
        total_rows = math.ceil(total_beats * subdivisions)
        num_channels = max((e['channel'] for e in events), default=0) + 1
        # Populate grid
        grid = [[[] for _ in range(num_channels)] for _ in range(total_rows)]
        for e in events:
            beat_f = e['abs_ticks'] / tpb
            row = int(beat_f * subdivisions)
            if 0 <= row < total_rows:
                grid[row][e['channel']].append(e)
        # Set model
        model = TrackerModel(grid, subdivisions, num_channels)
        self.view.setModel(model)
        self.view.resizeColumnToContents(0)
        self.events = events

    def on_play(self):
        if not self.synth or not self.events: return
        if not self.stream:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2,
                                          dtype='int16', callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, outdata, frames, time_info, status):
        data = self.synth.get_samples(frames)
        pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, 2)
        outdata[:] = pcm

    def on_stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close(); self.stream = None
        if self.synth:
            try: self.synth.system_reset()
            except: pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Load pixel font
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(script_dir, 'fonts', 'PixelCode.ttf')
    if os.path.exists(fp):
        fid = QFontDatabase.addApplicationFont(fp)
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0], 10))
    else:
        app.setFont(QFont('Courier New', 10))
    win = ChronoMIDITrackerWindow()
    win.show()
    sys.exit(app.exec_())
