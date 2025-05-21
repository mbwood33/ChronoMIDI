#!/usr/bin/env python3
# chronomidi_gui.py
# ChronoMIDI GUI with sample‑accurate MIDI scheduling via sounddevice callback

import sys
import os
import mido
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QFormLayout, QStyledItemDelegate,
    QAbstractItemView, QHeaderView
)
from PyQt5.QtGui import QColor, QFont, QPalette, QFontDatabase
from PyQt5.QtCore import pyqtSignal
from fluidsynth import Synth

# Note name conversion
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Note name conversion function

def midi_note_to_name(note):
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f"{name}{octave}"

# Control Change number to human-readable name mapping
CONTROL_CHANGE_NAMES = {
    0: 'Bank Select', 1: 'Modulation Wheel', 2: 'Breath Controller',
    4: 'Foot Controller', 5: 'Portamento Time', 6: 'Data Entry MSB',
    7: 'Channel Volume', 8: 'Balance', 10: 'Pan',
    11: 'Expression Controller', 12: 'Effect Control 1', 13: 'Effect Control 2',
    64: 'Sustain Pedal', 65: 'Portamento On/Off', 66: 'Sostenuto Pedal',
    67: 'Soft Pedal', 68: 'Legato Footswitch', 69: 'Hold 2',
    70: 'Sound Controller 1', 71: 'Sound Controller 2',
    72: 'Sound Controller 3', 73: 'Sound Controller 4', 74: 'Sound Controller 5',
    75: 'Sound Controller 6', 76: 'Sound Controller 7', 77: 'Sound Controller 8',
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

class EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setStyleSheet(
            "QLineEdit { background-color: #444444; color: white;"
            " selection-background-color: #666666; selection-color: white; }"
        )
        return editor

class ChronoMIDIWindow(QMainWindow):
    event_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI")
        self.resize(1000, 600)

        # Audio scheduling
        self.sr = 44100
        self.sample_events = []  # list of (sample_offset, event_index)
        self.current_sample = 0
        self.stream = None

        # Synth
        self.synth = None

        # Data
        self.events = []
        self.channels = []

        # Connect highlight
        self.event_signal.connect(self.on_event_highlight)

        # Setup UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # File label
        self.label_file = QLabel("No file loaded")
        self.label_file.setStyleSheet("color: white; font-weight: bold;")
        main_layout.addWidget(self.label_file)

        # Metadata panel
        self.meta_group = QGroupBox("File Metadata")
        meta_layout = QFormLayout()
        self.meta_group.setLayout(meta_layout)
        self.label_tempo = QLabel("N/A")
        self.label_time_sig = QLabel("N/A")
        self.label_key = QLabel("N/A")
        self.label_meta = QLabel("N/A")
        for w in (self.label_tempo, self.label_time_sig, self.label_key, self.label_meta):
            w.setStyleSheet("color: white;")
        meta_layout.addRow("Tempo:", self.label_tempo)
        meta_layout.addRow("Time Sig:", self.label_time_sig)
        meta_layout.addRow("Key Sig:", self.label_key)
        meta_layout.addRow("Other Meta:", self.label_meta)
        self.meta_group.setStyleSheet("QGroupBox { color: white; }")
        main_layout.addWidget(self.meta_group)

        # Event tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { background-color: #222; color: white; padding: 5px; }"
            "QTabBar::tab:selected { background-color: #555; }"
        )
        main_layout.addWidget(self.tab_widget)

        # Controls
        ctrl_layout = QHBoxLayout()
        for txt, slot in [
            ("Open MIDI...", self.open_file),
            ("Load SF2...", self.open_sf2),
            ("Play", self.on_play),
            ("Pause", self.on_pause),
            ("Stop", self.on_stop),
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(slot)
            btn.setStyleSheet(
                "QPushButton { background-color: #333; color: white; padding: 5px; }"
                "QPushButton:hover { background-color: #444; }"
            )
            ctrl_layout.addWidget(btn)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)
        # Pre-create audio stream to eliminate startup lag
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype='int16',
            callback=self.audio_callback,
            blocksize=128
        )

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)")
        if path:
            self.on_stop()
            self.label_file.setText(os.path.basename(path))
            if self.synth:
                try: self.synth.system_reset()
                except: pass
            self.load_midi(path)

    def open_sf2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load SoundFont", "", "SF2 Files (*.sf2)")
        if not path:
            return
        if not self.synth:
            self.synth = Synth()
        sfid = self.synth.sfload(path)
        self.synth.program_select(0, sfid, 0, 0)

    def load_midi(self, path):
        # Stop any prior playback
        self.on_stop()
        # Load MIDI
        mid = mido.MidiFile(path)
        # Extract file-level metadata
        tempo = 500000
        ts = (4, 4)
        key = None
        others = []
        for msg in mido.merge_tracks(mid.tracks):
            if msg.is_meta:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'time_signature':
                    ts = (msg.numerator, msg.denominator)
                elif msg.type == 'key_signature':
                    key = msg.key
                else:
                    others.append(msg.type)
        self.label_tempo.setText(f"{mido.tempo2bpm(tempo):.2f} BPM")
        self.label_time_sig.setText(f"{ts[0]}/{ts[1]}")
        self.label_key.setText(key or "N/A")
        self.label_meta.setText(', '.join(others) or "N/A")
        # Prepare event timing
        tpb = mid.ticks_per_beat
        curr_t = tempo
        abs_ticks = 0
        abs_time = 0.0
        evs = []
        # Build event list
        for msg in mido.merge_tracks(mid.tracks):
            delta = msg.time
            abs_ticks += delta
            abs_time += mido.tick2second(delta, tpb, curr_t)
            # Dynamic tempo changes
            if msg.is_meta and msg.type == 'set_tempo':
                curr_t = msg.tempo
                continue
            if msg.is_meta:
                continue
            # Skip non-channel messages
            if not hasattr(msg, 'channel'):
                continue
            # Compute measure & beat (1-based)
            tb = abs_ticks / tpb
            meas = int(tb // ts[0]) + 1
            beat = tb % ts[0]
            ev = {
                'time_s': abs_time,
                'measure': meas,
                'beat': beat,
                'channel': msg.channel,
                'type': msg.type,
                'note': getattr(msg, 'note', None),
                'velocity': getattr(msg, 'velocity', None),
                'control': getattr(msg, 'control', None),
                'value': getattr(msg, 'value', None),
                'pitch': getattr(msg, 'pitch', None),
                'program': getattr(msg, 'program', None),
                'abs_ticks': abs_ticks,
            }
            evs.append(ev)
        # Compute note durations in beats
        active = {}
        for i, e in enumerate(evs):
            if e['type'] == 'note_on' and e['velocity'] > 0:
                active[(e['channel'], e['note'])] = i
                e['duration_beats'] = 0.0
            elif e['type'] == 'note_off' and (e['channel'], e['note']) in active:
                j = active.pop((e['channel'], e['note']))
                delta_ticks = e['abs_ticks'] - evs[j]['abs_ticks']
                evs[j]['duration_beats'] = delta_ticks / tpb
        # Default duration for any that remain
        for e in evs:
            if 'duration_beats' not in e:
                e['duration_beats'] = 0.0
        # Save and prepare for playback
        self.events = evs
        self.channels = sorted({e['channel'] for e in evs})
        self.sample_events = sorted(
            (int(e['time_s'] * self.sr), idx)
            for idx, e in enumerate(self.events)
        )
        self.current_sample = 0
        # Refresh UI
        self.update_event_tabs()

    def update_event_tabs(self):
        self.tab_widget.clear(); self.tables=[]
        self.tab_widget.addTab(self._make_table(self.events),"All")
        for ch in self.channels:
            sub=[e for e in self.events if e['channel']==ch]
            self.tab_widget.addTab(self._make_table(sub),f"Ch{ch}")

    def _make_table(self, evts):
        tb=QTableWidget(); self.tables.append(tb)
        tb.setItemDelegate(EditDelegate(tb)); tb.setFont(QApplication.font())
        tb.setStyleSheet(
            "QTableWidget { background-color: black; color: white; gridline-color: gray; }"
            "QHeaderView::section { background-color: #444444; color: white; font-weight: bold; }"
            "QTableWidget::item:selected { background-color: #444444; color: white; }"
        )
        # add Duration column
        tb.setColumnCount(7)
        tb.setHorizontalHeaderLabels(['Measure','Beat','Dur','Time(s)','Ch','Type','Param'])
        tb.setRowCount(len(evts))
        hdr=tb.horizontalHeader()
        for i in range(3): hdr.setSectionResizeMode(i,QHeaderView.ResizeToContents)
        for i in range(3,6): hdr.setSectionResizeMode(i,QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6,QHeaderView.Interactive); tb.setColumnWidth(6,200)
        colors={'note_on':'#8BE9FD','note_off':'#6272A4','control_change':'#FFB86C','program_change':'#50FA7B','pitchwheel':'#FF79C6'}
        for r,e in enumerate(evts):
            tb.setItem(r,0,QTableWidgetItem(str(e['measure'])))
            # beat 1-based
            tb.setItem(r,1,QTableWidgetItem(f"{e['beat']+1:.2f}"))
            tb.setItem(r,2,QTableWidgetItem(f"{e['duration_beats']:.2f}"))
            tb.setItem(r,3,QTableWidgetItem(f"{e['time_s']:.3f}"))
            tb.setItem(r,4,QTableWidgetItem(str(e['channel']+1)))  # display channels 1-based
            ti=QTableWidgetItem(e['type']); ti.setForeground(QColor(colors.get(e['type'],'#F8F8F2'))); tb.setItem(r,5,ti)
            parts=[]
            if e['note']!=None: parts.append(midi_note_to_name(e['note'])+f"({e['note']})")
            if e['velocity']!=None: parts.append(f"vel={e['velocity']}")
            if e['control']!=None:
                cc=CONTROL_CHANGE_NAMES.get(e['control'],f"CC{e['control']}")
                parts.append(f"{cc}={e['value']}")
            if e['pitch']!=None: parts.append(f"pitch={e['pitch']}")
            tb.setItem(r,6,QTableWidgetItem(', '.join(parts)))
        tb.verticalHeader().setDefaultSectionSize(18)
        return tb

    def on_event_highlight(self, idx):
        ti=self.tab_widget.currentIndex()
        if ti<len(self.tables):
            tbl=self.tables[ti]; tbl.clearSelection(); tbl.selectRow(idx)
            it=tbl.item(idx,0)
            if it: tbl.scrollToItem(it,QAbstractItemView.PositionAtCenter)

    def on_play(self):
        if not self.events or not self.synth: return
        if self.stream and self.stream.active: return
        if not self.stream:
            self.sample_events=sorted((int(e['time_s']*self.sr),i) for i,e in enumerate(self.events))
            self.current_sample=0
            self.stream=sd.OutputStream(samplerate=self.sr,channels=2,dtype='int16',callback=self.audio_callback,blocksize=128)
        self.stream.start()

    def audio_callback(self,outdata,frames,time_info,status):
        start=self.current_sample; end=start+frames
        while self.sample_events and self.sample_events[0][0]<end:
            off,idx=self.sample_events.pop(0)
            e=self.events[idx]; t=e['type']
            if t=='note_on': self.synth.noteon(e['channel'],e['note'],e['velocity'])
            elif t=='note_off': self.synth.noteoff(e['channel'],e['note'])
            elif t=='control_change': self.synth.cc(e['channel'],e['control'],e['value'])
            elif t=='program_change' and e['program']!=None: self.synth.program_change(e['channel'],e['program'])
            elif t=='pitchwheel' and e['pitch']!=None: self.synth.pitch_bend(e['channel'],e['pitch'])
            self.event_signal.emit(idx)
        self.current_sample=end
        pcm_data=self.synth.get_samples(frames)
        pcm=np.frombuffer(pcm_data,dtype=np.int16).reshape(-1,2)
        outdata[:]=pcm

    def on_pause(self):
        if self.stream and self.stream.active: self.stream.stop()

    def on_stop(self):
        if self.stream: self.stream.stop(); self.stream.close(); self.stream=None
        if self.synth:
            try: self.synth.system_reset()
            except: 
                for ch in range(16): self.synth.cc(ch,123,0)

if __name__=='__main__':
    app=QApplication(sys.argv)
    script_dir=os.path.dirname(os.path.abspath(__file__))
    fp=os.path.join(script_dir,'fonts','PixelCode.ttf')
    if os.path.exists(fp):
        fid=QFontDatabase.addApplicationFont(fp); fam=QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0],10))
    else: app.setFont(QFont('Courier New',10))
    pal=QPalette(); pal.setColor(QPalette.Window,QColor('black')); pal.setColor(QPalette.WindowText,QColor('white'))
    pal.setColor(QPalette.Base,QColor('black')); pal.setColor(QPalette.Text,QColor('white'))
    pal.setColor(QPalette.Button,QColor('#333')); pal.setColor(QPalette.ButtonText,QColor('white'))
    pal.setColor(QPalette.Highlight,QColor('#444444')); pal.setColor(QPalette.HighlightedText,QColor('white'))
    app.setPalette(pal)
    w=ChronoMIDIWindow(); w.show(); sys.exit(app.exec_())
