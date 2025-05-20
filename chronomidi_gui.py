#!/usr/bin/env python3
# chronomidi_gui.py
# GUI prototype for ChronoMIDI with real-time event highlighting

import sys
import os
import threading
import time
import mido
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QFormLayout, QStyledItemDelegate,
    QAbstractItemView
)
from PyQt5.QtGui import QColor, QFont, QPalette, QFontDatabase
from PyQt5.QtCore import pyqtSignal, Qt
from fluidsynth import Synth

# Note name conversion
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_name(note):
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f"{name}{octave}"

class EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setStyleSheet(
            "QLineEdit { background-color: #444444; color: white;"
            " selection-background-color: #666666; selection-color: white; }"
        )
        return editor

class ChronoMIDIWindow(QMainWindow):
    # Signal to highlight row index
    event_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI")
        self.resize(900, 600)

        # ––––––––––––––––––––––––
        # Setup the central widget & layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # File display label (now correctly defined)
        self.label_file = QLabel("No file loaded")
        self.label_file.setStyleSheet("color: white; font-weight: bold;")
        main_layout.addWidget(self.label_file)
        # ––––––––––––––––––––––––

        # Data placeholders
        self.events = []
        self.channels = []
        self.midi_path = None
        self.sf2_path = None
        self.synth = None
        self.play_thread = None
        self.pause_flag = False
        self.stop_flag = False
        self.playback_start = 0.0
        self.pause_time = 0.0
        # Tables per tab
        self.tables = []

        # Connect highlight signal
        self.event_signal.connect(self.on_event_highlight)

                # Setup UI
        central = QWidget()
        self.setCentralWidget(central)
        # Create main layout
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(self.label_file)
        # Main layout continued
        

        # Metadata panel
        self.meta_group = QGroupBox("File Metadata")
        meta_layout = QFormLayout()
        self.meta_group.setLayout(meta_layout)
        self.label_tempo = QLabel("N/A")
        self.label_time_sig = QLabel("N/A")
        self.label_key_sig = QLabel("N/A")
        self.label_meta = QLabel("N/A")
        # File name label updated on load
        # Already defined above as self.label_file
        for w in (self.label_tempo, self.label_time_sig, self.label_key_sig, self.label_meta):
            w.setStyleSheet("color: white;")
        meta_layout.addRow("Tempo:", self.label_tempo)
        meta_layout.addRow("Time Sig:", self.label_time_sig)
        meta_layout.addRow("Key Sig:", self.label_key_sig)
        meta_layout.addRow("Other Meta:", self.label_meta)
        self.meta_group.setStyleSheet("QGroupBox { color: white; }")
        main_layout.addWidget(self.meta_group)

        # Event tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { background: #222; color: white; padding: 5px; }"
            "QTabBar::tab:selected { background: #555; }"
        )
        main_layout.addWidget(self.tab_widget)

        # Transport & file controls
        ctrl_layout = QHBoxLayout()
        btn_open = QPushButton("Open MIDI File...")
        btn_open.clicked.connect(self.open_file)
        btn_sf2 = QPushButton("Select SoundFont...")
        btn_sf2.clicked.connect(self.open_sf2)
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        for b in (btn_open, btn_sf2, self.btn_play, self.btn_pause, self.btn_stop):
            b.setStyleSheet(
                "QPushButton { background: #333; color: white; padding: 5px; }"
                "QPushButton:hover { background: #444; }"
            )
            ctrl_layout.addWidget(b)
        ctrl_layout.addStretch()
        main_layout.addLayout(ctrl_layout)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)")
        if path:
            self.midi_path = path
            self.load_midi(path)

    def open_sf2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SoundFont", "", "SoundFont Files (*.sf2)")
        if path:
            self.sf2_path = path

    def load_midi(self, path):
        # Update file label
        self.label_file.setText(os.path.basename(path))
        # Load MIDI and extract metadata
        mid = mido.MidiFile(path)
        # Global metadata
        tempo = 500000
        time_sig = (4, 4)
        key_sig = None
        other_meta = []
        for track in mid.tracks:
            for msg in track:
                if msg.is_meta:
                    if msg.type == 'set_tempo': tempo = msg.tempo
                    elif msg.type == 'time_signature': time_sig = (msg.numerator, msg.denominator)
                    elif msg.type == 'key_signature': key_sig = msg.key
                    else: other_meta.append(msg.type)
        bpm = mido.tempo2bpm(tempo)
        self.label_tempo.setText(f"{bpm:.2f} BPM")
        self.label_time_sig.setText(f"{time_sig[0]}/{time_sig[1]}")
        self.label_key_sig.setText(key_sig or "N/A")
        self.label_meta.setText(', '.join(other_meta) or "N/A")
        # Build event list with dynamic tempo
        ticks_per_beat = mid.ticks_per_beat
        curr_tempo = tempo
        abs_time = 0.0
        abs_ticks = 0
        events = []
        for msg in mido.merge_tracks(mid.tracks):
            delta_ticks = msg.time
            # convert delta ticks to seconds using current tempo
            delta_secs = mido.tick2second(delta_ticks, ticks_per_beat, curr_tempo)
            abs_time += delta_secs
            abs_ticks += delta_ticks
            # handle tempo change
            if msg.is_meta and msg.type == 'set_tempo':
                curr_tempo = msg.tempo
                continue
            if msg.is_meta:
                continue
            # calculate measure and beat
            total_beats = abs_ticks / ticks_per_beat
            measure = int(total_beats // time_sig[0]) + 1
            beat_in_measure = total_beats % time_sig[0]
            events.append({
                'time_s': abs_time,
                'measure': measure,
                'beat': beat_in_measure,
                'channel': getattr(msg, 'channel', 0),
                'type': msg.type,
                'note': getattr(msg, 'note', None),
                'velocity': getattr(msg, 'velocity', None),
                'control': getattr(msg, 'control', None),
                'value': getattr(msg, 'value', None),
                'pitch': getattr(msg, 'pitch', None),
                'program': getattr(msg, 'program', None),
            })
        self.events = events
        self.channels = sorted({e['channel'] for e in events})
        self.update_event_tabs()

    def update_event_tabs(self):
        self.tab_widget.clear()
        self.tables = []
        self.tab_widget.addTab(self._add_table(self.events), "All")
        for ch in self.channels:
            tab = self._add_table([e for e in self.events if e['channel']==ch])
            self.tab_widget.addTab(tab, f"Ch {ch}")

    def _add_table(self, events):
        tb = QTableWidget()
        self.tables.append(tb)
        tb.setItemDelegate(EditDelegate(tb))
        tb.setStyleSheet(
            "QTableWidget { background-color: black; color: white; gridline-color: gray; }"
            "QHeaderView::section { background-color: #444; color: white; font: bold; }"
            "QTableWidget::item:selected { background-color: #444444; color: white; }"
        )
        tb.setFont(QApplication.font())
        tb.setColumnCount(6)
        tb.setHorizontalHeaderLabels(['Measure','Beat','Time(s)','Channel','Type','Param'])
        tb.setRowCount(len(events))
        type_colors = {
            'note_on': QColor('#8BE9FD'), 'note_off': QColor('#6272A4'),
            'control_change': QColor('#FFB86C'),'program_change': QColor('#50FA7B'),
            'pitchwheel': QColor('#FF79C6'),'default': QColor('#F8F8F2')
        }
        for r,e in enumerate(events):
            tb.setItem(r,0,QTableWidgetItem(str(e['measure'])))
            tb.setItem(r,1,QTableWidgetItem(f"{e['beat']:.2f}"))
            tb.setItem(r,2,QTableWidgetItem(f"{e['time_s']:.3f}"))
            tb.setItem(r,3,QTableWidgetItem(str(e['channel'])))
            ti=QTableWidgetItem(e['type']); ti.setForeground(type_colors.get(e['type'],type_colors['default']))
            tb.setItem(r,4,ti)
            parts=[]
            if e['note']!=None: parts.append(midi_note_to_name(e['note'])+f"({e['note']})")
            if e['velocity']!=None: parts.append(f"vel={e['velocity']}")
            if e['control']!=None: parts.append(f"ctrl={e['control']}={e['value']}")
            tb.setItem(r,5,QTableWidgetItem(','.join(parts)))
        tb.verticalHeader().setDefaultSectionSize(18)
        # limit column widths to avoid oversized tables
        from PyQt5.QtWidgets import QHeaderView
        header = tb.horizontalHeader()
        # Columns 0–4 auto-size, Param column interactive with max width
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        tb.setColumnWidth(5, 200)
        return tb

    def on_event_highlight(self, row_index):
        """
        Highlight and scroll to the given row index in the active table.
        """
        # Determine current table based on active tab
        current_tab = self.tab_widget.currentIndex()
        if current_tab < len(self.tables):
            table = self.tables[current_tab]
            # Clear previous selection
            table.clearSelection()
            # Select the row
            table.selectRow(row_index)
            # Scroll to the row
            item = table.item(row_index, 0)
            if item:
                table.scrollToItem(item, QAbstractItemView.PositionAtCenter)


    def on_play(self):
        if not (self.events and self.sf2_path): return
        if self.play_thread and self.play_thread.is_alive():
            if self.pause_flag:
                paused_dur = time.monotonic() - self.pause_time
                self.playback_start += paused_dur
                self.pause_flag = False
            return
        self.pause_flag=False; self.stop_flag=False
        self.playback_start=time.monotonic()
        if not self.synth:
            self.synth=Synth(); self.synth.start(driver="dsound", midi_driver="winmidi")
            sfid=self.synth.sfload(self.sf2_path)
            self.synth.program_select(0,sfid,0,0)
        self.play_thread=threading.Thread(target=self._playback,daemon=True)
        self.play_thread.start()

    def on_pause(self):
        if self.play_thread and self.play_thread.is_alive():
            self.pause_flag=True; self.pause_time=time.monotonic()

    def on_stop(self):
        self.stop_flag=True

    def _playback(self):
        for idx,e in enumerate(self.events):
            if self.stop_flag: break
            while self.pause_flag and not self.stop_flag: time.sleep(0.1)
            if self.stop_flag: break
            target=self.playback_start+e['time_s']
            now=time.monotonic();
            if target>now: time.sleep(target-now)
            t=e['type']
            if t=='note_on': self.synth.noteon(e['channel'],e['note'],e['velocity'])
            elif t=='note_off': self.synth.noteoff(e['channel'],e['note'])
            elif t=='control_change': self.synth.cc(e['channel'],e['control'],e['value'])
            elif t=='program_change' and e['program']!=None: self.synth.program_change(e['channel'],e['program'])
            elif t=='pitchwheel':
                # use captured pitch value
                if e['pitch'] is not None:
                    self.synth.pitch_bend(e['channel'], e['pitch'])
            # emit row index for highlighting
            self.event_signal.emit(idx)
        if self.synth: self.synth.delete(); self.synth=None
        self.pause_flag=False; self.stop_flag=False

if __name__=='__main__':
    app=QApplication(sys.argv)
    # load font
    script_dir=os.path.dirname(os.path.abspath(__file__))
    font_fp=os.path.join(script_dir,'fonts','PixelCode.ttf')
    if os.path.exists(font_fp):
        fid=QFontDatabase.addApplicationFont(font_fp)
        fam=QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0],10))
    else: app.setFont(QFont('Courier New',10))
    # dark palette
    pal=QPalette(); pal.setColor(QPalette.Window,QColor('black')); pal.setColor(QPalette.WindowText,QColor('white'))
    pal.setColor(QPalette.Base,QColor('black')); pal.setColor(QPalette.Text,QColor('white'))
    pal.setColor(QPalette.Button,QColor('#333')); pal.setColor(QPalette.ButtonText,QColor('white'))
    pal.setColor(QPalette.Highlight,QColor('#444444')); pal.setColor(QPalette.HighlightedText,QColor('white'))
    app.setPalette(pal)
    w=ChronoMIDIWindow(); w.show(); sys.exit(app.exec_())
