#!/usr/bin/env python3
# chronomidi_gui.py
# ChronoMIDI GUI with sample-accurate MIDI scheduling via sounddevice callback,
# high-resolution OpenGL equalizer, and a QAbstractTableModel/QTableView for events

import sys
import os
from collections import deque

import mido
import numpy as np
import sounddevice as sd
from fluidsynth import Synth

from PyQt5.QtCore import (
    pyqtSignal, QTimer, QRectF, Qt,
    QAbstractTableModel, QModelIndex, QVariant
)
from PyQt5.QtGui import QColor, QFont, QPalette, QFontDatabase, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableView,
    QGroupBox, QFormLayout, QStyledItemDelegate,
    QAbstractItemView, QHeaderView, QOpenGLWidget
)
from OpenGL.GL import (
    glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    glViewport, glMatrixMode, GL_PROJECTION, glLoadIdentity, glOrtho,
    GL_MODELVIEW, glBegin, GL_QUADS, glColor4f, glVertex2f, glEnd,
    glEnable, glBlendFunc,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
)

# ─── Helpers ───────────────────────────────────────────────────────────────

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def midi_note_to_name(n):
    octave = (n // 12) - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"

CONTROL_CHANGE_NAMES = {
    7: 'Volume', 10: 'Pan', 11: 'Expression', 64: 'Sustain', 123: 'All Notes Off'
    # extend as desired...
}

class EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setStyleSheet(
            "QLineEdit { background-color: #444444; color: white; }"
            "QLineEdit { selection-background-color: #666666; selection-color: white; }"
        )
        return editor

# ─── Equalizer ─────────────────────────────────────────────────────────────

class EqualizerGLWidget(QOpenGLWidget):
    """OpenGL equalizer with fade‐out trails for a reactive, blurred look."""

    def __init__(self,
                 sr: int = 44100,
                 bands: int = 64,
                 decay: float = 0.90,
                 fmin: float = 0.0,
                 scale: str = "linear",
                 fade_alpha: float = 0.1,
                 parent=None):
        super().__init__(parent)
        self.sr        = sr
        self.bands     = bands
        self.decay     = decay
        self.fmin      = fmin
        self.scale     = scale.lower()
        self.fade_alpha= fade_alpha   # opacity of the fading quad
        self.levels    = [0.0] * bands

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // 60)  # repaint @60FPS

    def clear(self):
        self.levels = [0.0] * self.bands

    def push_audio(self, pcm: np.ndarray):
        mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
        spec  = np.abs(np.fft.rfft(mono, n=len(mono)))
        freqs = np.fft.rfftfreq(len(mono), 1/self.sr)
        fmax  = self.sr / 2

        # build band edges
        if self.scale == "log":
            lo = max(self.fmin, 1e-3)
            edges = np.logspace(np.log10(lo), np.log10(fmax), self.bands+1)
        else:
            edges = np.linspace(self.fmin, fmax, self.bands+1)

        mags = []
        for i in range(self.bands):
            lo, hi = edges[i], edges[i+1]
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            mags.append(spec[idx].mean() if idx.size>0 else 0.0)

        m = max(mags) or 1.0
        for i,v in enumerate(mags):
            v /= m
            if v > self.levels[i]:
                self.levels[i] = v
            else:
                self.levels[i] *= self.decay

    def initializeGL(self):
        # enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 1)

    def resizeGL(self, w:int, h:int):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    def paintGL(self):
        # 1) draw a translucent black quad to fade previous frame
        glColor4f(0, 0, 0, self.fade_alpha)
        glBegin(GL_QUADS)
        glVertex2f(0,       0)
        glVertex2f(self.width(), 0)
        glVertex2f(self.width(), self.height())
        glVertex2f(0,       self.height())
        glEnd()

        # 2) draw current bars on top
        w, h   = self.width(), self.height()
        bar_w  = w / self.bands * 0.98  # 98% width
        for i, lvl in enumerate(self.levels):
            r = lvl
            g = lvl * 0.8 + 0.2
            b = lvl * 0.8 + 0.2
            glColor4f(r, g, b, 1.0)
            x0    = i * (w / self.bands)
            bar_h = lvl * h
            glBegin(GL_QUADS)
            glVertex2f(      x0,   0)
            glVertex2f(x0 + bar_w, 0)
            glVertex2f(x0 + bar_w, bar_h)
            glVertex2f(      x0,   bar_h)
            glEnd()

            
# ─── Events Model ──────────────────────────────────────────────────────────

COLOR_MAP = {
    'note_on':'#8BE9FD','note_off':'#6272A4',
    'control_change':'#FFB86C','program_change':'#50FA7B',
    'pitchwheel':'#FF79C6'
}

class EventsModel(QAbstractTableModel):
    HEADERS = ['Measure','Beat','Dur','Time(s)','Ch','Type','Param']

    def __init__(self, events):
        super().__init__()
        self._events = events

    def rowCount(self, parent=QModelIndex()):
        return len(self._events)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        e = self._events[index.row()]
        c = index.column()
        if role == Qt.DisplayRole:
            if c == 0:   return str(e['measure'])
            if c == 1:   return f"{e['beat']+1:.2f}"
            if c == 2:   return f"{e['duration_beats']:.2f}"
            if c == 3:   return f"{e['time_s']:.3f}"
            if c == 4:   return str(e['channel']+1)
            if c == 5:   return e['type']
            if c == 6:
                parts = []
                if e['note']    is not None: parts.append(midi_note_to_name(e['note'])+f"({e['note']})")
                if e['velocity']is not None: parts.append(f"vel={e['velocity']}")
                if e['control'] is not None:
                    cc = CONTROL_CHANGE_NAMES.get(e['control'],f"CC{e['control']}")
                    parts.append(f"{cc}={e['value']}")
                if e['pitch']   is not None: parts.append(f"pitch={e['pitch']}")
                return ', '.join(parts)
        if role == Qt.ForegroundRole and c == 5:
            return QColor(COLOR_MAP.get(e['type'], '#F8F8F2'))
        return QVariant()

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return QVariant()

# ─── Main Window ──────────────────────────────────────────────────────────

class ChronoMIDIWindow(QMainWindow):
    event_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI")
        self.resize(1000, 800)

        # audio / synth
        self.sr = 44100
        self.sample_events = []
        self.current_sample = 0
        self.is_playing = False
        self.audio_queue = deque()
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=2, dtype='int16',
            callback=self.audio_callback, blocksize=1024
        )
        self.synth = None

        # data
        self.events = []
        self.channels = []

        # highlight signal
        self.event_signal.connect(self.on_event_highlight)

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # File label
        self.label_file = QLabel("No file loaded")
        self.label_file.setStyleSheet("color: white;")
        main_layout.addWidget(self.label_file)

        # Metadata
        meta = QGroupBox("File Metadata")
        ml = QFormLayout()
        meta.setLayout(ml)
        self.label_tempo    = QLabel("N/A")
        self.label_time_sig = QLabel("N/A")
        self.label_key      = QLabel("N/A")
        self.label_meta     = QLabel("N/A")
        for w in (self.label_tempo, self.label_time_sig,
                  self.label_key, self.label_meta):
            w.setStyleSheet("color: white;")
        ml.addRow("Tempo:",     self.label_tempo)
        ml.addRow("Time Sig:",  self.label_time_sig)
        ml.addRow("Key Sig:",   self.label_key)
        # ml.addRow("Other Meta:",self.label_meta)
        meta.setStyleSheet("QGroupBox { color: white; }")
        main_layout.addWidget(meta)

        # Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { background: #222; color: white; padding: 5px; }"
            "QTabBar::tab:selected { background: #555; }"
        )
        main_layout.addWidget(self.tab_widget)

        # Equalizer
        # 512 bars, linear scale from 0–22 kHz:
        self.eq = EqualizerGLWidget(sr=44100,
                                    bands=256,
                                    decay=0.92,
                                    fmin=0.0,
                                    scale="linear")
        self.eq.setFixedHeight(200)
        main_layout.addWidget(self.eq)

        # drain audio queue → EQ
        self.eq_timer = QTimer(self)
        self.eq_timer.timeout.connect(self._drain_audio_queue)
        self.eq_timer.start(1000 // 60)

        # Controls
        ctrl = QHBoxLayout()
        for txt, slot in [
            ("Open MIDI...", self.open_file),
            ("Load SF2...",  self.open_sf2),
            ("Play",         self.on_play),
            ("Pause",        self.on_pause),
            ("Stop",         self.on_stop),
        ]:
            b = QPushButton(txt)
            b.clicked.connect(slot)
            b.setStyleSheet("background:#333; color:white; padding:5px;")
            ctrl.addWidget(b)
        ctrl.addStretch()
        main_layout.addLayout(ctrl)

    # ─── Audio Queue → EQ ───────────────────────────────────────────────────

    def _drain_audio_queue(self):
        while self.audio_queue:
            chunk = self.audio_queue.popleft()
            self.eq.push_audio(chunk)

    # ─── File / SF2 ──────────────────────────────────────────────────────────

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)"
        )
        if not path:
            return
        self.on_stop()
        self.label_file.setText(os.path.basename(path))
        if self.synth:
            try: self.synth.system_reset()
            except: pass
        self.load_midi(path)

    def open_sf2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load SoundFont", "", "SF2 Files (*.sf2)"
        )
        if not path:
            return
        if not self.synth:
            self.synth = Synth()
        sfid = self.synth.sfload(path)
        self.synth.program_select(0, sfid, 0, 0)

    # ─── MIDI Loading ────────────────────────────────────────────────────────

    def load_midi(self, path):
        self.on_stop()
        mid = mido.MidiFile(path)
        tempo, ts, key, others = 500000, (4,4), None, []
        for msg in mido.merge_tracks(mid.tracks):
            if msg.is_meta:
                if msg.type=='set_tempo':     tempo=msg.tempo
                elif msg.type=='time_signature': ts=(msg.numerator,msg.denominator)
                elif msg.type=='key_signature':  key=msg.key
                else: others.append(msg.type)
        self.label_tempo.setText(f"{mido.tempo2bpm(tempo):.2f} BPM")
        self.label_time_sig.setText(f"{ts[0]}/{ts[1]}")
        self.label_key.setText(key or "N/A")
        self.label_meta.setText(', '.join(others) or "N/A")

        tpb, curr_t, abs_ticks, abs_time = (
            mid.ticks_per_beat, tempo, 0, 0.0
        )
        evs = []
        for msg in mido.merge_tracks(mid.tracks):
            delta = msg.time
            abs_ticks += delta
            abs_time  += mido.tick2second(delta, tpb, curr_t)
            if msg.is_meta and msg.type=='set_tempo':
                curr_t = msg.tempo
                continue
            if msg.is_meta or not hasattr(msg,'channel'):
                continue
            tb   = abs_ticks / tpb
            meas = int(tb // ts[0]) + 1
            beat = tb % ts[0]
            evs.append({
                'time_s': abs_time, 'measure':meas, 'beat':beat,
                'channel': msg.channel, 'type':msg.type,
                'note':getattr(msg,'note',None),
                'velocity':getattr(msg,'velocity',None),
                'control':getattr(msg,'control',None),
                'value':getattr(msg,'value',None),
                'pitch':getattr(msg,'pitch',None),
                'program':getattr(msg,'program',None),
                'abs_ticks':abs_ticks
            })
        # durations
        active = {}
        for i, e in enumerate(evs):
            if e['type']=='note_on' and e['velocity']>0:
                active[(e['channel'],e['note'])]=i; e['duration_beats']=0.0
            elif e['type']=='note_off' and (e['channel'],e['note']) in active:
                j=active.pop((e['channel'],e['note']))
                dt = e['abs_ticks'] - evs[j]['abs_ticks']
                evs[j]['duration_beats'] = dt / tpb
        for e in evs:
            if 'duration_beats' not in e:
                e['duration_beats'] = 0.0

        self.events = evs
        self.channels = sorted({e['channel'] for e in evs})
        self.sample_events = sorted(
            (int(e['time_s']*self.sr), idx)
            for idx, e in enumerate(self.events)
        )
        self.current_sample = 0
        self.eq.clear()
        self.update_event_tabs()

    # ─── Table View ─────────────────────────────────────────────────────────

    def update_event_tabs(self):
        self.tab_widget.clear()
        self.tables = []

        # All events
        model = EventsModel(self.events)
        view  = QTableView()
        view.setModel(model)
        view.setItemDelegate(EditDelegate(view))
        view.verticalHeader().setDefaultSectionSize(16)
        view.setStyleSheet(
            "QTableView { background-color: black; color: white; gridline-color: gray; }"
            "QHeaderView::section { background-color: #444444; color: white; }"
            "QTableView::item:selected { background-color: #444444; color: white; }"
        )
        for col in range(model.columnCount()):
            view.resizeColumnToContents(col)
        self.tab_widget.addTab(view, "All")
        self.tables.append(view)

        # Per-channel tabs
        for ch in self.channels:
            sub = [e for e in self.events if e['channel']==ch]
            m2  = EventsModel(sub)
            v2  = QTableView()
            v2.setModel(m2)
            v2.setItemDelegate(EditDelegate(v2))
            v2.verticalHeader().setDefaultSectionSize(16)
            v2.setStyleSheet(view.styleSheet())
            for col in range(m2.columnCount()):
                v2.resizeColumnToContents(col)
            self.tab_widget.addTab(v2, f"Ch{ch+1}")
            self.tables.append(v2)

    # ─── Playback & Highlight ───────────────────────────────────────────────

    def on_event_highlight(self, idx):
        ti  = self.tab_widget.currentIndex()
        tbl = self.tables[ti]
        tbl.clearSelection()
        tbl.selectRow(idx)
        tbl.scrollTo(tbl.model().index(idx, 0), QAbstractItemView.PositionAtCenter)

    def on_play(self):
        if not (self.events and self.synth):
            return
        self.sample_events = sorted(
            (int(e['time_s']*self.sr), idx)
            for idx, e in enumerate(self.events)
        )
        self.current_sample = 0
        self.eq.clear()
        self.stream.start()
        self.is_playing = True

    def audio_callback(self, outdata, frames, time_info, status):
        start = self.current_sample
        end   = start + frames
        while self.sample_events and self.sample_events[0][0] < end:
            off, idx = self.sample_events.pop(0)
            e = self.events[idx]; t = e['type']
            if t=='note_on':       self.synth.noteon(e['channel'],e['note'],e['velocity'])
            elif t=='note_off':    self.synth.noteoff(e['channel'],e['note'])
            elif t=='control_change':
                                    self.synth.cc(e['channel'],e['control'],e['value'])
            elif t=='program_change' and e['program'] is not None:
                                    self.synth.program_change(e['channel'],e['program'])
            elif t=='pitchwheel' and e['pitch'] is not None:
                                    self.synth.pitch_bend(e['channel'],e['pitch'])
            self.event_signal.emit(idx)

        self.current_sample = end
        pcm = np.frombuffer(self.synth.get_samples(frames),
                             dtype=np.int16).reshape(-1,2)
        outdata[:] = pcm
        if self.is_playing:
            # enqueue for GUI-thread FFT
            self.audio_queue.append(pcm.copy())

    def on_pause(self):
        if self.stream.active:
            self.stream.stop()
        self.is_playing = False

    def on_stop(self):
        if self.stream.active:
            self.stream.stop()
        self.is_playing = False
        self.eq.clear()
        if self.synth:
            try:    self.synth.system_reset()
            except:
                for ch in range(16): self.synth.cc(ch,123,0)

# ─── Entry Point ─────────────────────────────────────────────────────────

if __name__=='__main__':
    app = QApplication(sys.argv)
    script_dir = os.path.dirname(__file__)
    fp = os.path.join(script_dir, 'fonts', 'PixelCode.ttf')
    if os.path.exists(fp):
        fid = QFontDatabase.addApplicationFont(fp)
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0], 9))
    else:
        app.setFont(QFont('Courier New', 9))

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor('black'))
    pal.setColor(QPalette.WindowText, QColor('white'))
    pal.setColor(QPalette.Base, QColor('black'))
    pal.setColor(QPalette.Text, QColor('white'))
    pal.setColor(QPalette.Button, QColor('#333'))
    pal.setColor(QPalette.ButtonText, QColor('white'))
    pal.setColor(QPalette.Highlight, QColor('#444444'))
    pal.setColor(QPalette.HighlightedText, QColor('white'))
    app.setPalette(pal)

    w = ChronoMIDIWindow()
    w.show()
    sys.exit(app.exec_())
