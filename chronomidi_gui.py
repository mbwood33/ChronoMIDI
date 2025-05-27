#!/usr/bin/env python3
# chronomidi_gui.py – ChronoMIDI GUI/playback/visualizer

import sys, os, subprocess, random
from collections import deque
import math 

import mido
import numpy as np # Ensure numpy is imported for array operations
import sounddevice as sd
from fluidsynth import Synth

import oscilloscope_computations

from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal,
    QAbstractTableModel, QModelIndex, QVariant, QPointF
)
from PyQt5.QtGui import (
    QColor, QFont, QPalette, QFontDatabase,
    QImage, QPainter, QPen, QPolygonF, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QTableView,
    QPushButton, QGroupBox, QFormLayout, QMessageBox,
    QAbstractItemView, QHeaderView, QStyledItemDelegate, QOpenGLWidget
)
from OpenGL.GL import (
    glViewport, glMatrixMode, glLoadIdentity, glOrtho,
    glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_PROJECTION, GL_MODELVIEW,
    glEnable, glBlendFunc,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_LINE_STRIP,
    glLineWidth, 

    # VBO related imports (used by Oscilloscope)
    glGenBuffers, glBindBuffer, glBufferData, glDrawArrays,
    glEnableClientState, glDisableClientState,
    glVertexPointer, glColorPointer,
    GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
    GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT,

    # Added for EqualizerGLWidget (or other widgets using immediate mode)
    # These are needed for glColor4f, glBegin, glEnd, glVertex2f calls
    glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS 
)

# ─── Helpers ───────────────────────────────────────────────────────────────

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def midi_note_to_name(n: int) -> str:
    return NOTE_NAMES[n % 12] + str((n // 12) - 1)

CONTROL_CHANGE_NAMES = {
    7:'Volume',10:'Pan',11:'Expression',
    64:'Sustain',123:'All Notes Off'
}

class EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        e=super().createEditor(parent, option, index)
        e.setStyleSheet(
            "QLineEdit{background:#444;color:white;}"
            "QLineEdit{selection-background-color:#666;"
            " selection-color:white;}")
        return e



class PandasModel(QAbstractTableModel):
    # ... (PandasModel code) ...
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Vertical:
                return str(self._data.index[section])
        return QVariant()

class ReadOnlyDelegate(QStyledItemDelegate):
    # ... (ReadOnlyDelegate code) ...
    def createEditor(self, parent, option, index):
        return None # Prevents editing

# ─── OpenGL Equalizer Widget (No Change) ──────────────────────────────────

class EqualizerGLWidget(QOpenGLWidget):
    def __init__(self, sr=44100, bands=128, decay=0.92, parent=None):
        super().__init__(parent)
        self.sr, self.bands, self.decay = sr, bands, decay
        self.levels=[0.0]*bands
        QTimer(self,timeout=self.update,interval=1000//60).start()

    def push_audio(self,pcm:np.ndarray):
        mono=pcm.mean(axis=1).astype(np.float32)/32768.0
        spec=np.abs(np.fft.rfft(mono))
        chunk=max(1,len(spec)//self.bands)
        mags=[spec[i*chunk:(i+1)*chunk].mean() for i in range(self.bands)]
        peak=max(mags) or 1.0
        for i,m in enumerate(mags):
            val=m/peak
            self.levels[i]=val if val>self.levels[i] else self.levels[i]*self.decay

    def clear(self): self.levels=[0.0]*self.bands

    # OpenGL
    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0,0,0,1)

    def resizeGL(self,w,h):
        glViewport(0,0,w,h)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(0,w,0,h,-1,1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        w,h=self.width(),self.height()
        slot=w/self.bands; barw=slot*0.9
        for i,lvl in enumerate(self.levels):
            # Using blueish-white as discussed
            glColor4f(lvl, lvl, lvl * 0.8 + 0.2, 1)

            x=i*slot; barh=lvl*h
            glBegin(GL_QUADS)
            glVertex2f(x,0); glVertex2f(x+barw,0)
            glVertex2f(x+barw,barh); glVertex2f(x,barh)
            glVertex2f(x,barh) # Ensure GL_QUADS has 4 vertices
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
    LINEAR_MODE = 0
    CIRCULAR_MODE = 1

    def __init__(self, width=512, height=512, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height) 

        self.hue_offset = 0 
        self.trace_history = deque(maxlen=100) # Max 100 ghosts

        self.audio_queue = deque()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // 60)

        self.current_mode = self.LINEAR_MODE 

        self.vbo_vertex = None
        self.vbo_color = None

        # --- Pre-allocate max possible NumPy arrays ---
        max_lines_to_draw = (self.trace_history.maxlen + 1) * 2 
        max_points_per_line = width // 2 # Assuming this is consistent
        
        self.max_total_vertices = max_lines_to_draw * max_points_per_line

        # Pre-allocate the buffers that Cython will write into
        self.all_vertices_buffer = np.zeros((self.max_total_vertices, 2), dtype=np.float32)
        self.all_colors_buffer = np.zeros((self.max_total_vertices, 4), dtype=np.float32)


    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 1)

        self.vbo_vertex = glGenBuffers(1)
        self.vbo_color = glGenBuffers(1)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1) 
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.current_mode = (self.current_mode + 1) % 2 
            self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        w, h = self.width(), self.height()
        center_x, center_y = w / 2.0, h / 2.0

        while self.audio_queue:
            pcm = self.audio_queue.popleft()
            mono = pcm.mean(axis=1).astype(np.float32) / 32768.0
            self.trace_history.append(mono)

        num_traces = len(self.trace_history)
        if num_traces == 0:
            return

        mono_data_len_ref = len(self.trace_history[-1]) 

        hue_step_per_ghost = 360 / max(1, num_traces + 10)
        max_linear_scroll_dist = 100
        
        max_spiral_radius_offset = min(w, h) * 0.45 
        spiral_angle_offset_per_ghost = 0.05 * math.pi 

        points_to_draw = w // 2 
        sample_step = max(1, mono_data_len_ref // points_to_draw)

        # --- GLOW PARAMETERS ---
        glow_color_base = QColor(200, 200, 255) 
        glow_alpha_factor = 0.25 
        glow_width_linear = 3.0 
        glow_width_circular = 4.0 
        
        glow_offset_x = 0.5 
        glow_offset_y = 0.5
        glow_radius_offset_amount = 2.0 
        
        draw_commands = [] 
        current_vertex_offset = 0 # Tracks current position in self.all_vertices_buffer / self.all_colors_buffer

        # --- Populate Data for Ghosts (Glow + Core) using Cython ---
        for i, mono_data in enumerate(self.trace_history):
            ghost_hue = (self.hue_offset + i * hue_step_per_ghost) % 360
            alpha_val = int(255 * (i / max(1, num_traces))**2 * 0.6 + 5)
            alpha_val = min(255, max(0, alpha_val))

            qt_color = QColor.fromHsv(int(ghost_hue), 220, 255, alpha_val)

            # Pre-calculate float color components for Cython
            glow_r, glow_g, glow_b = glow_color_base.redF(), glow_color_base.greenF(), glow_color_base.blueF()
            glow_a_current = qt_color.alphaF() * glow_alpha_factor 
            
            core_r, core_g, core_b, core_a = qt_color.redF(), qt_color.greenF(), qt_color.blueF(), qt_color.alphaF()

            # --- GLOW Pass ---
            start_index_glow = current_vertex_offset
            oscilloscope_computations.fill_trace_data_cython(
                mono_data, self.all_vertices_buffer, self.all_colors_buffer,
                start_index_glow,
                float(w), float(h), float(center_x), float(center_y),
                points_to_draw, sample_step,
                self.current_mode,
                i, num_traces,
                max_linear_scroll_dist,
                max_spiral_radius_offset, spiral_angle_offset_per_ghost,
                glow_offset_x, glow_offset_y, glow_radius_offset_amount,
                glow_r, glow_g, glow_b, glow_a_current,
                True, # is_glow_pass = True
                False # <--- FIX: is_current_trace = False for ghosts
            )
            draw_commands.append((start_index_glow, points_to_draw, glow_width_linear if self.current_mode == self.LINEAR_MODE else glow_width_circular))
            current_vertex_offset += points_to_draw

            # --- CORE Pass ---
            start_index_core = current_vertex_offset
            oscilloscope_computations.fill_trace_data_cython(
                mono_data, self.all_vertices_buffer, self.all_colors_buffer,
                start_index_core,
                float(w), float(h), float(center_x), float(center_y),
                points_to_draw, sample_step,
                self.current_mode,
                i, num_traces,
                max_linear_scroll_dist,
                max_spiral_radius_offset, spiral_angle_offset_per_ghost,
                glow_offset_x, glow_offset_y, glow_radius_offset_amount, # These are not used for core, but must be passed
                core_r, core_g, core_b, core_a,
                False, # is_glow_pass = False
                False # <--- FIX: is_current_trace = False for ghosts
            )
            draw_commands.append((start_index_core, points_to_draw, 1.0)) # Core width is 1.0
            current_vertex_offset += points_to_draw

        # Update base hue for the next frame's "newest" ghost
        self.hue_offset = (self.hue_offset + 5) % 360

        # --- Populate Data for CURRENT (Newest) Trace (Glow + Core) using Cython ---
        current_mono_data = self.trace_history[-1]

        # --- CURRENT GLOW Pass (always full alpha, white-ish glow) ---
        start_index_current_glow = current_vertex_offset
        oscilloscope_computations.fill_trace_data_cython(
            current_mono_data, self.all_vertices_buffer, self.all_colors_buffer,
            start_index_current_glow,
            float(w), float(h), float(center_x), float(center_y),
            points_to_draw, sample_step,
            self.current_mode,
            0, 1, # i, num_traces - these values will be ignored by Cython for current trace
            max_linear_scroll_dist, # These will be ignored by Cython as it's the current trace
            max_spiral_radius_offset, spiral_angle_offset_per_ghost, # These will be ignored by Cython as it's the current trace
            glow_offset_x, glow_offset_y, glow_radius_offset_amount,
            glow_color_base.redF(), glow_color_base.greenF(), glow_color_base.blueF(), 1.0, # Full alpha for current glow
            True, # is_glow_pass = True
            True # is_current_trace = True
        )
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
            0, 1, # i, num_traces - ignored
            max_linear_scroll_dist, # Ignored
            max_spiral_radius_offset, spiral_angle_offset_per_ghost, # Ignored
            glow_offset_x, glow_offset_y, glow_radius_offset_amount, # Ignored
            1.0, 1.0, 1.0, 1.0, # Pure white core
            False, # is_glow_pass = False
            True # is_current_trace = True
        )
        draw_commands.append((start_index_current_core, points_to_draw, 1.5)) 
        current_vertex_offset += points_to_draw


        # --- Send ALL data to GPU (only two glBufferData calls!) ---
        # Buffer the entire pre-allocated NumPy array.
        # glDrawArrays will use the start_idx and num_pts to pick out the relevant part.
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
        glBufferData(GL_ARRAY_BUFFER, self.all_vertices_buffer.nbytes, self.all_vertices_buffer, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
        glBufferData(GL_ARRAY_BUFFER, self.all_colors_buffer.nbytes, self.all_colors_buffer, GL_DYNAMIC_DRAW)

        # --- Enable Client States and Set Pointers (once) ---
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertex)
        glVertexPointer(2, GL_FLOAT, 0, None) 

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_color)
        glColorPointer(4, GL_FLOAT, 0, None) 

        # --- Execute ALL Draw Calls (only glDrawArrays calls) ---
        for start_idx, num_pts, line_width in draw_commands:
            glLineWidth(line_width)
            glDrawArrays(GL_LINE_STRIP, start_idx, num_pts)

        # --- Disable Client States ---
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


# ─── Visualizer Window (No Change) ────────────────────────────────────────

class VisualizerWindow(QMainWindow):
    def __init__(self, sr):
        super().__init__()
        self.setWindowTitle("Oscilloscope Visualizer")
        self.osc = Oscilloscope(width=512, height=512)
        # expose its audio_queue to ChronoMIDI
        self.audio_queue = self.osc.audio_queue

        cw = QWidget()
        layout = QVBoxLayout(cw)
        layout.addWidget(self.osc)
        self.setCentralWidget(cw)
        self.resize(532, 550)


# ─── Event Table Model (No Change) ────────────────────────────────────────

COLOR_MAP={'note_on':'#8BE9FD','note_off':'#6272A4',
           'control_change':'#FFB86C','program_change':'#50FA7B',
           'pitchwheel':'#FF79C6'}

class EventsModel(QAbstractTableModel):
    HEAD=['Measure','Beat','Dur','Time(s)','Ch','Type','Param']
    def __init__(self,events): super().__init__(); self.ev=events
    def rowCount(self,parent=QModelIndex()): return len(self.ev)
    def columnCount(self,parent=QModelIndex()): return len(self.HEAD)
    def data(self,idx,role=Qt.DisplayRole):
        if not idx.isValid(): return QVariant()
        e=self.ev[idx.row()]; c=idx.column()
        if role==Qt.DisplayRole:
            if c==0: return e['measure']
            if c==1: return f"{e['beat']+1:.2f}"
            if c==2: return f"{e['duration_beats']:.2f}"
            if c==3: return f"{e['time_s']:.3f}"
            if c==4: return e['channel']+1
            if c==5: return e['type']
            if c==6:
                parts=[]
                if e['note'] is not None:
                    parts.append(f"{midi_note_to_name(e['note'])}({e['note']})")
                if e['velocity'] is not None:
                    parts.append(f"vel={e['velocity']}")
                if e['control'] is not None:
                    cc_name=CONTROL_CHANGE_NAMES.get(e['control'],f"CC{e['control']}")
                    parts.append(f"{cc_name}={e['value']}")
                if e['pitch'] is not None:
                    parts.append(f"pitch={e['pitch']}")
                return ', '.join(parts)
        if role==Qt.ForegroundRole and c==5:
            return QColor(COLOR_MAP.get(e['type'],'#F8F8F2'))
        return QVariant()
    def headerData(self,s,o,r):
        if o==Qt.Horizontal and r==Qt.DisplayRole: return self.HEAD[s]
        return QVariant()

# ─── Main Window (No Change) ──────────────────────────────────────────

class ChronoMIDI(QMainWindow):
    event_signal=pyqtSignal(int)
    def __init__(self):
        super().__init__(); self.setWindowTitle("ChronoMIDI"); self.resize(1000,800)

        self.midi_path=None; self.sf2_path=None
        self.sr=44100; self.eq_queue=deque()
        self.events=[]; self.channels=[]
        self.sample_events=[]; self.cur_sample=0
        self.is_playing=False; self.synth=None; self.vis_win=None

        self.stream=sd.OutputStream(samplerate=self.sr,channels=2,dtype='int16',
                                    callback=self._audio_cb,blocksize=1024)

        central=QWidget(); self.setCentralWidget(central)
        v=QVBoxLayout(central)

        self.lbl_file=QLabel("No file loaded"); self.lbl_file.setStyleSheet("color:white;")
        v.addWidget(self.lbl_file)

        meta=QGroupBox("File Metadata")
        self.lbl_tempo=QLabel(); self.lbl_ts=QLabel(); self.lbl_key=QLabel(); self.lbl_meta=QLabel()
        for l in (self.lbl_tempo,self.lbl_ts,self.lbl_key,self.lbl_meta): l.setStyleSheet("color:white;")
        f=QFormLayout(meta); f.addRow("Tempo:",self.lbl_tempo); f.addRow("Time Sig:",self.lbl_ts)
        f.addRow("Key Sig:",self.lbl_key);
        # f.addRow("Other:",self.lbl_meta)
        v.addWidget(meta)

        self.tabs=QTabWidget(); self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;} "
            "QTabBar::tab{background:#222;color:white;padding:5px;} "
            "QTabBar::tab:selected{background:#555;}")
        v.addWidget(self.tabs)

        self.eq=EqualizerGLWidget(sr=self.sr,bands=256); self.eq.setFixedHeight(160)
        v.addWidget(self.eq)
        QTimer(self,timeout=self._drain_eq,interval=1000//60).start()

        h=QHBoxLayout()
        def btn(t,cb): b=QPushButton(t,clicked=cb); b.setStyleSheet("background:#333;color:white;padding:6px;"); h.addWidget(b)
        btn("Open MIDI…",self.open_midi); btn("Load SF2…",self.open_sf2)
        btn("Play",self.play); btn("Pause",self.pause); btn("Stop",self.stop)
        btn("Visualizer…",self.show_vis); btn("Export MP3…",self.export_mp3)
        h.addStretch(); v.addLayout(h)

        self.event_signal.connect(self._hilite)

    # EQ queue
    def _drain_eq(self): 
        while self.eq_queue: self.eq.push_audio(self.eq_queue.popleft())

    # row highlight
    def _hilite(self,idx):
        tab=self.tabs.currentIndex(); tbl=self.tables[tab]
        tbl.selectRow(idx); tbl.scrollTo(tbl.model().index(idx,0),
                                         QAbstractItemView.PositionAtCenter)

    # File/SF2
    def open_midi(self):
        p,_=QFileDialog.getOpenFileName(self,"Open MIDI","","MIDI Files (*.mid *.midi)")
        if not p: return
        self.stop(); self.midi_path=p; self.lbl_file.setText(os.path.basename(p))
        self._load_midi(p)

    def open_sf2(self):
        p,_=QFileDialog.getOpenFileName(self,"Load SoundFont","","SF2 Files (*.sf2)")
        if not p: return
        self.sf2_path=p
        if not self.synth: self.synth=Synth()
        sfid=self.synth.sfload(p); self.synth.program_select(0,sfid,0,0)

    # MIDI parsing
    def _load_midi(self,path):
        mid=mido.MidiFile(path)
        tempo,ts,key,other=500000,(4,4),None,[]
        for m in mido.merge_tracks(mid.tracks):
            if m.is_meta:
                if m.type=='set_tempo': tempo=m.tempo
                elif m.type=='time_signature': ts=(m.numerator,m.denominator)
                elif m.type=='key_signature': key=m.key
                else: other.append(m.type)
        self.lbl_tempo.setText(f"{mido.tempo2bpm(tempo):.2f} BPM")
        self.lbl_ts.setText(f"{ts[0]}/{ts[1]}"); self.lbl_key.setText(key or "N/A")
        self.lbl_meta.setText(', '.join(other) or "N/A")

        tpb=mid.ticks_per_beat; cur_t=tempo; ticks=0; tsec=0.0; ev=[]
        for m in mido.merge_tracks(mid.tracks):
            ticks+=m.time; tsec+=mido.tick2second(m.time,tpb,cur_t)
            if m.is_meta and m.type=='set_tempo': cur_t=m.tempo; continue
            if m.is_meta or not hasattr(m,'channel'): continue
            beat_all=ticks/tpb; meas=int(beat_all//ts[0])+1; beat=beat_all%ts[0]
            ev.append(dict(time_s=tsec,measure=meas,beat=beat,abs=ticks,
                           channel=m.channel,type=m.type,
                           note=getattr(m,'note',None),
                           velocity=getattr(m,'velocity',None),
                           control=getattr(m,'control',None),
                           value=getattr(m,'value',None),
                           pitch=getattr(m,'pitch',None),
                           program=getattr(m,'program',None)))
        active={}
        for i,e in enumerate(ev):
            if e['type']=='note_on' and e['velocity']>0:
                active[(e['channel'],e['note'])]=i; e['duration_beats']=0.0
            elif e['type']=='note_off' and (e['channel'],e['note']) in active:
                j=active.pop((e['channel'],e['note']))
                ev[j]['duration_beats']=(e['abs']-ev[j]['abs'])/tpb
        for e in ev: e.setdefault('duration_beats',0.0)

        self.events=ev; self.channels=sorted({e['channel'] for e in ev})
        self.sample_events=sorted((int(e['time_s']*self.sr),i) for i,e in enumerate(ev))
        self.cur_sample=0; self.eq.clear(); self._build_tables()

    def _build_tables(self):
        self.tabs.clear(); self.tables=[]
        self._add_table("All",self.events)
        for ch in self.channels:
            self._add_table(f"Ch{ch+1}",[e for e in self.events if e['channel']==ch])

    def _add_table(self,title,evts):
        model=EventsModel(evts)
        view=QTableView(); view.setModel(model); view.setItemDelegate(EditDelegate(view))
        view.setStyleSheet(
            "QTableView{background:black;color:white;gridline-color:gray;}"
            "QHeaderView::section{background:#444;color:white;}"
            "QTableView::item:selected{background:#444;color:white;}")
        view.verticalHeader().setDefaultSectionSize(16)
        for c in range(model.columnCount()): view.resizeColumnToContents(c)
        self.tabs.addTab(view,title); self.tables.append(view)

    # Playback
    def play(self):
        if not (self.events and self.synth): return
        self.sample_events=sorted((int(e['time_s']*self.sr),i) for i,e in enumerate(self.events))
        self.cur_sample=0; self.eq.clear(); self.stream.start(); self.is_playing=True

    def pause(self):
        if self.stream.active: self.stream.stop(); self.is_playing=False

    def stop(self):
        if self.stream.active: self.stream.stop()
        self.is_playing=False; self.eq.clear()
        if self.synth:
            try:self.synth.system_reset()
            except: [self.synth.cc(ch,123,0) for ch in range(16)]

    def _audio_cb(self,out,frames,_,__):
        s0,s1=self.cur_sample,self.cur_sample+frames
        while self.sample_events and self.sample_events[0][0]<s1:
            _,idx=self.sample_events.pop(0); e=self.events[idx]; ch=e['channel']
            if e['type']=='note_on': self.synth.noteon(ch,e['note'],e['velocity'])
            elif e['type']=='note_off': self.synth.noteoff(ch,e['note'])
            elif e['type']=='control_change': self.synth.cc(ch,e['control'],e['value'])
            elif e['type']=='program_change': self.synth.program_change(ch,e['program'])
            elif e['type']=='pitchwheel': self.synth.pitch_bend(ch,e['pitch'])
            self.event_signal.emit(idx)
        self.cur_sample=s1
        pcm=np.frombuffer(self.synth.get_samples(frames),dtype=np.int16).reshape(-1,2)
        out[:]=pcm
        if self.is_playing:
            self.eq_queue.append(pcm.copy())
            if self.vis_win: self.vis_win.audio_queue.append(pcm.copy()) # Audio data sent to Oscilloscope

    # Visualizer / export
    def show_vis(self):
        if self.vis_win is None: self.vis_win=VisualizerWindow(sr=self.sr)
        self.vis_win.show()

    def export_mp3(self):
        if not (self.midi_path and self.sf2_path):
            QMessageBox.warning(self,"Export","Load MIDI & SoundFont first."); return
        fn,_=QFileDialog.getSaveFileName(self,"Save MP3","","MP3 Files (*.mp3)")
        if not fn: return
        if not fn.lower().endswith(".mp3"): fn+=".mp3"
        wav=fn[:-4]+"_tmp.wav"
        try:
            subprocess.run(["fluidsynth","-q","-i","-F",wav,"-r",str(self.sr),
                            self.sf2_path,self.midi_path],check=True)
            subprocess.run(["ffmpeg","-y","-i",wav,"-codec:a","libmp3lame","-qscale:a","2",fn],check=True)
        except FileNotFoundError as e:
            QMessageBox.critical(self,"Export",f"Tool not found:\n{e}"); return
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self,"Export",f"Export failed:\n{e}"); return
        finally: 
            if os.path.exists(wav): os.remove(wav)
        QMessageBox.information(self,"Export",f"Saved {fn}")

# ─── Application bootstrap ───────────────────────────────────────────────

if __name__=="__main__":
    app=QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(__file__), "ChronoMIDI.png")   # Load app icon
    app.setWindowIcon(QIcon(icon_path))     # Set app icon    
    font_fp=os.path.join(os.path.dirname(__file__),"fonts","PixelCode.ttf")
    if os.path.exists(font_fp):
        fid=QFontDatabase.addApplicationFont(font_fp)
        fam=QFontDatabase.applicationFontFamilies(fid)
        if fam: app.setFont(QFont(fam[0],9))
    else:
        app.setFont(QFont("Courier New",9))

    pal=QPalette()
    pal.setColor(QPalette.Window,QColor('black'))
    pal.setColor(QPalette.Base,QColor('black'))
    pal.setColor(QPalette.WindowText,QColor('white'))
    pal.setColor(QPalette.Text,QColor('white'))
    pal.setColor(QPalette.Button,QColor('#333'))
    pal.setColor(QPalette.ButtonText,QColor('white'))
    pal.setColor(QPalette.Highlight,QColor('#444'))
    pal.setColor(QPalette.HighlightedText,QColor('white'))
    app.setPalette(pal)

    ChronoMIDI().show(); sys.exit(app.exec_())