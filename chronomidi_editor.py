#!/usr/bin/env python3
import sys
import os
import mido

from PyQt5.QtCore import QRectF, QPointF, Qt
from PyQt5.QtGui import QColor, QBrush, QPen, QPainter, QFontDatabase, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsItem, QHBoxLayout
)

# ─── Model ────────────────────────────────────────────────────────────────

class Note:
    """Simple note model: start/end in ticks, MIDI pitch, channel, velocity."""
    def __init__(self, start_tick, end_tick, pitch, channel, velocity):
        self.start_tick = start_tick
        self.end_tick   = end_tick
        self.pitch      = pitch
        self.channel    = channel
        self.velocity   = velocity

# ─── Transform ────────────────────────────────────────────────────────────

class TickTransform:
    """Convert between MIDI ticks ↔ scene X coordinates."""
    def __init__(self, ticks_per_beat, pixels_per_beat):
        self.tpb = ticks_per_beat
        self.ppb = pixels_per_beat

    def x_from_tick(self, tick):
        return (tick / self.tpb) * self.ppb

    def tick_from_x(self, x):
        return int((x / self.ppb) * self.tpb)

# ─── NoteItem ─────────────────────────────────────────────────────────────

class NoteItem(QGraphicsRectItem):
    """A draggable, resizable rectangle representing one MIDI note."""
    HANDLE_WIDTH = 6.0

    def __init__(self, note: Note, transform: TickTransform, pixels_per_note: float, color: QColor):
        super().__init__()
        self.note = note
        self.tr   = transform
        self.ppn  = pixels_per_note

        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setBrush(color)
        self.setPen(QPen(Qt.NoPen))

        self._dragging_edge = None
        self.update_rect_from_model()

    def update_rect_from_model(self):
        x = self.tr.x_from_tick(self.note.start_tick)
        w = self.tr.x_from_tick(self.note.end_tick) - x
        y = (127 - self.note.pitch) * self.ppn
        h = self.ppn
        self.setRect(QRectF(x, y, w, h))

    def boundingRect(self):
        br = super().boundingRect()
        return br.adjusted(-self.HANDLE_WIDTH, 0, self.HANDLE_WIDTH, 0)

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        r = self.rect()
        painter.setBrush(QColor(200, 200, 200))
        painter.setPen(Qt.NoPen)
        # left handle
        painter.drawRect(r.left() - self.HANDLE_WIDTH, r.top(),
                         self.HANDLE_WIDTH, r.height())
        # right handle
        painter.drawRect(r.right(), r.top(),
                         self.HANDLE_WIDTH, r.height())

    def mousePressEvent(self, event):
        pos = event.pos()
        r = self.rect()
        if abs(pos.x() - r.left()) <= self.HANDLE_WIDTH:
            self._dragging_edge = 'left'
        elif abs(pos.x() - r.right()) <= self.HANDLE_WIDTH:
            self._dragging_edge = 'right'
        else:
            self._dragging_edge = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_edge:
            delta = event.pos().x() - event.lastPos().x()
            r = self.rect()
            if self._dragging_edge == 'left':
                new_left = r.left() + delta
                if new_left < r.right() - self.HANDLE_WIDTH:
                    r.setLeft(new_left)
            else:
                new_right = r.right() + delta
                if new_right > r.left() + self.HANDLE_WIDTH:
                    r.setRight(new_right)
            self.setRect(r)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        r = self.rect()
        new_start = self.tr.tick_from_x(r.left())
        new_end   = self.tr.tick_from_x(r.right())
        # update model
        self.note.start_tick = new_start
        self.note.end_tick   = new_end
        # snap
        self.update_rect_from_model()
        self._dragging_edge = None
        super().mouseReleaseEvent(event)

# ─── PianoRollView ────────────────────────────────────────────────────────

class PianoRollView(QGraphicsView):
    """GraphicsView that draws a piano-roll grid and holds NoteItems."""
    def __init__(self):
        super().__init__()
        self.pixels_per_note   = 4
        self.pixels_per_beat   = 100
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # black background and repaint all on scroll/zoom
        self.setBackgroundBrush(QBrush(QColor('black')))
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

    def drawBackground(self, painter: QPainter, rect):
        vp = self.viewport().rect()
        # fill entire viewport
        painter.fillRect(vp, QColor('black'))

        # compute visible note range
        vs = self.mapToScene(vp).boundingRect()
        top = int(vs.top() // self.pixels_per_note)
        bot = int(vs.bottom() // self.pixels_per_note) + 1

        light = QPen(QColor(80, 80, 80), 1)
        dark  = QPen(QColor(60, 60, 60), 1)

        for note in range(top, bot):
            y = note * self.pixels_per_note
            vy = self.mapFromScene(QPointF(0, y)).y()
            pen = light if (note % 2) else dark
            painter.setPen(pen)
            painter.drawLine(0, vy, vp.width(), vy)

    def load_notes(self, notes, ticks_per_beat):
        """Clear scene, build a TickTransform, and add NoteItems."""
        self.scene.clear()
        tr = TickTransform(ticks_per_beat, self.pixels_per_beat)

        # color map per channel
        channel_colors = [
            QColor('#8BE9FD'), QColor('#6272A4'), QColor('#FFB86C'),
            QColor('#50FA7B'), QColor('#FF79C6'), QColor('#F8F8F2'),
        ]
        for note in notes:
            color = channel_colors[note.channel % len(channel_colors)]
            item = NoteItem(note, tr, self.pixels_per_note, color)
            self.scene.addItem(item)

# ─── MainWindow ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI Editor")
        self.resize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        btn_open = QPushButton("Open MIDI…")
        btn_open.clicked.connect(self.open_midi)
        layout.addWidget(btn_open)

        self.view = PianoRollView()
        layout.addWidget(self.view)

    def open_midi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)")
        if not path:
            return

        mid = mido.MidiFile(path)
        tpb = mid.ticks_per_beat

        # collect note-on/off pairs
        evs = []
        ongoing = {}
        abs_ticks = 0
        for msg in mido.merge_tracks(mid.tracks):
            abs_ticks += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                ongoing[(msg.channel, msg.note)] = (abs_ticks, msg.velocity)
            elif (msg.type == 'note_off' or (msg.type=='note_on' and msg.velocity==0)):
                key = (msg.channel, msg.note)
                if key in ongoing:
                    start_tick, vel = ongoing.pop(key)
                    evs.append(Note(start_tick, abs_ticks,
                                    pitch=msg.note,
                                    channel=msg.channel,
                                    velocity=vel))
        # load into view
        self.view.load_notes(evs, tpb)

# ─── Entry Point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # optional: load pixel font
    script = os.path.dirname(__file__)
    fp = os.path.join(script, 'fonts', 'PixelCode.ttf')
    if os.path.exists(fp):
        fid = QFontDatabase.addApplicationFont(fp)
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam:
            app.setFont(QFont(fam[0], 9))
    else:
        app.setFont(QFont('Courier New', 9))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
