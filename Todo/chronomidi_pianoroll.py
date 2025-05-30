#!/usr/bin/env python3
# chronomidi_pianoroll.py
# Piano Roll prototype with Open MIDI functionality

import sys
import os
import mido
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget,
    QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene
)
from PyQt5.QtGui import QBrush, QColor, QPen, QPainter
from PyQt5.QtCore import QRectF, Qt, QLineF

class PianoRollView(QGraphicsView):
    def __init__(self, notes, pixels_per_second=200, pixels_per_note=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixels_per_second = pixels_per_second
        self.pixels_per_note = pixels_per_note
        self.notes = notes
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._init_scene()

    def _init_scene(self):
        scene = QGraphicsScene(self)
        self.setScene(scene)
        max_time = 0.0
        # draw only note rectangles; grid drawn in drawBackground
        for note in self.notes:
            pitch = note['pitch']
            start = note['start']
            duration = note['duration']
            max_time = max(max_time, start + duration)
            rect = QRectF(
                start * self.pixels_per_second,
                (127 - pitch) * self.pixels_per_note,
                duration * self.pixels_per_second,
                self.pixels_per_note
            )
            brush = QBrush(QColor(100, 200, 250, 180))
            pen = QPen(QColor(80, 160, 200))
            scene.addRect(rect, pen, brush)
        width = max_time * self.pixels_per_second if max_time > 0 else 5 * self.pixels_per_second
        height = 128 * self.pixels_per_note
        scene.setSceneRect(0, 0, width, height)

    def drawBackground(self, painter, rect):
        # fill background area
        painter.fillRect(rect, QColor(20, 20, 20))
        # draw piano keys on left
        key_width = self.pixels_per_note * 5
        for i in range(128):
            y = i * self.pixels_per_note
            is_sharp = i % 12 in (1, 3, 6, 8, 10)
            color = QColor(80, 80, 80) if is_sharp else QColor(200, 200, 200)
            painter.fillRect(rect.left(), y, key_width, self.pixels_per_note, color)
        # separator line
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.drawLine(QLineF(rect.left() + key_width, rect.top(),
                                rect.left() + key_width, rect.bottom()))
        # draw horizontal grid lines in note area
        top = int(rect.top() // self.pixels_per_note) * self.pixels_per_note
        bottom = int(rect.bottom())
        pen_dark = QPen(QColor(40, 40, 40))
        pen_light = QPen(QColor(60, 60, 60))
        for y in range(top, bottom + self.pixels_per_note, self.pixels_per_note):
            octave_index = ((127 - (y // self.pixels_per_note)) % 12)
            pen = pen_light if octave_index in (1, 3, 6, 8, 10) else pen_dark
            painter.setPen(pen)
            painter.drawLine(QLineF(rect.left() + key_width, y, rect.right(), y))

    def update_notes(self, notes):
        # refresh note list and redraw scene
        self.notes = notes
        self._init_scene()
        self.viewport().update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Roll Prototype")
        self.pixels_per_second = 200
        self.pixels_per_note = 10
        self.notes = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)

        # Open button
        btn_open = QPushButton("Open MIDI...")
        btn_open.clicked.connect(self.open_midi)
        self.layout.addWidget(btn_open)

        # Piano roll view
        self.view = PianoRollView(self.notes, self.pixels_per_second, self.pixels_per_note)
        self.layout.addWidget(self.view)
        self.resize(800, 600)

    def open_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)"
        )
        if not path:
            return
        mid = mido.MidiFile(path)
        notes = []
        ongoing = {}
        tempo = 500000
        tpb = mid.ticks_per_beat
        current_time = 0.0
        # parse events
        for msg in mido.merge_tracks(mid.tracks):
            # delta seconds
            if msg.is_meta and msg.type == 'set_tempo':
                tempo = msg.tempo
            delta = mido.tick2second(msg.time, tpb, tempo)
            current_time += delta
            if msg.type == 'note_on' and msg.velocity > 0:
                ongoing[(msg.channel, msg.note)] = current_time
            elif (msg.type == 'note_off' or (msg.type=='note_on' and msg.velocity==0)):
                key = (msg.channel, msg.note)
                if key in ongoing:
                    start = ongoing.pop(key)
                    notes.append({
                        'pitch': msg.note,
                        'start': start,
                        'duration': current_time - start
                    })
        # update view
        self.view.update_notes(notes)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
