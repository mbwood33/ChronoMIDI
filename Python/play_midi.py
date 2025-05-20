#!/usr/bin/env python3
# chronomidi_gui.py
# GUI prototype for ChronoMIDI using PyQt5 with dark theme and monospaced font

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QGroupBox, QFormLayout, QStyledItemDelegate
)
from PyQt5.QtGui import QColor, QFont, QPalette
import mido

# Note name conversion
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_name(note):
    octave = (note // 12) - 1
    name = NOTE_NAMES[note % 12]
    return f"{name}{octave}"

# Delegate to style cell editor background/text
class EditDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        editor.setStyleSheet(
            "QLineEdit { background-color: #444444; color: white;"
            " selection-background-color: #666666; selection-color: white; }"
        )
        return editor

class ChronoMIDIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoMIDI")
        self.resize(900, 600)

        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Metadata panel
        self.meta_group = QGroupBox("File Metadata")
        meta_layout = QFormLayout()
        self.meta_group.setLayout(meta_layout)
        self.label_tempo = QLabel("N/A")
        self.label_time_sig = QLabel("N/A")
        self.label_key_sig = QLabel("N/A")
        self.label_meta = QLabel("N/A")
        for widget in (self.label_tempo, self.label_time_sig, self.label_key_sig, self.label_meta):
            widget.setStyleSheet("color: white;")
        meta_layout.addRow("Tempo:", self.label_tempo)
        meta_layout.addRow("Time Sig:", self.label_time_sig)
        meta_layout.addRow("Key Sig:", self.label_key_sig)
        meta_layout.addRow("Other Meta:", self.label_meta)
        self.meta_group.setStyleSheet("QGroupBox { color: white; }")
        main_layout.addWidget(self.meta_group)

        # Tab widget for event tables
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabWidget::pane { border: none; }"
            "QTabBar::tab { background: #222; color: white; padding: 5px; }"
            "QTabBar::tab:selected { background: #555; }"
        )
        main_layout.addWidget(self.tab_widget)

        # Controls
        btn_layout = QHBoxLayout()
        btn_open = QPushButton("Open MIDI File...")
        btn_open.clicked.connect(self.open_file)
        btn_open.setStyleSheet(
            "QPushButton { background: #333; color: white; padding: 5px; }"
            "QPushButton:hover { background: #444; }"
        )
        btn_layout.addStretch()
        btn_layout.addWidget(btn_open)
        main_layout.addLayout(btn_layout)

        # Data placeholders
        self.midi_file = None
        self.events = []
        self.channels = []

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open MIDI File", "", "MIDI Files (*.mid *.midi)"
        )
        if path:
            self.load_midi(path)

    def load_midi(self, path):
        mid = mido.MidiFile(path)
        self.midi_file = mid

        # Extract metadata
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

        # Build events list
        ticks_per_beat = mid.ticks_per_beat
        sec_per_tick = tempo / 1e6 / ticks_per_beat
        abs_ticks = 0
        events = []
        for msg in mido.merge_tracks(mid.tracks):
            abs_ticks += msg.time
            if msg.is_meta: continue
            time_s = abs_ticks * sec_per_tick
            total_beats = abs_ticks / ticks_per_beat
            measure = int(total_beats // time_sig[0]) + 1
            beat_in_measure = total_beats % time_sig[0]
            events.append({
                'time_s': time_s,
                'measure': measure,
                'beat': beat_in_measure,
                'channel': getattr(msg, 'channel', 0),
                'type': msg.type,
                'note': getattr(msg, 'note', None),
                'velocity': getattr(msg, 'velocity', None),
                'control': getattr(msg, 'control', None),
                'value': getattr(msg, 'value', None),
            })
        self.events = events
        self.channels = sorted(set(e['channel'] for e in events))
        self.update_event_tabs()

    def update_event_tabs(self):
        self.tab_widget.clear()
        self.tab_widget.addTab(self.create_table(self.events), "All")
        for ch in self.channels:
            ch_ev = [e for e in self.events if e['channel'] == ch]
            self.tab_widget.addTab(self.create_table(ch_ev), f"Ch {ch}")

    def create_table(self, events):
        tb = QTableWidget()
        tb.setItemDelegate(EditDelegate(tb))
        # Table styling
        tb.setStyleSheet(
            "QTableWidget { background-color: black; color: white; gridline-color: gray; }"
            "QHeaderView::section { background-color: #444; color: white; font: bold; }"
            "QTableWidget::item:selected { background-color: #444444; color: white; }"
        )
        tb.setFont(QFont("Courier New", 10))
        tb.setColumnCount(6)
        tb.setHorizontalHeaderLabels(['Measure', 'Beat', 'Time(s)', 'Channel', 'Type', 'Param'])
        tb.setRowCount(len(events))

        type_colors = {
            'note_on': QColor('#8BE9FD'),       # bright cyan
            'note_off': QColor('#6272A4'),      # softer purple
            'control_change': QColor('#FFB86C'), # warm orange
            'program_change': QColor('#50FA7B'), # fresh green
            'pitchwheel': QColor('#FF79C6'),    # vibrant pink
            # default for other events:
            'default': QColor('#F8F8F2'),       # off-white
        }

        for row, e in enumerate(events):
            tb.setItem(row, 0, QTableWidgetItem(str(e['measure'])))
            tb.setItem(row, 1, QTableWidgetItem(f"{e['beat']:.2f}"))
            tb.setItem(row, 2, QTableWidgetItem(f"{e['time_s']:.3f}"))
            tb.setItem(row, 3, QTableWidgetItem(str(e['channel'])))

            type_item = QTableWidgetItem(e['type'])
            type_item.setForeground(type_colors.get(e['type'], QColor('magenta')))
            tb.setItem(row, 4, type_item)

            parts = []
            if e['note'] is not None:
                parts.append(midi_note_to_name(e['note']) + f"({e['note']})")
            if e['velocity'] is not None:
                parts.append(f"vel={e['velocity']}")
            if e['control'] is not None:
                parts.append(f"ctrl={e['control']}={e['value']}")
            tb.setItem(row, 5, QTableWidgetItem(', '.join(parts)))

        tb.resizeColumnsToContents()
        return tb

if __name__ == '__main__':
    import os
    from PyQt5.QtGui import QFontDatabase

    # 1. Create application
    app = QApplication(sys.argv)

    # 2. Load custom pixel font if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(script_dir, 'fonts', 'PixelCode.ttf')  # adjust filename if needed
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            pixel_family = families[0]
            app.setFont(QFont(pixel_family, 10))
    else:
        # Fallback monospace
        app.setFont(QFont('Courier New', 10))

    # 3. Dark palette configuration
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('black'))
    palette.setColor(QPalette.WindowText, QColor('white'))
    palette.setColor(QPalette.Base, QColor('black'))
    palette.setColor(QPalette.Text, QColor('white'))
    palette.setColor(QPalette.Button, QColor('#333'))
    palette.setColor(QPalette.ButtonText, QColor('white'))
    palette.setColor(QPalette.Highlight, QColor('#444444'))
    palette.setColor(QPalette.HighlightedText, QColor('white'))
    app.setPalette(palette)

    # 4. Launch main window
    w = ChronoMIDIWindow()
    w.show()
    sys.exit(app.exec_())
