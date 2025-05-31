# models.py
# Contains data models and delegates for ChronoMIDI.

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QVariant, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QStyledItemDelegate, QLineEdit

# It's assumed that utils.py is in the same directory or Python's path
from utils import midi_note_to_name, CONTROL_CHANGE_NAMES, calculate_beat_measure

# Color map for different MIDI message types in the event table
COLOR_MAP = {
    'note_on': '#8BE9FD',       # Light blue
    'note_off': '#6272A4',      # Dark blue-gray
    'control_change': '#FFB86C',# Orange
    'program_change': '#50FA7B',# Green
    'pitchwheel': '#FF79C6'     # Pink
}

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
        # QLineEdit is implicitly created by super().createEditor for string data,
        # so we don't need to import it unless we were creating it directly.
        # However, it's good practice if we know the type.
        if isinstance(e, QLineEdit):
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
            return QVariant()
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Returns the header data for rows or columns.
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._data.columns[section])
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
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
        """
        return None # Prevents editing

class EventsModel(QAbstractTableModel):
    """
    A custom table model for displaying parsed MIDI events in a QTableView.
    It formats MIDI event data (time, type, parameters) for presentation.
    """
    HEAD = ['Measure', 'Beat', 'Dur', 'Time(s)', 'Ch', 'Type', 'Param'] # Column headers

    def __init__(self, events: list, ticks_per_beat: int, time_signature_changes: list):
        """
        Initializes the model with a list of parsed MIDI event dictionaries.
        """
        super().__init__()
        self.ev = events
        self.ticks_per_beat = ticks_per_beat
        self.time_signature_changes = time_signature_changes

    def rowCount(self, parent=QModelIndex()):
        return len(self.ev)

    def columnCount(self, parent=QModelIndex()):
        return len(self.HEAD)

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return QVariant()

        e = self.ev[idx.row()]
        c = idx.column()

        if role == Qt.DisplayRole:
            if c == 0: # Measure
                measure, _ = calculate_beat_measure(e['abs'], self.ticks_per_beat, self.time_signature_changes)
                return measure
            if c == 1: # Beat
                _, beat = calculate_beat_measure(e['abs'], self.ticks_per_beat, self.time_signature_changes)
                return f"{beat+1:.2f}"
            if c == 2: # Duration
                return f"{e['duration_beats']:.2f}"
            if c == 3: return f"{e['time_s']:.3f}"
            if c == 4:
                return e['channel']+1 if e['channel'] is not None else "N/A"
            if c == 5: return e['type']
            if c == 6: # Parameters column
                parts = []
                if e['note'] is not None:
                    parts.append(f"{midi_note_to_name(e['note'])}({e['note']})")
                if e['velocity'] is not None:
                    parts.append(f"vel={e['velocity']}")
                if e['control'] is not None:
                    cc_name = CONTROL_CHANGE_NAMES.get(e['control'], f"CC{e['control']}")
                    parts.append(f"{cc_name}={e['value']}")
                if e['pitch'] is not None:
                    parts.append(f"pitch={e['pitch']}")
                if e['program'] is not None:
                    parts.append(f"prg={e['program']}")
                return ', '.join(parts)

        elif role == Qt.ForegroundRole and c == 5:
            return QColor(COLOR_MAP.get(e['type'], '#F8F8F2'))

        return QVariant()

    def headerData(self, s, o, r):
        if o == Qt.Horizontal and r == Qt.DisplayRole:
            return self.HEAD[s]
        return QVariant()
