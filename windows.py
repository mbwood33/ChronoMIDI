# windows.py
# Contains QMainWindow classes for separate visualizer windows.

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from oscilloscope_widget import Oscilloscope
from kaleidoscope_widget import KaleidoscopeVisualizerGLWidget

class VisualizerWindow(QMainWindow):
    """
    A separate QMainWindow that hosts the Oscilloscope visualization.
    It provides a dedicated window for the oscilloscope.
    """
    def __init__(self, sr):
        """
        Initializes the VisualizerWindow.

        Args:
            sr (int): Sample rate of the audio, passed to the Oscilloscope.
        """
        super().__init__()
        self.setWindowTitle("Oscilloscope Visualizer")

        # Create an instance of the Oscilloscope widget
        self.osc = Oscilloscope(width=512, height=512)

        # Expose the oscilloscope's audio queue to the main ChronoMIDI window
        # This allows ChronoMIDI to push audio data directly to the visualizer
        self.audio_queue = self.osc.audio_queue

        # Set up the central widget and layout
        cw = QWidget() # Create a central widget
        layout = QVBoxLayout(cw) # Create a vertical layout for the central widget
        layout.addWidget(self.osc) # Add the oscilloscope to the layout
        self.setCentralWidget(cw) # Set the central widget of the QMainWindow
        self.resize(532, 550) # Set a fixed size for the visualizer window (slightly larger than oscilloscope)

class KaleidoscopeVisualizerWindow(QMainWindow):
    """
    A separate QMainWindow that hosts the KaleidoscopeVisualizerGLWidget.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaleidoscope Visualizer")
        self.kv = KaleidoscopeVisualizerGLWidget()
        # Removed: self.audio_queue = self.kv.audio_queue  # Expose the audio queue
        cw = QWidget()
        layout = QVBoxLayout(cw)
        layout.addWidget(self.kv)
        self.setCentralWidget(cw)
        self.resize(600, 600)   # Set a reasonable default size

    def showEvent(self, event):
        """
        Overrides showEvent to clear particles when the window is shown.
        """
        super().showEvent(event)
        self.kv.clear_particles() # Clear particles when the window becomes visible
