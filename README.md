# ChronoMIDI

## Overview
ChronoMIDI is a Python-based desktop application built with PyQt5 for real-time MIDI playback, audio visualization (oscilloscope, equalizer), and MIDI file analysis. It allows users to load MIDI files, apply SoundFonts (.sf2), and export rendered audio to MP3.

## Features
* **MIDI Playback:** Load and play MIDI files with adjustable tempo and SoundFont support.
* **Real-time Visualization:** Dynamic equalizer and oscilloscope displays react to the audio output.
* **MIDI Event Table:** View detailed information about MIDI events (notes, control changes, etc.) in a tabular format.
* **SoundFont Support:** Load custom SoundFont (.sf2) files to change the instrument sounds.
* **MP3 Export:** Render MIDI playback to an MP3 audio file.

## Prerequisites

Before running ChronoMIDI, you'll need the following:

### 1. Python

* **Python 3.8 or newer** is recommended.
    You can download Python from [python.org](https://www.python.org/downloads/).

### 2. External Command-Line Tools

ChronoMIDI relies on two external executables for its core functionality:

* **FluidSynth:** A real-time software synthesizer that powers the MIDI playback.
    * **Windows:** Download the installer from [FluidSynth's official website](http://www.fluidsynth.org/download/). Ensure `fluidsynth.exe` is added to your system's PATH environment variable during installation, or manually.
    * **macOS (Homebrew):** `brew install fluidsynth`
    * **Linux (apt):** `sudo apt-get install fluidsynth`
* **FFmpeg:** A powerful multimedia framework used for MP3 export.
    * **Windows:** Download a build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html). You will need to manually add the `bin` directory (containing `ffmpeg.exe`) to your system's PATH.
    * **macOS (Homebrew):** `brew install ffmpeg`
    * **Linux (apt):** `sudo apt-get install ffmpeg`

## Setup and Running

Follow these steps to set up the project and run ChronoMIDI from its Python source:

1.  **Clone the Repository (or download the source code):**
    If you're using Git, clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/ChronoMIDI.git](https://github.com/mbwood33/ChronoMIDI)
    cd ChronoMIDI
    ```
    If you're downloading, extract the ZIP file and navigate into the main project folder.

2.  **Create and Activate a Virtual Environment (Highly Recommended):**
    Using a virtual environment isolates your project's Python dependencies from your system's global Python installation.
    ```bash
    python -m venv venv
    ```
    * **Activate the environment:**
        * **Windows:**
            ```bash
            .\venv\Scripts\activate
            ```
        * **macOS / Linux:**
            ```bash
            source venv/bin/activate
            ```
    (You should see `(venv)` preceding your command prompt, indicating the environment is active.)

3.  **Install Python Dependencies:**
    With your virtual environment activated, install all required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` file yet, you can generate one by activating your virtual environment, installing your project's dependencies manually (e.g., `pip install PyQt5 mido numpy sounddevice pyfluidsynth`), and then running `pip freeze > requirements.txt`)*

4.  **Verify Asset Placement:**
    Ensure that the `fonts` folder (containing `PixelCode.ttf`) and your application icon (`chronomidi_icon.png` or `chronomidi_icon.ico`) are located in the project's root directory, alongside `chronomidi_gui.py`.

    Your project structure should resemble this:
    ```
    ChronoMIDI/
    ├── chronomidi_gui.py
    ├── chronomidi_icon.png (or .ico)
    ├── fonts/
    │   └── PixelCode.ttf
    ├── requirements.txt
    └── venv/
    ```

5.  **Run the Application:**
    With your virtual environment active, execute the main script:
    ```bash
    python chronomidi_gui.py
    ```
    The ChronoMIDI application window should now appear.

## Troubleshooting

* **"ModuleNotFoundError":** Ensure your virtual environment is active and you've run `pip install -r requirements.txt`.
* **"fluidsynth" or "ffmpeg" not found:** Verify that these tools are installed on your system and their executables are correctly added to your system's PATH environment variable.
* **Custom Font/Icon not showing:** Double-check that the `fonts` folder and icon file are in the correct relative locations as described in "Verify Asset Placement."

# Soundfonts
A great website to find several great soundfonts for use with the application can be found at [https://musical-artifacts.com/](https://musical-artifacts.com/)