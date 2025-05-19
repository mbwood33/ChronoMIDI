# ChronoMIDI

A high-precision MIDI sequencer in Python that:
- Plays .mid files through SoundFont (.sf2) via FluidSynth  
- Displays live, colorized, aligned MIDI event logs  
- Supports custom audio drivers, channels, banks, and presets  

## Quickstart

```bash
git clone https://github.com/yourusername/ChronoMIDI.git
cd ChronoMIDI
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate.bat on Windows
pip install mido pyFluidSynth python-rtmidi colorama
./chronomidi.py examples/song.mid examples/FluidR3_GM.sf2 --verbose