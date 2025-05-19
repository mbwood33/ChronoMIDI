#!/usr/bin/env python3
import sys
import time
import mido
from fluidsynth import Synth

def play_midi_with_sf2(midi_path, sf2_path, audio_driver="dsound"):
    # --- Start FluidSynth with no MIDI-in driver ---
    fs = Synth()
    fs.start(driver=audio_driver, midi_driver="none")

    # --- Load SoundFont & select program on channel 0 ---
    sfid = fs.sfload(sf2_path)
    fs.program_select(0, sfid, 0, 0)

    # --- Stream & dispatch events ---
    for msg in mido.MidiFile(midi_path).play():
        if msg.is_meta:
            continue

        if msg.type == "note_on":
            fs.noteon(msg.channel, msg.note, msg.velocity)
        elif msg.type == "note_off":
            fs.noteoff(msg.channel, msg.note)
        elif msg.type == "control_change":
            fs.cc(msg.channel, msg.control, msg.value)
        elif msg.type == "program_change":
            fs.program_change(msg.channel, msg.program)
        elif msg.type == "pitchwheel":
            # mido’s .pitch is a signed 14-bit value
            fs.pitch_bend(msg.channel, msg.pitch)
        # …you can add more handlers here (aftertouch, sys-ex, etc.)

    # let reverb/chorus tails ring out
    time.sleep(1.0)
    fs.delete()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: play_midi.py <midi_file> <sf2_file> [audio_driver]")
        sys.exit(1)

    midi_file = sys.argv[1]
    sf2_file  = sys.argv[2]
    driver    = sys.argv[3] if len(sys.argv) > 3 else "dsound"

    play_midi_with_sf2(midi_file, sf2_file, driver)
