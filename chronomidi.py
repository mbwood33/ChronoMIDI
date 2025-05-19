#!/usr/bin/env python3
"""
chronomidi.py

High-precision SF2-based MIDI sequencer prototype with async logging:
  • Loads a MIDI file
  • Builds a queue of non-meta events with timestamps
  • Plays back through FluidSynth + your SF2
  • Optionally displays events in real time (--verbose), with aligned, colorized columns
  • Improved timing: absolute scheduling via time.monotonic()
  • Asynchronous logging: spins up a logger thread to avoid blocking audio
  • Event-type–specific coloring and note-name display for clearer logs
"""

import argparse
import sys
import time
import threading
import queue
from collections import namedtuple

import mido
from fluidsynth import Synth
from colorama import init as colorama_init, Fore, Style

# Initialize colorama for Windows/ANSI support
colorama_init(autoreset=True)

# Event: holds absolute timestamp and MIDI message
Event = namedtuple("Event", ["time", "message"])

# Note names for MIDI: C0 = note 0 (octave -1 in scientific pitch), but we map C-1..C9
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_note_to_name(note):
    """
    Convert MIDI note number (0-127) to note name, e.g. 60 -> C4
    Scientific pitch: octave = (note // 12) - 1
    """
    name = NOTE_NAMES[note % 12]
    octave = (note // 12) - 1
    return f"{name}{octave}"

class Sequencer:
    """
    Loads a MIDI file into a list of (timestamp, MidiMessage) events.
    Provides play(synth, verbose=False) with precise, absolute scheduling and async logging.
    """

    def __init__(self, midi_path):
        mid = mido.MidiFile(midi_path)
        self.events = []
        abs_time = 0.0
        for msg in mid:
            abs_time += msg.time
            if not msg.is_meta:
                self.events.append(Event(abs_time, msg))
        self.events.sort(key=lambda e: e.time)

    def _logger(self, log_queue):
        """Logger thread: prints lines from queue until None sentinel."""
        while True:
            line = log_queue.get()
            if line is None:
                break
            print(line)
            log_queue.task_done()

    def play(self, synth, verbose=False):
        """
        Play events using absolute timing. If verbose, log asynchronously.
        """
        # Color mapping for event types
        type_colors = {
            "note_on": Fore.CYAN,
            "note_off": Fore.BLUE,
            "control_change": Fore.YELLOW,
            "program_change": Fore.GREEN,
            "pitchwheel": Fore.RED,
        }

        log_queue = None
        if verbose:
            log_queue = queue.Queue()
            t = threading.Thread(target=self._logger, args=(log_queue,), daemon=True)
            t.start()

        start_time = time.monotonic()
        for evt in self.events:
            target = start_time + evt.time
            now = time.monotonic()
            wait = target - now
            if wait > 0:
                time.sleep(wait)

            msg = evt.message
            if verbose:
                # Timestamp string
                tstr = f"{evt.time:7.3f}s"
                # Parameter string with note-name mapping
                if msg.type in ("note_on", "note_off"):
                    note_name = midi_note_to_name(msg.note)
                    param = (f"ch={msg.channel:2} note={msg.note:3}({note_name}) "
                             f"vel={msg.velocity:3}")
                elif msg.type == "control_change":
                    param = f"ch={msg.channel:2} ctrl={msg.control:3} val={msg.value:3}"
                elif msg.type == "program_change":
                    param = f"ch={msg.channel:2} prog={msg.program:3}"
                elif msg.type == "pitchwheel":
                    param = f"ch={msg.channel:2} pitch={msg.pitch:6}"
                else:
                    param = str(msg)
                # Choose color by event type
                type_col = type_colors.get(msg.type, Fore.MAGENTA)
                # Build colored, aligned log line
                line = (
                    f"{Fore.GREEN}{tstr}{Style.RESET_ALL} "
                    f"{type_col}{msg.type:14}{Style.RESET_ALL} "
                    f"{param}"
                )
                log_queue.put(line)

            # Dispatch to FluidSynth immediately
            if msg.type == "note_on":
                synth.noteon(msg.channel, msg.note, msg.velocity)
            elif msg.type == "note_off":
                synth.noteoff(msg.channel, msg.note)
            elif msg.type == "control_change":
                synth.cc(msg.channel, msg.control, msg.value)
            elif msg.type == "program_change":
                synth.program_change(msg.channel, msg.program)
            elif msg.type == "pitchwheel":
                synth.pitch_bend(msg.channel, msg.pitch)

        # Shutdown logger thread
        if verbose:
            log_queue.put(None)
            t.join()


def parse_args():
    p = argparse.ArgumentParser(
        prog="chronomidi",
        description="ChronoMIDI – High-precision SF2 MIDI sequencer with live event display"
    )
    p.add_argument("midi_file", help=".mid file path")
    p.add_argument("sf2_file", help=".sf2 soundfont path")
    p.add_argument(
        "--driver", default=None,
        help="Audio driver: alsa, dsound, coreaudio... (default auto)"
    )
    p.add_argument(
        "--channel", type=int, default=0,
        help="MIDI channel (0–15), default=0"
    )
    p.add_argument(
        "--bank", type=int, default=0,
        help="SoundFont bank#, default=0"
    )
    p.add_argument(
        "--preset", type=int, default=0,
        help="SoundFont preset#, default=0"
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Colorized, aligned MIDI event logs"
    )
    return p.parse_args()


def main():
    args = parse_args()

    fs = Synth()
    if args.driver:
        fs.start(driver=args.driver, midi_driver="none")
    else:
        fs.start(midi_driver="none")

    sfid = fs.sfload(args.sf2_file)
    fs.program_select(args.channel, sfid, args.bank, args.preset)

    seq = Sequencer(args.midi_file)
    print(f"Loaded {len(seq.events)} events from '{args.midi_file}'")
    seq.play(fs, verbose=args.verbose)

    # let tails ring
    time.sleep(1.0)
    fs.delete()

if __name__ == "__main__":
    main()
