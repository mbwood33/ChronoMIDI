# utils.py
# Utility functions and constants for the chronomidi project.

# List of standard MIDI note names for display purposes
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_note_to_name(n: int) -> str:
    """
    Converts a MIDI note number (0-127) to its musical name (e.g., C4, A#3).

    Args:
        n (int): The MIDI note number.

    Returns:
        str: The musical name of the note.
    """
    # Calculate note name (C, C#, D, etc.) using modulo 12
    # Calculate octave number: MIDI note 0 is C-1, so (n // 12) - 1
    return NOTE_NAMES[n % 12] + str((n // 12) - 1)

# Dictionary mapping MIDI Control Change (CC) numbers to human-readable names
CONTROL_CHANGE_NAMES = {
    0: 'Bank Select', 1: 'Modulation Wheel', 2: 'Breath Controller',
    4: 'Foot Controller', 5: 'Portamento Time', 6: 'Data Entry MSB',
    7: 'Channel Volume', 8: 'Balance', 10: 'Pan',
    11: 'Expression Controller', 12: 'Effect Control 1', 13: 'Effect Control 2',
    64: 'Sustain Pedal', 65: 'Portamento On/Off', 66: 'Sostenuto Pedal',
    67: 'Soft Pedal', 68: 'Legato Footswitch', 69: 'Hold 2',
    70: 'Sound Controller 1', 71: 'Sound Controller 2',
    72: 'Sound Controller 3', 73: 'Sound Controller 4',
    74: 'Sound Controller 5', 75: 'Sound Controller 6',
    76: 'Sound Controller 7', 77: 'Sound Controller 8',
    78: 'Sound Controller 9', 79: 'Sound Controller 10',
    80: 'General Purpose Controller 1', 81: 'General Purpose Controller 2',
    82: 'General Purpose Controller 3', 83: 'General Purpose Controller 4',
    84: 'Portamento Control',
    91: 'Effects 1 Depth (Reverb)', 92: 'Effects 2 Depth (Tremolo)',
    93: 'Effects 3 Depth (Chorus)', 94: 'Effects 4 Depth (Celeste)',
    95: 'Effects 5 Depth (Phaser)',
    96: 'Data Increment', 97: 'Data Decrement', 98: 'NRPN LSB',
    99: 'NRPN MSB', 100: 'RPN LSB', 101: 'RPN MSB',
    121: 'Reset All Controllers', 122: 'Local Control On/Off',
    123: 'All Notes Off', 124: 'Omni Mode Off', 125: 'Omni Mode On',
    126: 'Mono Mode On', 127: 'Poly Mode On',
}

def get_time_signature_at_tick(absolute_tick: int, time_signature_changes: list) -> tuple:
    """
    Determines the active time signature at a given absolute MIDI tick.

    Args:
        absolute_tick (int): The absolute tick position in the MIDI file.
        time_signature_changes (list): A sorted list of (tick, numerator, denominator, cumulative_measures) tuples.

    Returns:
        tuple: A (numerator, denominator) tuple representing the active time signature.
    """
    active_ts = (4, 4) # Default time signature
    for ts_tick, num, den, _ in time_signature_changes:
        if absolute_tick >= ts_tick:
            active_ts = (num, den)
        else:
            break # List is sorted, so we've passed the relevant changes
    return active_ts

def calculate_beat_measure(absolute_tick: int, ticks_per_beat: int, time_signature_changes: list) -> tuple:
    """
    Calculates the cumulative measure and beat number for a given absolute tick,
    using pre-calculated cumulative measures from time_signature_changes.

    Args:
        absolute_tick (int): The absolute tick position.
        ticks_per_beat (int): The MIDI file's ticks per beat (ticks per quarter note).
        time_signature_changes (list): A sorted list of (tick, numerator, denominator, cumulative_measures_at_start_of_this_ts_block) tuples.
            This list is expected to be sorted by tick and include a (0, 4, 4, 0) entry if no
            time signature is explicitly set at the beginning.

    Returns:
        tuple: A (measure, beat_in_measure) tuple, where measure is the cumulative measure
               from the beginning of the MIDI file.
    """
    # Find the most recent time signature change that applies to absolute_tick
    active_ts_info = time_signature_changes[0] # Default to the first entry (tick 0)
    for ts_info in time_signature_changes:
        ts_change_tick = ts_info[0]
        if absolute_tick >= ts_change_tick:
            active_ts_info = ts_info
        else:
            break # List is sorted, so we've passed the relevant changes

    active_ts_tick = active_ts_info[0]
    active_ts_numerator = active_ts_info[1]
    active_ts_denominator = active_ts_info[2]
    cumulative_measures_at_start_of_block = active_ts_info[3] # This is the key: full measures before this block

    # Calculate ticks within the current time signature block
    ticks_in_current_block = absolute_tick - active_ts_tick

    # Convert ticks in current block to quarter notes
    quarter_notes_in_current_block = ticks_in_current_block / ticks_per_beat

    # Convert quarter notes to the actual beat unit of the current time signature
    # The '4' in (active_ts_denominator / 4.0) represents a quarter note as the base unit.
    # For example, if active_ts_denominator is 8 (for 7/8), conversion_factor is 2.0.
    # This means 1 quarter note = 2 eighth notes.
    conversion_factor = active_ts_denominator / 4.0
    beats_in_current_block_in_target_unit = quarter_notes_in_current_block * conversion_factor

    # Calculate measure and beat within this current time signature block
    beats_per_measure_in_target_unit = active_ts_numerator

    if beats_per_measure_in_target_unit == 0: # Avoid division by zero
        measure_offset_in_block = 0
        beat_in_measure = 0.0
    else:
        measure_offset_in_block = int(beats_in_current_block_in_target_unit // beats_per_measure_in_target_unit)
        beat_in_measure = beats_in_current_block_in_target_unit % beats_per_measure_in_target_unit

    # Total cumulative measures = measures accumulated before this block + measures within this block
    # We add 1 for 1-indexing of the measure number.
    total_measure = cumulative_measures_at_start_of_block + measure_offset_in_block + 1

    return total_measure, beat_in_measure
