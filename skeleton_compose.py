"""
[Beat Name] — Compose
Key: [KEY] | BPM: [BPM] | [NBARS] bars (~M:SS)

Research: [YouTube tutorial IDs or links]
  - [Key finding 1]
  - [Key finding 2]
  - [Key finding 3]

Chord Progression (4-bar loop, [KEY] — [NUMERAL PATTERN]):
  Bar 0: [CHORD]  ([VOICING])  — [FUNCTION]
  Bar 1: [CHORD]  ([VOICING])  — [FUNCTION]
  Bar 2: [CHORD]  ([VOICING])  — [FUNCTION]
  Bar 3: [CHORD]  ([VOICING])  — [FUNCTION]

[N] Sound Layers:
  1. Drums     — [description]
  2. 808 bass  — [description]
  3. [Layer]   — [description]
  ...

Song Structure:
  Intro     bars  0– 7: [description]
  Hook A    bars  8–23: [description]
  Verse     bars 24–39: [description]
  Bridge    bars 40–47: [description]
  Hook B    bars 48–63: [description]

Kit: [kit name(s)]
"""

import os
import random
import numpy as np
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)

# ============================================================================
# CONFIG — edit these for each beat
# ============================================================================

BEAT_NAME  = 'MyBeat'                # used in filenames
OUTPUT_DIR = '/Users/ronantakizawa/Documents/MyBeat_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 140
BPB = 4   # beats per bar

# Section boundaries (0-indexed bars). Standard 64-bar trap layout.
INTRO_S,   INTRO_E   =  0,  8
HOOKA_S,   HOOKA_E   =  8, 24
VERSE_S,   VERSE_E   = 24, 40
BRIDGE_S,  BRIDGE_E  = 40, 48
HOOKB_S,   HOOKB_E   = 48, 64


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat (0-indexed)."""
    return float(bar * BPB + beat)


def shift_pitch(pitch_str, oct_offset):
    """Shift a pitch string by oct_offset octaves. e.g. shift_pitch('C4', 1) -> 'C5'."""
    if pitch_str is None or oct_offset == 0:
        return pitch_str
    n = note.Note(pitch_str)
    n.octave += oct_offset
    return n.nameWithOctave


# Hook octave pattern: within each 16-bar hook (4 cycles of 4 bars),
# first 8 bars at base octave, last 8 bars +1 octave.
# Both hooks use the same pattern.
HOOK_OCT_PATTERN = [0, 0, 1, 1]


# ============================================================================
# CHORD TABLES — define your progression here
# ============================================================================
# Each entry = one bar's chord voicing as music21 pitch strings.
# Choose inversions so adjacent chords share common tones.
CHORDS = [
    ['C3',  'E-3', 'G3' ],   # bar 0
    ['A-2', 'C3',  'E-3'],   # bar 1
    ['E-3', 'G3',  'B-3'],   # bar 2
    ['B-2', 'D3',  'F3' ],   # bar 3
]

# Bass roots for 808 — one per bar of the chord cycle
BASS_ROOTS = ['C2', 'A-1', 'E-2', 'B-1']


# ============================================================================
# MELODY / PATTERN TABLES
# ============================================================================
# Define melody patterns as lists of (pitch_or_None, duration_in_beats).
# One list per bar in a 4-bar cycle. None = rest.
#
# TIMBRE RULE TEMPLATE:
# [INSTRUMENT] TIMBRE RULE: [oscillator type], [envelope character],
# vel [range]. Never louder than [reference]. Do NOT [constraint].
#
# Example (pluck):
# PLUCK_HOOK = [
#     [('C4', 0.5), ('E-4', 0.5), ('G4', 0.5), ('E-4', 0.5),
#      ('C4', 0.5), ('G4', 0.5), ('E-4', 0.5), ('C4', 0.5)],
#     [('A-3', 0.5), ('C4', 0.5), ...],
#     ...
# ]
#
# Example (arp pattern — index into chord tones):
# PIANO_ARP_HOOK = [
#     (chord_tone_index, octave_shift, duration),
#     (0, 0, 0.75), (2, 0, 0.75), (1, 0, 0.75), (0, 1, 0.75), (2, 0, 1.0),
# ]
#
# Example (sparse hits):
# BELL_HITS = [
#     (bar_offset, beat, pitch),
#     (0, 0.0, 'G4'), (1, 2.0, 'E-4'), (2, 0.0, 'B-4'), (3, 3.0, 'F4'),
# ]


# ============================================================================
# DRUMS
# ============================================================================
# GM drum mapping: 36=kick 37=rim 38=snare 39=clap
#                  42=closed-HH 46=open-HH 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def drum_bar(bar, fast_hats=True, half_time=False, crash=False):
        """One bar of drums. Customize per beat style."""
        o = bb(bar)

        # Kick pattern
        hit(o + 0.0, 36, 100)
        # Add syncopated kicks as needed:
        # if bar % 2 == 0: hit(o + 2.5, 36, 65)

        if half_time:
            hit(o + 3.0, 39, 88)
        else:
            # Backbeat — choose snare (38), clap (39), or layer both
            hit(o + 1.0, 39, 90)
            hit(o + 3.0, 39, 94)

        # Hi-hats
        if fast_hats:
            # Customize: 1/32 continuous, bounce+triplets, etc.
            for i in range(8):
                vel = 50 if i % 2 == 0 else 35
                hit(o + i * 0.5, 42, vel)
        else:
            for i in range(4):
                hit(o + i * 1.0, 42, 32)

        # Optional per-bar elements
        # if bar % 2 == 1: hit(o + 2.0, 46, 55)     # open hat
        # if bar % 2 == 0: hit(o + 3.5, 37, 48)     # rim
        if crash: hit(o + 0.0, 49, 85)

    # --- Arrange drums per section ---

    # Intro: customize (e.g., no drums, or crash-only)
    # for bar in range(INTRO_S, INTRO_E): ...

    # Hook A
    for bar in range(HOOKA_S, HOOKA_E):
        idx = bar - HOOKA_S
        drum_bar(bar, fast_hats=True, crash=(idx % 8 == 0))

    # Verse
    for bar in range(VERSE_S, VERSE_E):
        idx = bar - VERSE_S
        drum_bar(bar, fast_hats=False, crash=(idx % 8 == 0))

    # Bridge
    for bar in range(BRIDGE_S, BRIDGE_E):
        idx = bar - BRIDGE_S
        drum_bar(bar, fast_hats=False, half_time=True, crash=(idx == 0))

    # Hook B
    for bar in range(HOOKB_S, HOOKB_E):
        idx = bar - HOOKB_S
        drum_bar(bar, fast_hats=True, crash=(idx % 8 == 0))

    return part


# ============================================================================
# 808 BASS
# ============================================================================

def create_808():
    """808 bass following chord roots. Customize pattern per style."""
    part = stream.Part()
    part.partName = '808 Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def bass_bar(bar, vel=90, ghost=True):
        o    = bb(bar)
        root = BASS_ROOTS[bar % 4]
        # Main hit
        n1 = note.Note(root, quarterLength=2.0)
        n1.volume.velocity = vel
        part.insert(o, n1)
        # Ghost/bounce hit (customize position and length)
        if ghost:
            n2 = note.Note(root, quarterLength=0.5)
            n2.volume.velocity = int(vel * 0.30)
            part.insert(o + 3.0, n2)

    # Arrange per section (vary vel, ghost, tail)
    for bar in range(HOOKA_S, HOOKA_E):
        bass_bar(bar, vel=94, ghost=True)
    for bar in range(VERSE_S, VERSE_E):
        bass_bar(bar, vel=86, ghost=(bar % 2 == 1))
    for bar in range(BRIDGE_S, BRIDGE_E):
        bass_bar(bar, vel=78, ghost=False)
    for bar in range(HOOKB_S, HOOKB_E):
        bass_bar(bar, vel=96, ghost=True)

    return part


# ============================================================================
# SYNTH LAYER TEMPLATE — copy for each melodic layer
# ============================================================================
# Patterns:
#   A) Phrase-based (melody/lead/stab): list of (pitch, dur) per bar
#   B) Arp-based (piano/keys): list of (chord_tone_idx, oct_shift, dur)
#   C) Sparse hits (bell/perc): list of (bar_offset, beat, pitch)

def create_melody_layer(name, phrases, sections, vel_range=(38, 50),
                        hook_keys=('hook_a', 'hook_b')):
    """Generic phrase-based melody layer.

    Args:
        name:      part name for MIDI track
        phrases:   dict mapping section key to 4-bar phrase pattern
                   each phrase = list of 4 lists of (pitch_or_None, dur)
        sections:  dict mapping section key to (start_bar, end_bar)
        vel_range: (min_vel, max_vel) tuple
        hook_keys: section keys that are hooks (get HOOK_OCT_PATTERN applied)
    """
    part = stream.Part()
    part.partName = name
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(phrase, start_bar, vel, oct_offset=0):
        for bar_off, motif in enumerate(phrase):
            o    = bb(start_bar + bar_off)
            beat = 0.0
            for pitch, dur in motif:
                if pitch is None:
                    part.insert(o + beat, note.Rest(quarterLength=dur))
                else:
                    p = shift_pitch(pitch, oct_offset)
                    nd = note.Note(p, quarterLength=dur)
                    nd.volume.velocity = int(np.clip(
                        vel + random.randint(-3, 3), 1, vel_range[1]))
                    part.insert(o + beat, nd)
                beat += dur

    for section_key, (s_bar, e_bar) in sections.items():
        if section_key not in phrases:
            continue
        phrase = phrases[section_key]
        is_hook = section_key in hook_keys
        for i, cycle in enumerate(range(s_bar, e_bar, len(phrase))):
            oct = HOOK_OCT_PATTERN[i] if is_hook and i < len(HOOK_OCT_PATTERN) else 0
            write_phrase(phrase, cycle,
                         vel=random.randint(vel_range[0], vel_range[1]),
                         oct_offset=oct)

    return part


def create_arp_layer(name, patterns, sections, vel_range=(38, 46),
                     hook_keys=('hook_a', 'hook_b')):
    """Generic arp-based layer (piano, keys).

    Args:
        name:     part name for MIDI track
        patterns: dict mapping section key to arp pattern
                  each pattern = list of (chord_tone_idx, oct_shift, dur)
        sections: dict mapping section key to (start_bar, end_bar)
        hook_keys: section keys that are hooks (get HOOK_OCT_PATTERN applied)
    """
    part = stream.Part()
    part.partName = name
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def arp_bar(bar, pattern, vel, oct_offset=0):
        o = bb(bar)
        chord_tones = CHORDS[bar % 4]
        beat = 0.0
        for tone_idx, oct_shift, dur in pattern:
            idx = min(tone_idx, len(chord_tones) - 1)
            pitch_str = chord_tones[idx]
            total_shift = oct_shift + oct_offset
            if total_shift != 0:
                n_obj = note.Note(pitch_str)
                n_obj.octave += total_shift
                pitch_str = n_obj.nameWithOctave
            nd = note.Note(pitch_str, quarterLength=dur)
            nd.volume.velocity = int(np.clip(
                vel + random.randint(-3, 3), 1, vel_range[1]))
            part.insert(o + beat, nd)
            beat += dur

    for section_key, (s_bar, e_bar) in sections.items():
        if section_key not in patterns:
            continue
        pattern = patterns[section_key]
        is_hook = section_key in hook_keys
        for bar in range(s_bar, e_bar):
            cycle_idx = (bar - s_bar) // 4
            oct = HOOK_OCT_PATTERN[cycle_idx] if is_hook and cycle_idx < len(HOOK_OCT_PATTERN) else 0
            arp_bar(bar, pattern, vel=random.randint(
                vel_range[0], vel_range[1]), oct_offset=oct)

    return part


def create_sparse_layer(name, hits, sections, vel_range=(36, 44),
                        hook_keys=('hook_a', 'hook_b')):
    """Generic sparse accent layer (bell, perc).

    Args:
        name:     part name for MIDI track
        hits:     list of (bar_offset, beat, pitch) within a 4-bar cycle
        sections: dict mapping section key to (start_bar, end_bar)
        hook_keys: section keys that are hooks (get HOOK_OCT_PATTERN applied)
    """
    part = stream.Part()
    part.partName = name
    part.insert(0, tempo.MetronomeMark(number=BPM))

    for section_key, (s_bar, e_bar) in sections.items():
        is_hook = section_key in hook_keys
        for i, cycle_start in enumerate(range(s_bar, e_bar, 4)):
            oct = HOOK_OCT_PATTERN[i] if is_hook and i < len(HOOK_OCT_PATTERN) else 0
            for bar_off, beat, pitch in hits:
                p = shift_pitch(pitch, oct)
                nd = note.Note(p, quarterLength=1.0)
                nd.volume.velocity = int(np.clip(
                    random.randint(vel_range[0], vel_range[1]),
                    1, vel_range[1]))
                part.insert(bb(cycle_start + bar_off, beat), nd)

    return part


# ============================================================================
# PAD (sustained chords) — common across beats
# ============================================================================

def create_pad(sections, vel_default=50):
    """Sustained pad chords from CHORDS table."""
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    for section_key, (s_bar, e_bar, vel) in sections.items():
        for bar in range(s_bar, e_bar):
            c = chord.Chord(CHORDS[bar % 4], quarterLength=4.0)
            c.volume.velocity = vel
            part.insert(bb(bar), c)

    return part


# ============================================================================
# MIDI HELPERS — reuse as-is
# ============================================================================

def insert_program(track, program):
    pos = 0
    for j, msg in enumerate(track):
        if msg.type == 'track_name':
            pos = j + 1
            break
    track.insert(pos, Message('program_change', program=program, time=0))


# Map part name keywords to GM program numbers.
# Extend this dict for new instrument types.
GM_PROGRAMS = {
    '808':   38,   # Synth Bass 1
    'bass':  38,
    'pad':   89,   # Pad 2 (warm)
    'pluck': 80,   # Lead 1 (square)
    'brass': 62,   # Synth Brass 1
    'piano': 0,    # Acoustic Grand Piano
    'bell':  14,   # Tubular Bells
    'lead':  14,
    'flute': 73,   # Flute
    'string':48,   # String Ensemble 1
}


def fix_instruments(mid, part_names):
    for i, track in enumerate(mid.tracks):
        if i == 0:
            continue
        pidx = i - 1
        if pidx >= len(part_names):
            break
        name = part_names[pidx].lower()
        if 'drum' in name:
            for msg in track:
                if hasattr(msg, 'channel'):
                    msg.channel = 9
        else:
            for keyword, program in GM_PROGRAMS.items():
                if keyword in name:
                    insert_program(track, program)
                    break


def save(score, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    score.write('midi', fp=path)
    mid  = MidiFile(path)
    names = [p.partName or '' for p in score.parts]
    fix_instruments(mid, names)
    mid.save(path)
    print(f'  {filename}')
    return path


def solo(part):
    s = stream.Score()
    s.append(part)
    return s


# ============================================================================
# COMPOSE & SAVE — customize part list
# ============================================================================

if __name__ == '__main__':
    print(f'Composing {BEAT_NAME} ...')
    print(f'  Key: C minor  |  BPM: {BPM}  |  64 bars')

    # Create parts (customize this list)
    parts = {
        'drums': create_drums(),
        '808':   create_808(),
        # 'pad':   create_pad({...}),
        # 'pluck': create_melody_layer('Pluck', ...),
        # 'piano': create_arp_layer('Piano', ...),
        # 'bell':  create_sparse_layer('Bell', ...),
    }

    # Save individual stems
    print('Saving individual stems ...')
    for stem_name, part in parts.items():
        save(solo(part), f'{BEAT_NAME}_{stem_name}.mid')

    # Save full arrangement
    print('\nSaving full arrangement ...')
    full = stream.Score()
    for part in parts.values():
        full.append(part)
    save(full, f'{BEAT_NAME}_FULL.mid')

    print(f'\nDone! MIDI files saved to:\n  {OUTPUT_DIR}')
