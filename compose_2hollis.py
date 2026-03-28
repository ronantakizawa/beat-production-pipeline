"""
2Hollis Type Beat -- MIDI Composition
Hyperpop/rage with stuttered plucks, four-on-the-floor kicks, beat switches.

Layers:
  1. Pluck Lead   -- stuttered 3-note pattern, 16th notes, C4-G4
  2. Pluck Layer2 -- same rhythm octave up (C5-G5) for shimmer
  3. Drums        -- four-on-the-floor kick, fat clap 2&4, 16th hats
  4. 808 Bass     -- chord roots, tight decay, beat 1 (+sometimes beat 3)

Chord Progressions (i-VI-III-VII -- dark, driving):
  Key Cm:  Cm -> Ab -> Eb -> Bb
  Key F#m: F#m -> D -> A -> E

Song Structure (64 bars):
  Intro     0- 3: pluck only, building
  Drop1     4-19: full drums + bass + plucks (16 bars)
  Break    20-23: ambient cutout (no drums, 4 bars)
  Drop2    24-39: full arrangement, pluck variation
  Ambient  40-43: another cutout (4 bars)
  Drop3    44-59: final drop, most energy
  Outro    60-63: fade

Usage:
  python compose_2hollis.py --name "2Hollis_Dark" --key Cm --bpm 150
  python compose_2hollis.py --name "2Hollis_Glitch2" --key F#m --bpm 155
"""

import argparse
import os
import random
import numpy as np
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = '/Users/ronantakizawa/Documents/2Hollis_Beat'

BPB = 4  # beats per bar

INTRO_S,  INTRO_E  =  0,  4
DROP1_S,  DROP1_E  =  4, 20
BREAK_S,  BREAK_E  = 20, 24
DROP2_S,  DROP2_E  = 24, 40
AMB_S,    AMB_E    = 40, 44
DROP3_S,  DROP3_E  = 44, 60
OUTRO_S,  OUTRO_E  = 60, 64


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat."""
    return float(bar * BPB + beat)


# ============================================================================
# CHORD PROGRESSIONS -- i-VI-III-VII (dark, driving)
# ============================================================================

# Key Cm: Cm -> Ab -> Eb -> Bb
CHORDS_CM = {
    'pluck_tones': [
        ['C4', 'Eb4', 'G4'],    # Cm
        ['Ab3', 'C4', 'Eb4'],   # Ab
        ['Eb4', 'G4', 'Bb4'],   # Eb
        ['Bb3', 'D4', 'F4'],    # Bb
    ],
    'pluck_high': [
        ['C5', 'Eb5', 'G5'],    # Cm oct up
        ['Ab4', 'C5', 'Eb5'],   # Ab oct up
        ['Eb5', 'G5', 'Bb5'],   # Eb oct up
        ['Bb4', 'D5', 'F5'],    # Bb oct up
    ],
    'bass_roots': ['C2', 'Ab1', 'Eb2', 'Bb1'],
}

# Key F#m: F#m -> D -> A -> E
CHORDS_FSM = {
    'pluck_tones': [
        ['F#4', 'A4', 'C#5'],   # F#m
        ['D4', 'F#4', 'A4'],    # D
        ['A4', 'C#5', 'E5'],    # A
        ['E4', 'G#4', 'B4'],    # E
    ],
    'pluck_high': [
        ['F#5', 'A5', 'C#6'],   # F#m oct up
        ['D5', 'F#5', 'A5'],    # D oct up
        ['A5', 'C#6', 'E6'],    # A oct up
        ['E5', 'G#5', 'B5'],    # E oct up
    ],
    'bass_roots': ['F#2', 'D2', 'A1', 'E2'],
}

PROGRESSION = {'Cm': CHORDS_CM, 'F#m': CHORDS_FSM}

# Stuttered gate pattern: hit-hit-rest-hit-rest-hit-hit-rest (per bar, 16th grid)
# 1=hit, 0=rest
STUTTER_PATTERN = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
# Variation for drop2/drop3
STUTTER_VAR     = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]


# ============================================================================
# PLUCK LEAD -- stuttered 3-note pattern, 16th notes
# ============================================================================

def create_pluck_lead(BPM, chords):
    part = stream.Part()
    part.partName = 'African Marimba'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    tones = chords['pluck_tones']

    def pluck_bar(bar, pattern, vel_base=60):
        o = bb(bar)
        ct = tones[bar % 4]
        for i in range(16):
            if not pattern[i]:
                continue
            pitch = ct[i % len(ct)]
            beat_pos = i * 0.25
            vel = vel_base + random.randint(-4, 4)
            if i % 4 == 0:
                vel = min(127, vel + 6)
            n = note.Note(pitch, quarterLength=0.25)
            n.volume.velocity = vel
            part.insert(o + beat_pos, n)

    # Intro: pluck only, building volume
    for bar in range(INTRO_S, INTRO_E):
        vel = 35 + (bar - INTRO_S) * 7
        pluck_bar(bar, STUTTER_PATTERN, vel_base=vel)

    # Drop1: full stuttered pluck
    for bar in range(DROP1_S, DROP1_E):
        pluck_bar(bar, STUTTER_PATTERN, vel_base=62)

    # Break: sparse pluck (every other hit)
    sparse = [1 if i % 4 == 0 else 0 for i in range(16)]
    for bar in range(BREAK_S, BREAK_E):
        pluck_bar(bar, sparse, vel_base=40)

    # Drop2: variation pattern
    for bar in range(DROP2_S, DROP2_E):
        pluck_bar(bar, STUTTER_VAR, vel_base=64)

    # Ambient: sparse again
    for bar in range(AMB_S, AMB_E):
        pluck_bar(bar, sparse, vel_base=38)

    # Drop3: main pattern, louder
    for bar in range(DROP3_S, DROP3_E):
        pluck_bar(bar, STUTTER_PATTERN, vel_base=68)

    # Outro: fade
    for bar in range(OUTRO_S, OUTRO_E):
        vel = max(25, 60 - (bar - OUTRO_S) * 10)
        pluck_bar(bar, STUTTER_PATTERN, vel_base=vel)

    return part


# ============================================================================
# PLUCK LAYER 2 -- octave up shimmer, same rhythm
# ============================================================================

def create_pluck_high(BPM, chords):
    part = stream.Part()
    part.partName = 'Glockenspiel'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    tones = chords['pluck_high']

    def pluck_bar(bar, pattern, vel_base=48):
        o = bb(bar)
        ct = tones[bar % 4]
        for i in range(16):
            if not pattern[i]:
                continue
            pitch = ct[i % len(ct)]
            beat_pos = i * 0.25
            vel = vel_base + random.randint(-3, 3)
            if i % 4 == 0:
                vel = min(127, vel + 4)
            n = note.Note(pitch, quarterLength=0.25)
            n.volume.velocity = vel
            part.insert(o + beat_pos, n)

    # No pluck_high in intro (staggered entry)

    # Drop1: enters bar 8 (4 bars after drop start)
    for bar in range(DROP1_S + 4, DROP1_E):
        pluck_bar(bar, STUTTER_PATTERN, vel_base=48)

    # Break: silent

    # Drop2: enters bar 28
    for bar in range(DROP2_S + 4, DROP2_E):
        pluck_bar(bar, STUTTER_VAR, vel_base=50)

    # Ambient: silent

    # Drop3: full, louder
    for bar in range(DROP3_S, DROP3_E):
        pluck_bar(bar, STUTTER_PATTERN, vel_base=54)

    return part


# ============================================================================
# DRUMS -- four-on-the-floor kick, fat clap 2&4, 16th hats
# ============================================================================
# GM: 36=kick, 39=clap, 42=closed-HH, 46=open-HH, 49=crash

def create_drums(BPM):
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def hollis_bar(bar, crash=False):
        o = bb(bar)

        # Four-on-the-floor kick
        for beat in range(4):
            hit(o + beat, 36, 105 + random.randint(-3, 3))

        # Fat clap on 2 and 4
        hit(o + 1.0, 39, 98 + random.randint(-3, 3))
        hit(o + 3.0, 39, 100 + random.randint(-3, 3))

        # 16th hi-hats: straight, on-beat louder
        for i in range(16):
            beat_pos = i * 0.25
            if i % 4 == 0:
                vel = 62
            elif i % 2 == 0:
                vel = 46
            else:
                vel = 34
            hit(o + beat_pos, 42, vel + random.randint(-3, 3))

        # Open hat on "and of 4" every other bar
        if bar % 2 == 1:
            hit(o + 3.5, 46, 50)

        if crash:
            hit(o + 0.0, 49, 90)

    # No drums in intro or breaks/ambient

    # Drop1
    for bar in range(DROP1_S, DROP1_E):
        hollis_bar(bar, crash=(bar == DROP1_S))

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        hollis_bar(bar, crash=(bar == DROP2_S))

    # Drop3
    for bar in range(DROP3_S, DROP3_E):
        hollis_bar(bar, crash=(bar == DROP3_S))

    # Outro: drums continue but fade velocity
    for bar in range(OUTRO_S, OUTRO_E):
        o = bb(bar)
        vel_scale = max(0.3, 1.0 - (bar - OUTRO_S) * 0.2)
        for beat in range(4):
            hit(o + beat, 36, int(100 * vel_scale))
        hit(o + 1.0, 39, int(95 * vel_scale))
        hit(o + 3.0, 39, int(95 * vel_scale))
        for i in range(8):
            hit(o + i * 0.5, 42, int(50 * vel_scale))

    return part


# ============================================================================
# 808 BASS -- chord roots, tight decay, beat 1 + sometimes beat 3
# ============================================================================

def create_808(BPM, chords):
    part = stream.Part()
    part.partName = '808 Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    roots = chords['bass_roots']

    def bass_bar(bar, vel=96, beat3=True):
        o = bb(bar)
        root = roots[bar % 4]

        # Beat 1: tight (2 beats = half bar)
        n1 = note.Note(root, quarterLength=2.0)
        n1.volume.velocity = vel
        part.insert(o, n1)

        # Beat 3: short hit (sometimes)
        if beat3:
            n2 = note.Note(root, quarterLength=1.0)
            n2.volume.velocity = int(vel * 0.7)
            part.insert(o + 2.0, n2)

    # No bass in intro or breaks/ambient

    # Drop1
    for bar in range(DROP1_S, DROP1_E):
        bass_bar(bar, vel=100, beat3=(bar % 2 == 0))

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        bass_bar(bar, vel=102, beat3=(bar % 2 == 0))

    # Drop3: most energy, beat3 every bar
    for bar in range(DROP3_S, DROP3_E):
        bass_bar(bar, vel=104, beat3=True)

    # Outro
    for bar in range(OUTRO_S, OUTRO_E):
        vel = max(60, 96 - (bar - OUTRO_S) * 10)
        bass_bar(bar, vel=vel, beat3=False)

    return part


# ============================================================================
# MIDI HELPERS
# ============================================================================

def insert_program(track, program):
    pos = 0
    for j, msg in enumerate(track):
        if msg.type == 'track_name':
            pos = j + 1
            break
    track.insert(pos, Message('program_change', program=program, time=0))


GM_PROGRAMS = {
    'marimba': 12,
    'glock':   9,
    '808':     38,
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


def save(score, filename, output_dir):
    path = os.path.join(output_dir, filename)
    score.write('midi', fp=path)
    mid = MidiFile(path)
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
# COMPOSE & SAVE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2Hollis type beat MIDI composer')
    parser.add_argument('--name', required=True, help='Beat name')
    parser.add_argument('--key', required=True, choices=['Cm', 'F#m'],
                        help='Key (Cm=dark, F#m=dark)')
    parser.add_argument('--bpm', required=True, type=int, help='BPM (130-165)')
    args = parser.parse_args()

    BPM = max(130, min(args.bpm, 165))
    beat_name = args.name
    chords = PROGRESSION[args.key]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Composing {beat_name} ...')
    print(f'  Key: {args.key}  |  BPM: {BPM}  |  64 bars')

    parts = {
        'pluck_lead':  create_pluck_lead(BPM, chords),
        'pluck_high':  create_pluck_high(BPM, chords),
        'drums':       create_drums(BPM),
        '808':         create_808(BPM, chords),
    }

    # Save individual stems
    print('Saving individual stems ...')
    for stem_name, part in parts.items():
        save(solo(part), f'{beat_name}_{stem_name}.mid', OUTPUT_DIR)

    # Save full arrangement
    print('\nSaving full arrangement ...')
    full = stream.Score()
    for part in parts.values():
        full.append(part)
    save(full, f'{beat_name}_FULL.mid', OUTPUT_DIR)

    print(f'\nDone! MIDI files saved to:\n  {OUTPUT_DIR}')
