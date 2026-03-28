"""
Techno -- MIDI Composition
Dark, minimal, hypnotic. Phrygian mode, acid bass, 16-bar blocks.

Layers:
  1. Acid Bass  -- 16th note repeating Phrygian pattern, 2-bar loop
  2. Pad        -- sustained Phrygian chord, atmospheric
  3. Drums      -- four-on-the-floor kick, clap 2&4, 16th hats

Keys (Phrygian mode):
  C Phrygian: C Db Eb F G Ab Bb
  A Phrygian: A Bb C D E F G

Structure (96 bars, 16-bar blocks):
  Block1  0-15:  pad only
  Block2 16-31:  kick enters
  Block3 32-47:  acid + hats enter
  Block4 48-55:  break (acid + pad, no drums)
  Block5 56-79:  full peak energy
  Block6 80-95:  outro fade

Usage:
  python compose_techno.py --name "Techno_Phrygian" --key C --bpm 132
"""

import argparse
import os
import random
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)

OUTPUT_DIR = '/Users/ronantakizawa/Documents/Techno_Beat'
BPB = 4

BLK1_S, BLK1_E =  0, 16
BLK2_S, BLK2_E = 16, 32
BLK3_S, BLK3_E = 32, 48
BRK_S,  BRK_E  = 48, 56
BLK5_S, BLK5_E = 56, 80
OUT_S,  OUT_E   = 80, 96


def bb(bar, beat=0.0):
    return float(bar * BPB + beat)


# ============================================================================
# PHRYGIAN SCALES & ACID PATTERNS
# ============================================================================

# C Phrygian: C Db Eb F G Ab Bb
SCALE_C = {
    'acid_pattern': ['C2', 'Db2', 'C2', 'F2', 'C2', 'Db2', 'G2', 'C2',
                     'Eb2', 'Db2', 'C2', 'F2', 'Eb2', 'Db2', 'C2', 'C2'],
    'acid_var':     ['C2', 'Db2', 'Eb2', 'Db2', 'C2', 'G1', 'Ab1', 'C2',
                     'F2', 'Eb2', 'Db2', 'C2', 'Db2', 'Eb2', 'F2', 'C2'],
    'pad': ['C3', 'Eb3', 'G3'],
    'bass_root': 'C1',
}

# A Phrygian: A Bb C D E F G
SCALE_A = {
    'acid_pattern': ['A1', 'Bb1', 'A1', 'D2', 'A1', 'Bb1', 'E2', 'A1',
                     'C2', 'Bb1', 'A1', 'D2', 'C2', 'Bb1', 'A1', 'A1'],
    'acid_var':     ['A1', 'Bb1', 'C2', 'Bb1', 'A1', 'E1', 'F1', 'A1',
                     'D2', 'C2', 'Bb1', 'A1', 'Bb1', 'C2', 'D2', 'A1'],
    'pad': ['A2', 'C3', 'E3'],
    'bass_root': 'A0',
}

SCALES = {'C': SCALE_C, 'A': SCALE_A}


# ============================================================================
# ACID BASS -- 16th note repeating Phrygian sequence
# ============================================================================

def create_acid(BPM, scale):
    part = stream.Part()
    part.partName = 'Acid Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    pat = scale['acid_pattern']
    var = scale['acid_var']

    def acid_bar(bar, pattern, vel_base=70):
        o = bb(bar)
        for i in range(16):
            pitch = pattern[i % len(pattern)]
            beat_pos = i * 0.25
            # Accent pattern: strong on downbeats, ghost on offbeats
            if i % 4 == 0:
                vel = vel_base + 10
            elif i % 2 == 0:
                vel = vel_base
            else:
                vel = vel_base - 12
            vel += random.randint(-3, 3)
            n = note.Note(pitch, quarterLength=0.25)
            n.volume.velocity = min(127, max(1, vel))
            part.insert(o + beat_pos, n)

    # Block3: acid enters
    for bar in range(BLK3_S, BLK3_E):
        acid_bar(bar, pat, vel_base=65)

    # Break: acid continues, variation pattern
    for bar in range(BRK_S, BRK_E):
        acid_bar(bar, var, vel_base=55)

    # Block5: full, main pattern
    for bar in range(BLK5_S, BLK5_E):
        p = pat if (bar - BLK5_S) % 8 < 6 else var  # variation every 8 bars
        acid_bar(bar, p, vel_base=72)

    # Outro: fade
    for bar in range(OUT_S, OUT_S + 8):
        vel = max(30, 65 - (bar - OUT_S) * 5)
        acid_bar(bar, pat, vel_base=vel)

    return part


# ============================================================================
# PAD -- sustained Phrygian chord
# ============================================================================

def create_pad(BPM, scale):
    part = stream.Part()
    part.partName = 'String Ensemble'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    pad_notes = scale['pad']

    def pad_bar(bar, vel=28):
        o = bb(bar)
        ch = chord.Chord(pad_notes, quarterLength=4.0)
        ch.volume.velocity = vel
        part.insert(o, ch)

    # Block1: pad only, building
    for bar in range(BLK1_S, BLK1_E):
        pad_bar(bar, vel=20 + (bar - BLK1_S))

    # Block2: pad continues
    for bar in range(BLK2_S, BLK2_E):
        pad_bar(bar, vel=30)

    # Block3: quieter behind acid
    for bar in range(BLK3_S, BLK3_E):
        pad_bar(bar, vel=22)

    # Break: pad prominent
    for bar in range(BRK_S, BRK_E):
        pad_bar(bar, vel=32)

    # Block5: quiet behind full arrangement
    for bar in range(BLK5_S, BLK5_E):
        pad_bar(bar, vel=20)

    # Outro: fade
    for bar in range(OUT_S, OUT_E):
        vel = max(12, 25 - (bar - OUT_S))
        pad_bar(bar, vel=vel)

    return part


# ============================================================================
# DRUMS -- four-on-the-floor, clap 2&4, 16th hats
# ============================================================================

def create_drums(BPM):
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def techno_bar(bar):
        o = bb(bar)
        # Four-on-the-floor kick
        for beat in range(4):
            hit(o + beat, 36, 105 + random.randint(-3, 3))
        # Clap on 2 & 4
        hit(o + 1.0, 39, 90 + random.randint(-3, 3))
        hit(o + 3.0, 39, 92 + random.randint(-3, 3))
        # 16th hats with velocity variation
        for i in range(16):
            beat_pos = i * 0.25
            if i % 4 == 0:
                vel = 50
            elif i % 2 == 0:
                vel = 38
            else:
                vel = 26
            hit(o + beat_pos, 42, vel + random.randint(-4, 4))

    def kick_only_bar(bar):
        o = bb(bar)
        for beat in range(4):
            hit(o + beat, 36, 100 + random.randint(-3, 3))

    # Block1: no drums
    # Block2: kick only (building)
    for bar in range(BLK2_S, BLK2_E):
        kick_only_bar(bar)

    # Block3: full drums
    for bar in range(BLK3_S, BLK3_E):
        techno_bar(bar)

    # Break: no drums

    # Block5: full drums, peak
    for bar in range(BLK5_S, BLK5_E):
        techno_bar(bar)

    # Outro: drums fade
    for bar in range(OUT_S, OUT_S + 8):
        o = bb(bar)
        vel_scale = max(0.3, 1.0 - (bar - OUT_S) * 0.1)
        for beat in range(4):
            hit(o + beat, 36, int(100 * vel_scale))

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
        elif 'bass' in name or 'acid' in name:
            insert_program(track, 38)
        elif 'string' in name:
            insert_program(track, 48)


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
    parser = argparse.ArgumentParser(description='Techno MIDI composer')
    parser.add_argument('--name', required=True)
    parser.add_argument('--key', required=True, choices=['C', 'A'])
    parser.add_argument('--bpm', required=True, type=int)
    args = parser.parse_args()

    BPM = max(124, min(args.bpm, 145))
    scale = SCALES[args.key]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Composing {args.name} ...')
    print(f'  Key: {args.key} Phrygian  |  BPM: {BPM}  |  96 bars')

    parts = {
        'acid':  create_acid(BPM, scale),
        'pad':   create_pad(BPM, scale),
        'drums': create_drums(BPM),
    }

    print('Saving stems ...')
    for stem_name, part in parts.items():
        save(solo(part), f'{args.name}_{stem_name}.mid', OUTPUT_DIR)

    print('\nSaving full arrangement ...')
    full = stream.Score()
    for part in parts.values():
        full.append(part)
    save(full, f'{args.name}_FULL.mid', OUTPUT_DIR)

    print(f'\nDone! -> {OUTPUT_DIR}')
