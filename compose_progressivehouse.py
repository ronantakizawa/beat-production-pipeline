"""
Progressive House -- MIDI Composition (Martin Garrix Style)
Major keys, emotional, uplifting. Simple catchy melodies, piano chords,
pluck arps, atmospheric pads.

Layers:
  1. Piano Chords -- sustained in breakdowns, rhythmic stabs in drops
  2. Lead Melody  -- simple catchy phrase, flute, C5-C6
  3. Pluck Arp    -- 16th note arpeggios through chord tones
  4. Pad          -- sustained whole-note chords, atmospheric
  5. Drums        -- four-on-the-floor kick, clap 2&4, 16th hats
  6. Bass         -- chord roots, quarter notes in drops

Chord Progression: I-V-vi-IV (anthem progression)
  Key C:  C  -> G  -> Am -> F
  Key G:  G  -> D  -> Em -> C
  Key D:  D  -> A  -> Bm -> G
  Key Ab: Ab -> Eb -> Fm -> Db

Song Structure (96 bars):
  Intro      0- 7: pad only (8 bars)
  Breakdown1 8-23: piano + melody enters bar 12, pluck bar 16 (16 bars)
  Buildup1  24-27: snare roll, all elements (4 bars)
  Drop1     28-43: full drums + bass + rhythmic piano + lead + pluck (16 bars)
  Breakdown2 44-55: piano + pad + melody octave lower (12 bars)
  Buildup2  56-59: snare roll (4 bars)
  Drop2     60-75: full energy (16 bars)
  Outro     76-95: drums + pad fade (20 bars)

Usage:
  python compose_progressivehouse.py --name "ProgHouse_Anthem" --key C --bpm 128
"""

import argparse
import os
import random
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)

OUTPUT_DIR = '/Users/ronantakizawa/Documents/ProgressiveHouse_Beat'
BPB = 4

INTRO_S,  INTRO_E  =  0,  8
BD1_S,    BD1_E    =  8, 24
BU1_S,    BU1_E    = 24, 28
DROP1_S,  DROP1_E  = 28, 44
BD2_S,    BD2_E    = 44, 56
BU2_S,    BU2_E    = 56, 60
DROP2_S,  DROP2_E  = 60, 76
OUTRO_S,  OUTRO_E  = 76, 96


def bb(bar, beat=0.0):
    return float(bar * BPB + beat)


# ============================================================================
# CHORD PROGRESSIONS — I-V-vi-IV
# ============================================================================

CHORDS_C = {
    'piano': [
        ['C3', 'E3', 'G3'],        # C major
        ['B2', 'D3', 'G3'],        # G/B (first inversion for voice leading)
        ['A2', 'C3', 'E3'],        # Am
        ['A2', 'C3', 'F3'],        # F/A (first inversion)
    ],
    'piano_high': [
        ['C4', 'E4', 'G4'],
        ['B3', 'D4', 'G4'],
        ['A3', 'C4', 'E4'],
        ['A3', 'C4', 'F4'],
    ],
    'pluck_tones': [
        ['C4', 'E4', 'G4', 'C5'],
        ['G3', 'B3', 'D4', 'G4'],
        ['A3', 'C4', 'E4', 'A4'],
        ['F3', 'A3', 'C4', 'F4'],
    ],
    'melody': ['E5', 'D5', 'C5', 'D5', 'E5', 'G5', 'E5', 'D5'],
    'pad': [
        ['C3', 'G3', 'E4'],
        ['G2', 'D3', 'B3'],
        ['A2', 'E3', 'C4'],
        ['F2', 'C3', 'A3'],
    ],
    'bass_roots': ['C2', 'G1', 'A1', 'F1'],
}

CHORDS_G = {
    'piano': [
        ['G3', 'B3', 'D4'],
        ['F#3', 'A3', 'D4'],
        ['E3', 'G3', 'B3'],
        ['E3', 'G3', 'C4'],
    ],
    'piano_high': [
        ['G4', 'B4', 'D5'],
        ['F#4', 'A4', 'D5'],
        ['E4', 'G4', 'B4'],
        ['E4', 'G4', 'C5'],
    ],
    'pluck_tones': [
        ['G3', 'B3', 'D4', 'G4'],
        ['D3', 'F#3', 'A3', 'D4'],
        ['E3', 'G3', 'B3', 'E4'],
        ['C3', 'E3', 'G3', 'C4'],
    ],
    'melody': ['B5', 'A5', 'G5', 'A5', 'B5', 'D6', 'B5', 'A5'],
    'pad': [
        ['G3', 'D4', 'B4'],
        ['D3', 'A3', 'F#4'],
        ['E3', 'B3', 'G4'],
        ['C3', 'G3', 'E4'],
    ],
    'bass_roots': ['G2', 'D2', 'E2', 'C2'],
}

CHORDS_D = {
    'piano': [
        ['D3', 'F#3', 'A3'],
        ['C#3', 'E3', 'A3'],
        ['B2', 'D3', 'F#3'],
        ['B2', 'D3', 'G3'],
    ],
    'piano_high': [
        ['D4', 'F#4', 'A4'],
        ['C#4', 'E4', 'A4'],
        ['B3', 'D4', 'F#4'],
        ['B3', 'D4', 'G4'],
    ],
    'pluck_tones': [
        ['D4', 'F#4', 'A4', 'D5'],
        ['A3', 'C#4', 'E4', 'A4'],
        ['B3', 'D4', 'F#4', 'B4'],
        ['G3', 'B3', 'D4', 'G4'],
    ],
    'melody': ['F#5', 'E5', 'D5', 'E5', 'F#5', 'A5', 'F#5', 'E5'],
    'pad': [
        ['D3', 'A3', 'F#4'],
        ['A2', 'E3', 'C#4'],
        ['B2', 'F#3', 'D4'],
        ['G2', 'D3', 'B3'],
    ],
    'bass_roots': ['D2', 'A1', 'B1', 'G1'],
}

CHORDS_AB = {
    'piano': [
        ['Ab3', 'C4', 'Eb4'],
        ['G3', 'Bb3', 'Eb4'],
        ['F3', 'Ab3', 'C4'],
        ['F3', 'Ab3', 'Db4'],
    ],
    'piano_high': [
        ['Ab4', 'C5', 'Eb5'],
        ['G4', 'Bb4', 'Eb5'],
        ['F4', 'Ab4', 'C5'],
        ['F4', 'Ab4', 'Db5'],
    ],
    'pluck_tones': [
        ['Ab3', 'C4', 'Eb4', 'Ab4'],
        ['Eb3', 'G3', 'Bb3', 'Eb4'],
        ['F3', 'Ab3', 'C4', 'F4'],
        ['Db3', 'F3', 'Ab3', 'Db4'],
    ],
    'melody': ['C5', 'Bb4', 'Ab4', 'Bb4', 'C5', 'Eb5', 'C5', 'Bb4'],
    'pad': [
        ['Ab3', 'Eb4', 'C5'],
        ['Eb3', 'Bb3', 'G4'],
        ['F3', 'C4', 'Ab4'],
        ['Db3', 'Ab3', 'F4'],
    ],
    'bass_roots': ['Ab1', 'Eb2', 'F1', 'Db2'],
}

PROGRESSION = {'C': CHORDS_C, 'G': CHORDS_G, 'D': CHORDS_D, 'Ab': CHORDS_AB}


# ============================================================================
# PIANO — sustained in breakdowns, rhythmic stabs in drops
# ============================================================================

def create_piano(BPM, chords):
    part = stream.Part()
    part.partName = 'Grand Piano'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    lo = chords['piano']
    hi = chords['piano_high']

    def sustained_bar(bar, vel=44):
        o = bb(bar)
        ch = chord.Chord(lo[bar % 4], quarterLength=4.0)
        ch.volume.velocity = vel
        part.insert(o, ch)

    def rhythmic_bar(bar, vel=56):
        o = bb(bar)
        tones = hi[bar % 4]
        # 8th note stabs: hits on beats 1, 1.5, 2.5, 3, 4
        pattern = [0.0, 0.5, 2.5, 3.0]
        for beat in pattern:
            ch = chord.Chord(tones, quarterLength=0.5)
            ch.volume.velocity = vel + random.randint(-3, 3)
            part.insert(o + beat, ch)

    # Breakdown1: sustained piano enters bar 8
    for bar in range(BD1_S, BD1_E):
        sustained_bar(bar, vel=40 + min(8, (bar - BD1_S)))

    # Buildup1: sustained, building
    for bar in range(BU1_S, BU1_E):
        sustained_bar(bar, vel=50)

    # Drop1: rhythmic stabs
    for bar in range(DROP1_S, DROP1_E):
        rhythmic_bar(bar, vel=56)

    # Breakdown2: sustained, quieter
    for bar in range(BD2_S, BD2_E):
        sustained_bar(bar, vel=38)

    # Buildup2
    for bar in range(BU2_S, BU2_E):
        sustained_bar(bar, vel=48)

    # Drop2: rhythmic stabs, louder
    for bar in range(DROP2_S, DROP2_E):
        rhythmic_bar(bar, vel=60)

    # Outro: sustained, fading
    for bar in range(OUTRO_S, min(OUTRO_S + 8, OUTRO_E)):
        vel = max(20, 40 - (bar - OUTRO_S) * 4)
        sustained_bar(bar, vel=vel)

    return part


# ============================================================================
# LEAD MELODY — simple, catchy, repetitive
# ============================================================================

def create_lead(BPM, chords):
    part = stream.Part()
    part.partName = 'Flute Solo'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    mel = chords['melody']  # 8-note phrase spanning 4 bars

    def melody_phrase(start_bar, vel_base=52, octave_shift=0):
        """Write 4-bar melody phrase (2 notes per bar)."""
        for i, pitch in enumerate(mel):
            bar = start_bar + i // 2
            beat = (i % 2) * 2.0
            o = bb(bar, beat)
            n = note.Note(pitch, quarterLength=1.5)
            if octave_shift != 0:
                n.transpose(octave_shift * 12, inPlace=True)
            n.volume.velocity = vel_base + random.randint(-3, 3)
            part.insert(o, n)

    # Breakdown1: melody enters bar 12
    for phrase_start in range(BD1_S + 4, BD1_E, 4):
        melody_phrase(phrase_start, vel_base=44)

    # Drop1: full melody
    for phrase_start in range(DROP1_S, DROP1_E, 4):
        melody_phrase(phrase_start, vel_base=56)

    # Breakdown2: melody octave lower
    for phrase_start in range(BD2_S, BD2_E, 4):
        melody_phrase(phrase_start, vel_base=40, octave_shift=-1)

    # Drop2: full melody, louder
    for phrase_start in range(DROP2_S, DROP2_E, 4):
        melody_phrase(phrase_start, vel_base=60)

    return part


# ============================================================================
# PLUCK ARP — 16th note arpeggios through chord tones
# ============================================================================

def create_pluck(BPM, chords):
    part = stream.Part()
    part.partName = 'Harp'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    tones = chords['pluck_tones']

    def arp_bar(bar, vel_base=38):
        o = bb(bar)
        ct = tones[bar % 4]
        for i in range(16):
            pitch = ct[i % len(ct)]
            beat_pos = i * 0.25
            vel = vel_base + random.randint(-3, 3)
            if i % 4 == 0:
                vel = min(127, vel + 5)
            n = note.Note(pitch, quarterLength=0.25)
            n.volume.velocity = vel
            part.insert(o + beat_pos, n)

    # Breakdown1: pluck enters bar 16 (staggered after melody)
    for bar in range(BD1_S + 8, BD1_E):
        arp_bar(bar, vel_base=34)

    # Buildup1
    for bar in range(BU1_S, BU1_E):
        arp_bar(bar, vel_base=38)

    # Drop1
    for bar in range(DROP1_S, DROP1_E):
        arp_bar(bar, vel_base=40)

    # Breakdown2: no pluck (stripped)

    # Buildup2
    for bar in range(BU2_S, BU2_E):
        arp_bar(bar, vel_base=36)

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        arp_bar(bar, vel_base=42)

    return part


# ============================================================================
# PAD — sustained whole-note chords, atmospheric
# ============================================================================

def create_pad(BPM, chords):
    part = stream.Part()
    part.partName = 'String Ensemble'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    pads = chords['pad']

    def pad_bar(bar, vel=30):
        o = bb(bar)
        ch = chord.Chord(pads[bar % 4], quarterLength=4.0)
        ch.volume.velocity = vel
        part.insert(o, ch)

    # Intro: pad only
    for bar in range(INTRO_S, INTRO_E):
        pad_bar(bar, vel=25 + (bar - INTRO_S) * 2)

    # Breakdown1
    for bar in range(BD1_S, BD1_E):
        pad_bar(bar, vel=32)

    # Buildup1
    for bar in range(BU1_S, BU1_E):
        pad_bar(bar, vel=36)

    # Drop1: pad quieter behind lead
    for bar in range(DROP1_S, DROP1_E):
        pad_bar(bar, vel=24)

    # Breakdown2: pad prominent
    for bar in range(BD2_S, BD2_E):
        pad_bar(bar, vel=36)

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        pad_bar(bar, vel=26)

    # Outro: pad fades
    for bar in range(OUTRO_S, OUTRO_E):
        vel = max(15, 30 - (bar - OUTRO_S))
        pad_bar(bar, vel=vel)

    return part


# ============================================================================
# DRUMS — four-on-the-floor kick, clap 2&4, 16th hats
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

    def house_bar(bar, crash=False):
        o = bb(bar)
        for beat in range(4):
            hit(o + beat, 36, 100 + random.randint(-3, 3))
        hit(o + 1.0, 39, 92 + random.randint(-3, 3))
        hit(o + 3.0, 39, 94 + random.randint(-3, 3))
        for i in range(8):
            beat_pos = i * 0.5
            vel = 48 if i % 2 == 0 else 34
            hit(o + beat_pos, 42, vel + random.randint(-3, 3))
        if crash:
            hit(o, 49, 85)

    def buildup_bar(bar, density):
        """Snare roll with increasing density."""
        o = bb(bar)
        hits_per_beat = min(8, 2 + density * 2)
        for i in range(int(hits_per_beat * 4)):
            beat = i / hits_per_beat
            if beat >= 4.0:
                break
            vel = 40 + density * 12 + random.randint(-3, 3)
            hit(o + beat, 39, min(110, int(vel)))
        # Keep kick
        for beat in range(4):
            hit(o + beat, 36, 95)

    # No drums in intro or breakdowns

    # Buildup1: snare roll
    for i, bar in enumerate(range(BU1_S, BU1_E)):
        buildup_bar(bar, density=i)

    # Drop1
    for bar in range(DROP1_S, DROP1_E):
        house_bar(bar, crash=(bar == DROP1_S))

    # Buildup2
    for i, bar in enumerate(range(BU2_S, BU2_E)):
        buildup_bar(bar, density=i)

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        house_bar(bar, crash=(bar == DROP2_S))

    # Outro: drums fade
    for bar in range(OUTRO_S, min(OUTRO_S + 12, OUTRO_E)):
        o = bb(bar)
        vel_scale = max(0.3, 1.0 - (bar - OUTRO_S) * 0.07)
        for beat in range(4):
            hit(o + beat, 36, int(95 * vel_scale))
        hit(o + 1.0, 39, int(85 * vel_scale))
        hit(o + 3.0, 39, int(85 * vel_scale))

    return part


# ============================================================================
# BASS — chord roots, quarter notes in drops
# ============================================================================

def create_bass(BPM, chords):
    part = stream.Part()
    part.partName = 'Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    roots = chords['bass_roots']

    def bass_drop_bar(bar, vel=90):
        o = bb(bar)
        root = roots[bar % 4]
        # Quarter notes on beats 1 and 3
        n1 = note.Note(root, quarterLength=1.5)
        n1.volume.velocity = vel
        part.insert(o, n1)
        n2 = note.Note(root, quarterLength=1.5)
        n2.volume.velocity = int(vel * 0.85)
        part.insert(o + 2.0, n2)

    # Drop1
    for bar in range(DROP1_S, DROP1_E):
        bass_drop_bar(bar, vel=92)

    # Drop2
    for bar in range(DROP2_S, DROP2_E):
        bass_drop_bar(bar, vel=96)

    # Outro: bass fades
    for bar in range(OUTRO_S, min(OUTRO_S + 8, OUTRO_E)):
        vel = max(40, 85 - (bar - OUTRO_S) * 8)
        bass_drop_bar(bar, vel=vel)

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
    'piano': 0,
    'flute': 73,
    'harp':  46,
    'string': 48,
    'bass':  38,
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
    parser = argparse.ArgumentParser(description='Progressive House MIDI composer')
    parser.add_argument('--name', required=True)
    parser.add_argument('--key', required=True, choices=['C', 'G', 'D', 'Ab'])
    parser.add_argument('--bpm', required=True, type=int)
    args = parser.parse_args()

    BPM = max(118, min(args.bpm, 135))
    chords = PROGRESSION[args.key]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Composing {args.name} ...')
    print(f'  Key: {args.key} major  |  BPM: {BPM}  |  96 bars')

    parts = {
        'piano':  create_piano(BPM, chords),
        'lead':   create_lead(BPM, chords),
        'pluck':  create_pluck(BPM, chords),
        'pad':    create_pad(BPM, chords),
        'drums':  create_drums(BPM),
        'bass':   create_bass(BPM, chords),
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
