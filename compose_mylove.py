"""
MyLove — Compose (Floyy Menor style modern reggaeton)
Key: A major (11B Camelot) | BPM: 100 | 72 bars (~2:53)

Chord Progression (4-bar loop, A major — I-V-vi-IV):
  Bar 0: A    (A3, C#4, E4)    — I  (tonic)
  Bar 1: E    (E3, G#3, B3)    — V  (dominant)
  Bar 2: F#m  (F#3, A3, C#4)   — vi (submediant)
  Bar 3: D    (D4, F#4, A4)    — IV (subdominant)

5 Sound Layers:
  1. Drums      — dembow pattern (kick + rim + hh + oh + shaker + crash)
  2. Sub bass   — sub_bass following chord roots A1/E1/F#1/D1
  3. Pad        — sustained string chords
  4. Lead       — A major lead synth motif (E5-C#5-A4)
  5. Reversed FX — transition markers

Song Structure (7 sections, 72 bars):
  intro    bars  0-8:   pad+lead only, filter sweep in
  verse1   bars  8-12:  drums+bass enter
  hook1    bars 12-28:  full kit
  bridge1  bars 28-32:  pad+lead only (drop)
  bridge2  bars 32-36:  drums+bass re-enter
  hook2    bars 36-60:  full kit
  outro    bars 60-72:  bass+pad+lead, fade out

Kit: REGGAETON 4 + reggaeton5 XXL + reggaeton3 URBANITO
"""

import os
import random
import numpy as np
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================

BEAT_NAME  = 'MyLove'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/MyLove_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM   = 100
BPB   = 4
NBARS = 72

# Section boundaries
SECTIONS = {
    'intro':   (0,   8),
    'verse1':  (8,  12),
    'hook1':   (12, 28),
    'bridge1': (28, 32),
    'bridge2': (32, 36),
    'hook2':   (36, 60),
    'outro':   (60, 72),
}

# Energy curve (72 values — builds through intro, peaks at hooks, fades at outro)
ENERGY_CURVE = [
    0.05, 0.08, 0.12, 0.18, 0.22, 0.28, 0.32, 0.38,     # intro (0-7)
    0.45, 0.52, 0.58, 0.62,                                # verse1 (8-11)
    0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95,       # hook1 (12-19)
    0.97, 1.00, 0.98, 0.95, 0.92, 0.88, 0.85, 0.80,       # hook1 (20-27)
    0.30, 0.25, 0.20, 0.18,                                # bridge1 (28-31)
    0.40, 0.50, 0.58, 0.65,                                # bridge2 (32-35)
    0.72, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95,       # hook2 (36-43)
    0.97, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88,       # hook2 (44-51)
    0.86, 0.84, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70,       # hook2 (52-59)
    0.60, 0.50, 0.42, 0.35, 0.28, 0.22, 0.18, 0.14,       # outro (60-67)
    0.10, 0.06, 0.03, 0.00,                                # outro (68-71)
]


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat (0-indexed)."""
    return float(bar * BPB + beat)


def energy_vel(bar, base_vel, min_scale=0.5):
    """Scale velocity by energy curve for dynamic arrangement."""
    if bar < len(ENERGY_CURVE):
        e = ENERGY_CURVE[bar]
        scale = min_scale + (1.0 - min_scale) * e
        return int(np.clip(base_vel * scale, 1, 127))
    return base_vel


# ============================================================================
# CHORD TABLES — A major: I-V-vi-IV
# ============================================================================

CHORDS = [
    ['A3',  'C#4', 'E4'],      # bar 0: A   (I)
    ['E3',  'G#3', 'B3'],      # bar 1: E   (V)
    ['F#3', 'A3',  'C#4'],     # bar 2: F#m (vi)
    ['D4',  'F#4', 'A4'],      # bar 3: D   (IV)
]

# Bass roots following chord progression in A major
BASS_ROOTS = ['A1', 'E1', 'F#1', 'D1']


# ============================================================================
# DRUMS — dembow pattern
# ============================================================================
# GM drum mapping: 36=kick 37=rim 38=snare
#                  42=closed-HH 46=open-HH 49=crash 70=shaker(maracas)

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    # Dembow patterns
    KICK_A = [0.0, 2.0]                                       # beats 1, 3
    KICK_B = [0.0, 0.75, 2.0]                                 # bouncy variant
    RIM_A  = [1.0, 1.75, 3.0, 3.75]                           # dembow syncopation
    RIM_B  = [1.0, 1.75, 2.5, 3.75]                           # shifted variant
    HH_8TH  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]      # straight 8ths


    def drum_bar(bar, fill=False):
        o = bb(bar)
        vs = energy_vel(bar, 100) / 100.0
        bar_in_cycle = bar % 4

        # Alternate kick pattern every other bar
        kicks = KICK_B if bar_in_cycle in (1, 3) else KICK_A
        kick_set = set(kicks)

        for b in kicks:
            hit(o + b, 36, 100 * vs)

        # Fill bar: snare roll on beats 3-4 (skip beats that overlap kick)
        if fill:
            for b in [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]:
                if b not in kick_set:
                    hit(o + b, 37, int(28 * vs * (0.7 + 0.3 * b / 4.0)))
            for b in RIM_A[:2]:
                if b not in kick_set:
                    hit(o + b, 37, 35 * vs)
        else:
            # Alternate rim pattern on bar 3 of cycle (skip kick beats)
            rims = RIM_B if bar_in_cycle == 2 else RIM_A
            for b in rims:
                if b not in kick_set:
                    hit(o + b, 37, 35 * vs)

        # Hi-hats: straight 8ths, skip beats that overlap kick
        for b in HH_8TH:
            if b not in kick_set:
                hit(o + b, 42, 55 * vs)

        # Open hat on odd bars (skip if overlaps kick)
        if bar % 2 == 1:
            if 3.5 not in kick_set:
                hit(o + 3.5, 46, 50 * vs)

    # Drums in: verse1, hook1, bridge2, hook2 (not intro, bridge1, outro)
    drum_sections = ['verse1', 'hook1', 'bridge2', 'hook2']
    for sec_name in drum_sections:
        s, e = SECTIONS[sec_name]
        for bar in range(s, e):
            is_fill = (bar == e - 1 and sec_name not in ('outro',))
            drum_bar(bar, fill=is_fill)

    return part


# ============================================================================
# SUB BASS — A major roots
# ============================================================================

def create_bass():
    """Sub bass following chord roots in A major.
    SUB BASS TIMBRE RULE: sine osc, LPF 200Hz, no harmonics above 200Hz."""
    part = stream.Part()
    part.partName = 'Sub Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Syncopated bass pattern from ref
    BASS_EVENTS = [
        (0.0,  2.0),    # main hit, sustained
        (2.5,  0.75),   # syncopated pickup
        (3.5,  0.5),    # anticipation
    ]

    # Bass active in: verse1, hook1, bridge2, hook2, outro
    bass_sections = ['verse1', 'hook1', 'bridge2', 'hook2', 'outro']
    for sec_name in bass_sections:
        s, e = SECTIONS[sec_name]
        for bar in range(s, e):
            root = BASS_ROOTS[bar % 4]
            vel = energy_vel(bar, 90)
            for beat, dur in BASS_EVENTS:
                n = note.Note(root, quarterLength=dur)
                n.volume.velocity = vel if beat == 0.0 else int(vel * 0.60)
                part.insert(bb(bar, beat), n)

    return part


# ============================================================================
# PAD — sustained chords (strings timbre)
# ============================================================================

def create_pad():
    """Sustained string pad from CHORDS table. Active in all sections.
    PAD TIMBRE RULE: 5x detuned saw, slow attack 0.5s, long release 3s,
    vel 35-50. HPF at 150Hz. Never louder than drums."""
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    for bar in range(NBARS):
        c = chord.Chord(CHORDS[bar % 4], quarterLength=4.0)
        vel = energy_vel(bar, 48, min_scale=0.4)
        c.volume.velocity = vel
        part.insert(bb(bar), c)

    return part


# ============================================================================
# LEAD MELODY — A major lead synth, 4-bar phrases following chord tones
# Voicings from chordprogressions/GitHub Free Progressions/A/A Major/
#   A(I): E5, C#5, A4  |  E(V): G#4, B4, E5
#   F#m(vi): F#4, A4, C#5  |  D(IV): D5, F#5, A5
# ============================================================================

def create_lead():
    """Lead synth in A major. 4-bar phrase (with variation) using chord tones
    from GitHub Free Progressions A Major voicings.
    Scale: A B C# D E F# G#. Range: F#4-F#5.
    LEAD TIMBRE RULE: sine+FM, medium attack, vel 35-48.
    Never louder than drums/bass. Do NOT use saw waves."""
    part = stream.Part()
    part.partName = 'Lead'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # 4-bar phrase A: follows A-E-F#m-D chord tones, dotted rhythms + rests
    PHRASE_A = [
        # Bar 0 (A): leap up to E5, ornament down to C#5, breathe
        [('E5', 0.75), ('C#5', 0.25), (None, 0.5), ('E5', 0.5),
         ('D5', 0.25), ('C#5', 0.25), ('A4', 1.0), (None, 0.5)],
        # Bar 1 (E): G#4 pickup into B4, leap to E5, fall back
        [('B4', 0.5), ('G#4', 0.75), (None, 0.25), ('B4', 0.5),
         ('E5', 0.75), (None, 0.25), ('G#4', 0.5), (None, 0.5)],
        # Bar 2 (F#m): A4 anchor, reach up to C#5, settle on F#4
        [('A4', 0.5), ('C#5', 0.75), (None, 0.25), ('F#4', 0.5),
         ('A4', 0.5), ('C#5', 0.5), (None, 0.5), (None, 0.5)],
        # Bar 3 (D): D5 dotted, leap to F#5 peak, resolve down
        [('D5', 0.75), ('F#5', 0.25), (None, 0.5), ('D5', 0.5),
         ('A4', 0.25), ('F#4', 0.25), ('D5', 1.0), (None, 0.5)],
    ]

    # 4-bar phrase B: variation — inverted contour, syncopated
    PHRASE_B = [
        # Bar 0 (A): start low, climb through C#5 to E5
        [('C#5', 0.5), ('E5', 0.5), (None, 0.25), ('A4', 0.75),
         (None, 0.5), ('E5', 0.5), ('C#5', 0.5), (None, 0.5)],
        # Bar 1 (E): syncopated G#-B oscillation
        [('G#4', 0.75), ('B4', 0.25), ('E5', 0.5), (None, 0.5),
         ('B4', 0.75), (None, 0.25), ('G#4', 0.75), (None, 0.25)],
        # Bar 2 (F#m): F#4 start, stepwise rise to C#5
        [('F#4', 0.5), ('A4', 0.5), ('C#5', 0.5), (None, 0.25),
         ('A4', 0.75), (None, 0.5), ('F#4', 0.75), (None, 0.25)],
        # Bar 3 (D): A4 pickup, D5 dotted, F#5 peak, long resolve
        [('A4', 0.5), ('D5', 0.75), (None, 0.25), ('F#5', 0.5),
         (None, 0.5), ('D5', 0.5), ('A4', 0.75), (None, 0.25)],
    ]

    for bar in range(NBARS):
        # Alternate between phrase A and B every 4 bars
        cycle = (bar // 4) % 2
        phrase = PHRASE_B if cycle == 1 else PHRASE_A
        phrase_bar = phrase[bar % 4]

        base_vel = 42
        if bar < SECTIONS['intro'][1]:
            base_vel = 30
        vel = energy_vel(bar, base_vel, min_scale=0.4)

        beat = 0.0
        for pitch, dur in phrase_bar:
            if pitch is None:
                part.insert(bb(bar, beat), note.Rest(quarterLength=dur))
            else:
                nd = note.Note(pitch, quarterLength=dur)
                nd.volume.velocity = int(np.clip(
                    vel + random.randint(-3, 3), 1, 55))
                part.insert(bb(bar, beat), nd)
            beat += dur

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
    'sub':    38,   # Synth Bass 1
    'bass':   38,
    'pad':    48,   # String Ensemble 1
    'lead':   80,   # Lead 1 (square) — lead_synth
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
# COMPOSE & SAVE
# ============================================================================

if __name__ == '__main__':
    print(f'Composing {BEAT_NAME} ...')
    print(f'  Key: A major (11B)  |  BPM: {BPM}  |  {NBARS} bars')
    print(f'  Genre: Reggaeton  |  Chords: A-E-F#m-D (I-V-vi-IV)')

    parts = {
        'drums':    create_drums(),
        'sub_bass': create_bass(),
        'pad':      create_pad(),
        'lead':     create_lead(),
    }

    print('Saving individual stems ...')
    for stem_name, part in parts.items():
        save(solo(part), f'{BEAT_NAME}_{stem_name}.mid')

    print('\nSaving full arrangement ...')
    full = stream.Score()
    for part in parts.values():
        full.append(part)
    save(full, f'{BEAT_NAME}_FULL.mid')

    print(f'\nDone! MIDI files saved to:\n  {OUTPUT_DIR}')
