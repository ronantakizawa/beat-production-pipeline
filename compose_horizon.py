"""
Horizon — Compose (Progressive House)
Ab major | 128 BPM | 96 bars (~3:00)

Research: Path B — 4 YouTube tutorials on Martin Garrix progressive house
  - Major keys for emotional feel, simple catchy melodies
  - Piano chords in breakdowns AND drops (rhythmic in drops)
  - Simple drums: kick every beat, clap on 2+4, ride 8ths
  - Wide pads, lots of reverb, filter sweeps in buildups

Chord Progression (4-bar loop, Ab major — I-V-vi-IV):
  Bar 0: Ab  (Ab3, C4, Eb4)   — I
  Bar 1: Eb  (Eb3, G3, Bb3)   — V
  Bar 2: Fm  (F3, Ab3, C4)    — vi
  Bar 3: Db  (Db3, F3, Ab3)   — IV

7 Sound Layers:
  1. Drums      — kick/clap/ride/hat/crash
  2. Sub bass   — FAUST sine (render only), MIDI roots here
  3. Mid bass   — rhythmic, sample-based
  4. Piano      — chord voicings, stretched/rhythmic
  5. Lead melody — Ab major pentatonic, simple catchy phrase
  6. Pluck      — same melody, octave lower
  7. Pad        — sustained chords

Song Structure (96 bars — 4-bar buildups, 20-bar drops):
  Intro       bars  0– 7: piano (stretched) + pad
  Breakdown 1 bars  8–23: full piano + pad + melody (quiet)
  Buildup 1   bars 24–27: snare roll, kicks enter (4 bars)
  Drop 1      bars 28–47: full energy (20 bars)
  Breakdown 2 bars 48–55: piano + pad only
  Buildup 2   bars 56–59: bigger snare roll (4 bars)
  Drop 2      bars 60–79: higher energy (20 bars)
  Outro       bars 80–95: fade, piano solo
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

BEAT_NAME  = 'Horizon'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Horizon_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 128
BPB = 4

# Section boundaries (4-bar buildups, 20-bar drops)
INTRO_S,  INTRO_E  =  0,  8
BREAK1_S, BREAK1_E =  8, 24
BUILD1_S, BUILD1_E = 24, 28
DROP1_S,  DROP1_E  = 28, 48
BREAK2_S, BREAK2_E = 48, 56
BUILD2_S, BUILD2_E = 56, 60
DROP2_S,  DROP2_E  = 60, 80
OUTRO_S,  OUTRO_E  = 80, 96


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat (0-indexed)."""
    return float(bar * BPB + beat)


def shift_pitch(pitch_str, oct_offset):
    if pitch_str is None or oct_offset == 0:
        return pitch_str
    n = note.Note(pitch_str)
    n.octave += oct_offset
    return n.nameWithOctave


# ============================================================================
# CHORD TABLES — Ab major I-V-vi-IV
# ============================================================================

CHORDS = [
    ['A-3', 'C4',  'E-4'],   # bar 0: Ab major
    ['G3',  'B-3', 'E-4'],   # bar 1: Eb major (1st inv — keeps notes near C4)
    ['F3',  'A-3', 'C4' ],   # bar 2: F minor
    ['D-3', 'F3',  'A-3'],   # bar 3: Db major
]

BASS_ROOTS = ['A-1', 'E-1', 'F1', 'D-1']

# Pad voicings — one octave above CHORDS to keep out of bass range
PAD_CHORDS = [
    ['A-4', 'C5',  'E-5'],   # bar 0: Ab major
    ['G4',  'B-4', 'E-5'],   # bar 1: Eb major (1st inv)
    ['F4',  'A-4', 'C5' ],   # bar 2: F minor
    ['D-4', 'F4',  'A-4'],   # bar 3: Db major
]

# ============================================================================
# MELODY — derived from Ab Major chord tones (chordprogressions/ reference)
# ============================================================================
# Reference voicings (octave 4-5): Ab4-C5-Eb5, Eb4-G4-Bb4, F4-Ab4-C5, Db4-F4-Ab4
# Rule: chord tones on strong beats (1, 3), Ab major scale passing tones between.
# Ab major scale: Ab Bb C Db Eb F G
# Each bar = list of (pitch_or_None, duration_in_beats)

LEAD_MELODY = [
    # Bar 0 (Ab I): rise through chord — Ab4→C5→Eb5, settle on C5
    [('A-4', 0.5), ('C5', 0.5), ('E-5', 1.0), ('C5', 2.0)],
    # Bar 1 (Eb V): chord tone Bb4, neighbor G4, step up to C5
    [('B-4', 1.0), ('G4', 0.5), ('B-4', 0.5), ('C5', 2.0)],
    # Bar 2 (Fm vi): peak F5, descend through chord — Eb5→C5→Ab4
    [('F5', 1.0), ('E-5', 0.5), ('C5', 0.5), ('A-4', 2.0)],
    # Bar 3 (Db IV): stepwise rise Ab4→Bb4→C5, resolve Db5, breathe
    [('A-4', 0.5), ('B-4', 0.5), ('C5', 0.5), ('D-5', 1.5), (None, 1.0)],
]

# Pluck = same melody, one octave lower
PLUCK_MELODY = [
    [(shift_pitch(p, -1) if p else None, d) for p, d in bar]
    for bar in LEAD_MELODY
]


# ============================================================================
# DRUMS
# ============================================================================
# GM mapping: 36=kick 39=clap 51=ride 42=closed-HH 46=open-HH 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def full_drum_bar(bar, crash=False):
        """Full progressive house pattern: kick every beat, clap 2+4, ride 8ths."""
        o = bb(bar)
        # Kick on every beat
        for b in range(4):
            hit(o + b, 36, 100)
        # Clap on beats 2 and 4
        hit(o + 1.0, 39, 88)
        hit(o + 3.0, 39, 92)
        # Ride on 8th notes, alternating velocity
        for i in range(8):
            vel = 52 if i % 2 == 0 else 36
            hit(o + i * 0.5, 51, vel)
        # Open hat every 2 bars on beat 4.5
        if bar % 2 == 1:
            hit(o + 3.5, 46, 50)
        if crash:
            hit(o + 0.0, 49, 85)

    def buildup_bar(bar, phase='early'):
        """Buildup: kicks only (half-time early, every beat late) + snare roll."""
        o = bb(bar)
        if phase == 'early':
            # Half-time kicks
            hit(o + 0.0, 36, 90)
            hit(o + 2.0, 36, 90)
        else:
            # Every beat
            for b in range(4):
                hit(o + b, 36, 95)

    def snare_roll(bar, density='quarter'):
        """Accelerating snare roll: quarter -> 8th -> 16th -> 32nd."""
        o = bb(bar)
        if density == 'quarter':
            for b in range(4):
                hit(o + b, 38, 55 + b * 5)
        elif density == 'eighth':
            for i in range(8):
                hit(o + i * 0.5, 38, 60 + i * 3)
        elif density == 'sixteenth':
            for i in range(16):
                hit(o + i * 0.25, 38, 65 + i * 2)
        elif density == 'thirtysecond':
            for i in range(32):
                hit(o + i * 0.125, 38, 70 + min(i, 20))

    def outro_bar(bar):
        """Outro: kick + ride only, thinning out."""
        o = bb(bar)
        bar_idx = bar - OUTRO_S
        if bar_idx < 8:
            for b in range(4):
                hit(o + b, 36, max(40, 90 - bar_idx * 6))
            for i in range(8):
                vel = max(20, (45 if i % 2 == 0 else 30) - bar_idx * 3)
                hit(o + i * 0.5, 51, vel)

    # No drums in intro (0-7) or breakdowns

    # Buildup 1 (4 bars): kicks + accelerating snare roll
    for bar in range(BUILD1_S, BUILD1_E):
        idx = bar - BUILD1_S
        buildup_bar(bar, 'early' if idx < 2 else 'late')
        # Snare roll: quarter→eighth→sixteenth→32nd over 4 bars
        snare_roll(bar, ['quarter', 'eighth', 'sixteenth', 'thirtysecond'][idx])

    # Drop 1: full pattern
    for bar in range(DROP1_S, DROP1_E):
        full_drum_bar(bar, crash=(bar == DROP1_S))

    # Buildup 2 (4 bars): bigger — starts at 8th, ends at 32nd
    for bar in range(BUILD2_S, BUILD2_E):
        idx = bar - BUILD2_S
        buildup_bar(bar, 'early' if idx < 1 else 'late')
        snare_roll(bar, ['eighth', 'sixteenth', 'thirtysecond', 'thirtysecond'][idx])

    # Drop 2: full pattern
    for bar in range(DROP2_S, DROP2_E):
        full_drum_bar(bar, crash=(bar == DROP2_S))

    # Outro (bars 80-95): thinning
    for bar in range(OUTRO_S, OUTRO_E):
        outro_bar(bar)

    return part


# ============================================================================
# SUB BASS (MIDI for FAUST rendering — roots on beat 1, sustained)
# ============================================================================

def create_sub_bass():
    part = stream.Part()
    part.partName = 'Sub Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Active in drops only
    for s_bar, e_bar in [(DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar):
            root = BASS_ROOTS[bar % 4]
            n = note.Note(root, quarterLength=3.5)
            n.volume.velocity = 90
            part.insert(bb(bar), n)

    return part


# ============================================================================
# MID BASS (sample-based — root on beat 1, ghost on beat 3.5)
# ============================================================================

def create_mid_bass():
    part = stream.Part()
    part.partName = 'Mid Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    for s_bar, e_bar in [(DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar):
            root = BASS_ROOTS[bar % 4]
            # Main hit on beat 1
            n1 = note.Note(root, quarterLength=2.0)
            n1.volume.velocity = 88
            part.insert(bb(bar), n1)
            # Ghost hit on beat 3.5
            n2 = note.Note(root, quarterLength=0.5)
            n2.volume.velocity = 50
            part.insert(bb(bar, 3.5), n2)

    return part


# ============================================================================
# PIANO — chord voicings, varies by section
# ============================================================================

def create_piano():
    part = stream.Part()
    part.partName = 'Piano'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def stretched_chords(s_bar, e_bar, vel=55):
        """Intro/outro: 2 bars per chord, only first 2 chords (Ab, Eb)."""
        for bar in range(s_bar, e_bar, 2):
            idx = ((bar - s_bar) // 2) % 2  # alternate Ab and Eb
            c = chord.Chord(CHORDS[idx], quarterLength=8.0)
            c.volume.velocity = vel
            part.insert(bb(bar), c)

    def sustained_chords(s_bar, e_bar, vel=60):
        """Breakdown: full 4-bar progression, 1 chord per bar, sustained."""
        for bar in range(s_bar, e_bar):
            c = chord.Chord(CHORDS[bar % 4], quarterLength=4.0)
            c.volume.velocity = vel
            part.insert(bb(bar), c)

    def rhythmic_chords(s_bar, e_bar, vel=65):
        """Drop: cut-up chords following 8th note rhythm for energy."""
        for bar in range(s_bar, e_bar):
            tones = CHORDS[bar % 4]
            # Rhythmic pattern: hit on 1, 1.5, 2.5, 3, 4 (offbeat feel)
            hits = [0.0, 0.5, 2.5, 3.0]
            durs = [0.5, 0.5, 0.5, 1.0]
            for h, d in zip(hits, durs):
                c = chord.Chord(tones, quarterLength=d)
                c.volume.velocity = vel + random.randint(-4, 4)
                part.insert(bb(bar, h), c)

    # Intro: stretched
    stretched_chords(INTRO_S, INTRO_E, vel=50)

    # Breakdown 1: sustained
    sustained_chords(BREAK1_S, BREAK1_E, vel=58)

    # No piano in buildups

    # Drop 1: rhythmic
    rhythmic_chords(DROP1_S, DROP1_E, vel=62)

    # Breakdown 2: sustained, emotional
    sustained_chords(BREAK2_S, BREAK2_E, vel=52)

    # Drop 2: rhythmic
    rhythmic_chords(DROP2_S, DROP2_E, vel=65)

    # Outro: stretched, fading
    stretched_chords(OUTRO_S, OUTRO_E, vel=45)

    return part


# ============================================================================
# LEAD MELODY
# ============================================================================

def create_lead():
    part = stream.Part()
    part.partName = 'Lead'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(40, 55)):
        # Hook octave pattern: within each 16-bar hook (4 cycles of 4 bars),
        # shift octaves as [base, base, +1, +1] per workflow.md
        for bar in range(s_bar, e_bar):
            bar_idx = bar % 4
            # Which 4-bar cycle within the current 16-bar hook?
            cycle_in_hook = ((bar - s_bar) % 16) // 4
            oct_shift = 1 if cycle_in_hook >= 2 else 0
            motif = LEAD_MELODY[bar_idx]
            beat = 0.0
            for pitch, dur in motif:
                if pitch is None:
                    part.insert(bb(bar, beat), note.Rest(quarterLength=dur))
                else:
                    p = shift_pitch(pitch, oct_shift) if oct_shift else pitch
                    n = note.Note(p, quarterLength=dur)
                    n.volume.velocity = random.randint(vel_range[0], vel_range[1])
                    part.insert(bb(bar, beat), n)
                beat += dur

    # Breakdown 1: melody enters quietly (last 8 bars)
    write_phrase(BREAK1_S + 8, BREAK1_E, vel_range=(38, 48))

    # Buildup 1: lead continues, building energy
    write_phrase(BUILD1_S, BUILD1_E, vel_range=(48, 58))

    # Drop 1: full melody
    write_phrase(DROP1_S, DROP1_E, vel_range=(55, 70))

    # Buildup 2: lead continues, building energy
    write_phrase(BUILD2_S, BUILD2_E, vel_range=(50, 60))

    # Drop 2: full melody
    write_phrase(DROP2_S, DROP2_E, vel_range=(58, 72))

    return part


# ============================================================================
# PLUCK — same melody, one octave lower, drops only
# ============================================================================

def create_pluck():
    part = stream.Part()
    part.partName = 'Pluck'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(35, 48)):
        for bar in range(s_bar, e_bar):
            bar_idx = bar % 4
            motif = PLUCK_MELODY[bar_idx]
            beat = 0.0
            for pitch, dur in motif:
                if pitch is None:
                    part.insert(bb(bar, beat), note.Rest(quarterLength=dur))
                else:
                    n = note.Note(pitch, quarterLength=dur)
                    n.volume.velocity = random.randint(vel_range[0], vel_range[1])
                    part.insert(bb(bar, beat), n)
                beat += dur

    # Drops only
    write_phrase(DROP1_S, DROP1_E, vel_range=(36, 46))
    write_phrase(DROP2_S, DROP2_E, vel_range=(38, 48))

    return part


# ============================================================================
# PAD — sustained chords, varies intensity
# ============================================================================

def create_pad():
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    sections = {
        'intro':  (INTRO_S,  INTRO_E,  32),
        'break1': (BREAK1_S, BREAK1_E, 38),
        'build1': (BUILD1_S, BUILD1_E, 35),
        'drop1':  (DROP1_S,  DROP1_E,  55),
        'break2': (BREAK2_S, BREAK2_E, 35),
        'build2': (BUILD2_S, BUILD2_E, 40),
        'drop2':  (DROP2_S,  DROP2_E,  58),
        'outro':  (OUTRO_S,  OUTRO_E,  30),
    }

    for section_key, (s_bar, e_bar, vel) in sections.items():
        for bar in range(s_bar, e_bar):
            c = chord.Chord(CHORDS[bar % 4], quarterLength=4.0)
            c.volume.velocity = vel
            part.insert(bb(bar), c)

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
    '808':   38,
    'bass':  38,
    'sub':   38,
    'mid':   38,
    'pad':   89,
    'pluck': 80,
    'piano': 0,
    'lead':  80,
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
    print(f'  Key: Ab major  |  BPM: {BPM}  |  96 bars')

    parts = {
        'drums':    create_drums(),
        'sub_bass': create_sub_bass(),
        'mid_bass': create_mid_bass(),
        'piano':    create_piano(),
        'lead':     create_lead(),
        'pluck':    create_pluck(),
        'pad':      create_pad(),
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
