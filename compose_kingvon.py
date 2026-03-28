"""
King Von Type Beat — Chicago Drill Compose
Key: C minor | BPM: 120 | 64 bars (~2:08)

Research:
  - Chopsquad DJ "Took Her To The O" reference: 161.5 BPM, C minor confirmed
    (chroma: C=1.000, Eb=0.853, F=0.618, G dominant)
  - Tutorial dg5PMqQ-u2s: strings power chords, bouncy piano on 5th (G),
    clap on 2/4, hi-hat rolls, 808 mirrors kick bounce
  - Tutorial iy12iNgtPus: synth pluck + pad, piano + violin layers,
    hi-hats two-step + rolls, 808 short-medium, rim bouncy pattern

Reference voicings:
  chordprogressions/GitHub Free Progressions/C/C Minor/ — 3-note triads in C4-C5 range
    Cm: [C4, Eb4, G4]  Ab: [Ab3, C4, Eb4]  Fm: [F4, Ab4, C5]  Gm: [G3, Bb3, D4]
  chordprogressions/More Genres/Trap Melodic.mid — Fm [C4, F4, Ab4] + C [C4, E4, G4]
  chordprogressions/More Genres/Trap Dark.mid — dark voicings in octave 4

Chord Progression (4-bar loop, C minor — i-VI-III-VII):
  Bar 0: Cm    (C4 Eb4 G4)   — i  (tonic, minor triad)
  Bar 1: Ab    (Ab3 C4 Eb4)  — VI (major triad, 1st inv feel)
  Bar 2: Eb    (Eb4 G4 Bb4)  — III
  Bar 3: Bb    (Bb3 D4 F4)   — VII

5 Sound Layers:
  1. Drums     — Chicago drill: kick+clap+hh+perc+crash
  2. 808 Bass  — short-medium, bouncy, follows kick
  3. Strings   — sustained minor triads (octave 4, reference voicings)
  4. Piano     — bouncy repetitive melody emphasizing G4 (fifth)
  5. Lead Synth— dark pluck, sparse, hook sections only

Song Structure:
  Intro     bars  0– 7: strings + piano only (cut melody bar 7)
  Hook A    bars  8–23: full arrangement
  Verse     bars 24–39: drums + 808 + piano (sparse)
  Bridge    bars 40–47: strings + lead only (drums drop)
  Hook B    bars 48–63: full arrangement

Kit: Juicy Jules Stardust
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

BEAT_NAME  = 'KingVon'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/KingVon_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 120
BPB = 4

# Section boundaries
INTRO_S,  INTRO_E  =  0,  8
HOOKA_S,  HOOKA_E  =  8, 24
VERSE_S,  VERSE_E  = 24, 40
BRIDGE_S, BRIDGE_E = 40, 48
HOOKB_S,  HOOKB_E  = 48, 64


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat."""
    return float(bar * BPB + beat)


# ============================================================================
# CHORD TABLES — C minor: i-VI-III-VII
# ============================================================================
# Voicings informed by chordprogressions/GitHub Free Progressions/C/C Minor/
# and More Genres/Trap Dark.mid — triads in octave 4 range for clarity.
# Full minor triads (keep the 3rd!) so minor tonality is unmistakable.

# String voicings — octave 3-4, darker register for menacing feel.
# Reference: C Minor triads from chordprogressions/, dropped an octave.
# Add D (9th) to Cm for dissonant tension (Cm add9 = C Eb G D).
STRING_CHORDS = [
    ['C3',  'Eb3', 'G3',  'D4'],   # Cm(add9) — dark, dissonant
    ['Ab2', 'C3',  'Eb3'],          # Ab       — low, heavy
    ['Eb3', 'G3',  'Bb3'],          # Eb       — mid-low
    ['Bb2', 'D3',  'F3',  'Ab3'],   # Bb7      — VII7, dark tension
]

# Piano chord tones — octave 4 (sits above strings)
PIANO_CHORDS = [
    ['C4',  'Eb4', 'G4'],   # Cm
    ['Ab3', 'C4',  'Eb4'],  # Ab
    ['Eb4', 'G4',  'Bb4'],  # Eb
    ['Bb3', 'D4',  'F4'],   # Bb
]

BASS_ROOTS = ['C2', 'Ab1', 'Eb2', 'Bb1']


# ============================================================================
# DRUMS — Chicago drill pattern
# ============================================================================
# GM mapping: 36=kick, 38=snare, 39=clap, 42=closed-HH, 46=open-HH,
#             37=rim/perc, 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def drill_bar(bar, full=True, half_time=False, crash=False):
        o = bb(bar)

        # Kick: beat 1, beat 3, occasional syncopation
        hit(o + 0.0, 36, 100)
        hit(o + 2.0, 36, 92)
        # Syncopated kick on beat 2.5 or 3.5 (alternating bars)
        if bar % 2 == 0:
            hit(o + 2.5, 36, 70)
        else:
            hit(o + 3.5, 36, 65)

        # Clap on beats 2 and 4 (Chicago drill signature)
        if half_time:
            hit(o + 3.0, 39, 90)
        else:
            hit(o + 1.0, 39, 92)
            hit(o + 3.0, 39, 96)

        # Hi-hats — two-step base + rolls
        if full:
            for i in range(16):  # 16th note grid
                beat_pos = i * 0.25
                if i % 4 == 0:
                    # On-beat: closed hat, medium vel
                    hit(o + beat_pos, 42, 55)
                elif i % 4 == 2:
                    # Offbeat 8th: closed hat, lighter
                    hit(o + beat_pos, 42, 40)
                elif i >= 8:
                    # Hi-hat rolls on beats 3-4 (16th bursts)
                    hit(o + beat_pos, 42, random.randint(28, 45))
            # Open hat accent on beat 4 offbeat
            if bar % 2 == 1:
                hit(o + 3.5, 46, 50)
        else:
            # Sparse hats (verse)
            for i in range(8):
                hit(o + i * 0.5, 42, random.randint(28, 42))

        # Bouncy perc/rim — offbeat hits (Chicago drill signature)
        if full:
            hit(o + 0.5, 37, 48)
            hit(o + 1.5, 37, 42)
            hit(o + 2.5, 37, 45)
            if bar % 4 != 3:
                hit(o + 3.5, 37, 38)

        if crash:
            hit(o + 0.0, 49, 88)

    # Intro: no drums
    # (bars 0-7 silent)

    # Hook A: full Chicago drill
    for bar in range(HOOKA_S, HOOKA_E):
        idx = bar - HOOKA_S
        drill_bar(bar, full=True, crash=(idx % 8 == 0))

    # Verse: drums + 808, sparser hats
    for bar in range(VERSE_S, VERSE_E):
        idx = bar - VERSE_S
        drill_bar(bar, full=False, crash=(idx == 0))

    # Bridge: no drums (strings + lead only)

    # Hook B: full arrangement
    for bar in range(HOOKB_S, HOOKB_E):
        idx = bar - HOOKB_S
        drill_bar(bar, full=True, crash=(idx % 8 == 0))

    return part


# ============================================================================
# 808 BASS — short-medium, bouncy, mirrors kick
# ============================================================================

def create_808():
    part = stream.Part()
    part.partName = '808 Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def bass_bar(bar, vel=94, ghost=True, octave_accent=False):
        o    = bb(bar)
        root = BASS_ROOTS[bar % 4]

        # Main hit on beat 1 (duration 2 beats = short-medium)
        n1 = note.Note(root, quarterLength=2.0)
        n1.volume.velocity = vel
        part.insert(o, n1)

        # Ghost bounce on beat 3 (short, 0.75 beats)
        if ghost:
            n2 = note.Note(root, quarterLength=0.75)
            n2.volume.velocity = int(vel * 0.35)
            part.insert(o + 2.0, n2)

        # Octave jump accent: half-step before beat 1 (every 4th bar)
        if octave_accent:
            oct_root = note.Note(root)
            oct_root.octave += 1
            na = note.Note(oct_root.nameWithOctave, quarterLength=0.25)
            na.volume.velocity = int(vel * 0.65)
            part.insert(o + 3.75, na)

    # Hook A
    for bar in range(HOOKA_S, HOOKA_E):
        idx = bar - HOOKA_S
        bass_bar(bar, vel=96, ghost=True,
                 octave_accent=((idx + 1) % 4 == 0))

    # Verse
    for bar in range(VERSE_S, VERSE_E):
        idx = bar - VERSE_S
        bass_bar(bar, vel=88, ghost=(bar % 2 == 1),
                 octave_accent=((idx + 1) % 4 == 0))

    # Bridge: no 808

    # Hook B
    for bar in range(HOOKB_S, HOOKB_E):
        idx = bar - HOOKB_S
        bass_bar(bar, vel=98, ghost=True,
                 octave_accent=((idx + 1) % 4 == 0))

    return part


# ============================================================================
# STRINGS — sustained minor triads (reference voicings, octave 4)
# ============================================================================

def create_strings():
    part = stream.Part()
    part.partName = 'Strings'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def string_bar(bar, vel=50, dur=4.0):
        o = bb(bar)
        ch = chord.Chord(STRING_CHORDS[bar % 4], quarterLength=dur)
        ch.volume.velocity = vel
        part.insert(o, ch)

    # Intro: strings present, quiet and ominous
    for bar in range(INTRO_S, INTRO_E):
        string_bar(bar, vel=36, dur=4.0)

    # Hook A: strings + everything
    for bar in range(HOOKA_S, HOOKA_E):
        string_bar(bar, vel=44, dur=4.0)

    # Verse: no strings (sparse)

    # Bridge: strings return, dark and prominent
    for bar in range(BRIDGE_S, BRIDGE_E):
        string_bar(bar, vel=48, dur=4.0)

    # Hook B
    for bar in range(HOOKB_S, HOOKB_E):
        string_bar(bar, vel=46, dur=4.0)

    return part


# ============================================================================
# PIANO — bouncy repetitive melody, emphasis on G4 (the fifth)
# ============================================================================
# C minor pentatonic: C4 Eb4 F4 G4 Bb4
# 8th note rhythm, repetitive pattern, velocity variation for bounce

# Piano melody — dark, sparse, lower register (C3-Bb4)
# Longer notes with rests for a brooding, menacing feel at 120 BPM
# Heavy use of Eb (minor 3rd) and Ab (minor 6th) for darkness
PIANO_PHRASES = {
    # Cm bar: Eb--C--G--rest  (slow, brooding)
    0: [('Eb4', 1.0), ('C4',  0.5), (None,  0.5),
        ('G3',  1.0), (None,  1.0)],
    # Ab bar: Ab--Eb--C--rest  (descending, heavy)
    1: [('Ab3', 1.0), ('Eb4', 0.5), (None,  0.5),
        ('C4',  1.0), (None,  1.0)],
    # Eb bar: Bb--G--Eb--rest  (descending triad)
    2: [('Bb3', 1.0), ('G3',  0.5), (None,  0.5),
        ('Eb4', 1.0), (None,  1.0)],
    # Bb bar: F--D--Bb--rest  (dark resolution)
    3: [('F4',  1.0), ('D4',  0.5), (None,  0.5),
        ('Bb3', 1.0), (None,  1.0)],
}


def create_piano():
    part = stream.Part()
    part.partName = 'Piano'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def piano_bar(bar, vel_base=45, mute_last_beat=False):
        o = bb(bar)
        phrase = PIANO_PHRASES[bar % 4]
        beat = 0.0
        for i, (pitch, dur) in enumerate(phrase):
            if mute_last_beat and beat >= 3.0:
                break
            vel = vel_base + random.randint(-4, 4)
            # Accent G4 (fifth) and Eb4 (minor 3rd) for minor character
            if pitch in ('G4', 'Eb4'):
                vel = min(127, vel + 8)
            nd = note.Note(pitch, quarterLength=dur)
            nd.volume.velocity = vel
            part.insert(o + beat, nd)
            beat += dur

    # Intro: piano present (cut melody on bar 7 per tutorial)
    for bar in range(INTRO_S, INTRO_E):
        piano_bar(bar, vel_base=40,
                  mute_last_beat=(bar == INTRO_E - 1))

    # Hook A
    for bar in range(HOOKA_S, HOOKA_E):
        piano_bar(bar, vel_base=48)

    # Verse: piano only (prominent)
    for bar in range(VERSE_S, VERSE_E):
        piano_bar(bar, vel_base=44)

    # Bridge: no piano

    # Hook B
    for bar in range(HOOKB_S, HOOKB_E):
        piano_bar(bar, vel_base=50)

    return part


# ============================================================================
# LEAD SYNTH — dark pluck, sparse, hook sections only
# ============================================================================
# C minor scale notes: C5 Eb5 G5 Bb4
# Long notes (1-2 beats), enters only on hook sections

LEAD_PHRASES = [
    # 4-bar phrase: very sparse, low register, menacing
    # Eb4 (minor 3rd) and Ab4 (minor 6th) for maximum darkness
    [('Eb4', 2.5), (None, 3.5), ('C4',  2.0), (None, 4.0),
     ('Ab3', 3.0), (None, 3.0), ('Eb4', 2.0), (None, 2.0)],
]


def create_lead():
    part = stream.Part()
    part.partName = 'Lead Synth'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_lead_phrase(start_bar, vel_base=38):
        o = bb(start_bar)
        beat = 0.0
        phrase = LEAD_PHRASES[0]
        for pitch, dur in phrase:
            if beat >= 16.0:  # 4 bars worth
                break
            if pitch is not None:
                nd = note.Note(pitch, quarterLength=dur)
                nd.volume.velocity = vel_base + random.randint(-3, 3)
                part.insert(o + beat, nd)
            beat += dur

    # Hook A: lead enters
    for cycle in range(HOOKA_S, HOOKA_E, 4):
        write_lead_phrase(cycle, vel_base=40)

    # Bridge: lead present (prominent, dark)
    for cycle in range(BRIDGE_S, BRIDGE_E, 4):
        write_lead_phrase(cycle, vel_base=44)

    # Hook B: lead returns
    for cycle in range(HOOKB_S, HOOKB_E, 4):
        write_lead_phrase(cycle, vel_base=42)

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
    '808':    38,   # Synth Bass 1
    'bass':   38,
    'string': 48,   # String Ensemble 1
    'piano':  0,    # Acoustic Grand Piano
    'lead':   80,   # Lead 1 (square)
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
    print(f'  Key: C minor  |  BPM: {BPM}  |  64 bars')

    parts = {
        'drums':   create_drums(),
        '808':     create_808(),
        'strings': create_strings(),
        'piano':   create_piano(),
        'lead':    create_lead(),
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
