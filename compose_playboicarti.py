"""
Playboi Carti / Opium Type Beat — Compose
Key: Ab minor | BPM: 150 | 64 bars (~1:42)

Research (Path B — YouTube Tutorial Transcripts):
  - QWfXApO3gAA: Miss The Rage style — 156 BPM, supersaw synths (7+8 voices,
    detuned), reversed samples, overdrive, clap on 3&7
  - cey0wKnWtS0: Carti beats — 144 BPM, G# minor, half-step tension chords
    (B&A#, D#&E), creepy/eerie, cowbell, multiple 808 patterns, open hats
  - bt-60g3S7WU: Opium beats — "comically repetitive" melodies, minor + half-step
    dissonance, 808 goes UP then falls, triplet hats, cowbell, open hat velocity
    bounce, 2-bar max patterns

Reference voicings:
  chordprogressions/GitHub Free Progressions/Ab/Ab Minor/ — triads Ab4 B4 Eb5
  chordprogressions/More Genres/Trap Dark.mid — Ab4 B4 Eb4 ↔ C#4 E4 Ab4

Chord Progression (2-bar alternation, Ab minor — dissonant half-step clusters):
  Bars 0-1: Abm(add2)  (Ab4 Bb4 B4 Eb5)  — Bb↔B cluster
  Bars 2-3: Db(add2)   (C#4 Eb4 E4 Ab4)  — Eb↔E cluster

6 Sound Layers:
  1. Drums       — kick+clap+hh(2 patterns)+cowbell+open hat+crash
  2. 808 Bass    — two alternating patterns, root→UP→root
  3. Supersaw Pad— dissonant sustained chords (7-voice detuned saw)
  4. Lead Synth  — comically repetitive 2-bar melody
  5. Reversed FX — reversed pad swells before transitions
  6. Atmosphere  — noise/texture layer (rendered only, not MIDI)

Song Structure:
  Intro     bars  0– 7: pad + reversed FX only (melody enters bar 4)
  Hook A    bars  8–23: full arrangement (triplet hats)
  Verse     bars 24–39: drums(sparse) + 808 + lead only
  Bridge    bars 40–47: pad + lead (no drums, contrast)
  Hook B    bars 48–63: full arrangement (intensified)

Kit: Obie Modern Trap
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

BEAT_NAME  = 'PlayboiCarti'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/PlayboiCarti_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 150
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


def shift_pitch(pitch_str, oct_offset):
    if pitch_str is None or oct_offset == 0:
        return pitch_str
    n = note.Note(pitch_str)
    n.octave += oct_offset
    return n.nameWithOctave


# Hook octave pattern per workflow: [base, base, +1, +1] within 16-bar hook
HOOK_OCT_PATTERN = [0, 0, 1, 1]


# ============================================================================
# CHORD TABLES — Ab minor, dissonant half-step clusters
# ============================================================================
# Voicings from chordprogressions/GitHub Free Progressions/Ab/Ab Minor/
# and More Genres/Trap Dark.mid — octave 4 register.
# Tutorial technique: stack adjacent half-step notes for evil/dissonant sound.
# Ab minor scale: Ab Bb B(Cb) C#(Db) Eb E(Fb) F#(Gb)
# Half-step tension pairs: Bb↔B and Eb↔E

# Pad/supersaw voicings — 4-note dissonant clusters
PAD_CHORDS = [
    ['Ab4', 'Bb4', 'B4',  'Eb5'],   # Abm(add2) — Bb↔B cluster (bars 0-1)
    ['C#4', 'Eb4', 'E4',  'Ab4'],   # Db(add2)  — Eb↔E cluster (bars 2-3)
]

# 808 roots — follows chord roots, goes UP then falls back per tutorial
BASS_ROOTS_A = ['Ab1', 'B1',  'Ab1']   # Pattern A: root→up→root (bars 0-1)
BASS_ROOTS_B = ['C#2', 'Eb2', 'C#2']   # Pattern B: root→up→root (bars 2-3)


# ============================================================================
# LEAD MELODY — derived from chordprogressions reference voicings
# ============================================================================
# Source: Trap Dark.mid Track 2 — descending Ab→B→Eb arpeggio motif with
#         octave drops. Ab Minor triads: Ab4 B4 Eb5 (i chord voicing).
# Tutorial technique: Eb↔E half-step dissonance for evil/creepy sound.
# Workflow: melodies quiet (vel 38-58), sparse, sit behind drums/bass.
# Tutorial: 2-bar max patterns, repetitive.
# LEAD TIMBRE RULE: saw+square dark pluck, vel 38-55. Never louder than
# drums/808. Do NOT use bright bell or sine.

# Phrase A (over Abm chord): Trap Dark descending arpeggio Ab→B→Eb
LEAD_PHRASE_A = [
    # 2 bars = 8 beats
    ('Ab5', 0.5),  ('B4',  0.25), ('Eb5', 0.75),  # descending arpeggio motif
    (None,  0.5),
    ('Ab5', 0.25), ('Eb5', 0.25), ('B4',  0.5),   # motif inverted rhythm
    (None,  1.0),                                    # breathe
    ('Eb5', 0.5),  ('Ab4', 0.5),                    # drop to low register
    (None,  0.5),  ('B4',  0.5),                    # return to mid
    ('Eb5', 1.0),  (None,  1.0),                    # sustain + rest
]

# Phrase B (over Db chord): Eb↔E half-step dissonance from tutorials
LEAD_PHRASE_B = [
    # 2 bars = 8 beats
    ('E5',  0.5),  ('Eb5', 0.25), ('C#5', 0.75),  # Eb/E dissonance + C# chord tone
    (None,  0.5),
    ('E5',  0.25), ('C#5', 0.25), ('Ab4', 0.5),   # descending to root
    (None,  1.0),                                    # breathe
    ('C#5', 0.5),  ('E5',  0.5),                    # rising tension
    (None,  0.5),  ('Eb5', 0.5),                    # half-step down (dissonant)
    ('Ab4', 1.0),  (None,  1.0),                    # resolve to root
]


# ============================================================================
# DRUMS — Opium/rage trap pattern
# ============================================================================
# GM drum mapping: 36=kick 38=snare 39=clap 42=closed-HH 46=open-HH
#                  49=crash 56=cowbell 37=rim

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def drum_bar(bar, pattern='A', crash=False):
        """One bar of Opium-style drums.
        Pattern A = sparse (verse), Pattern B = full with triplet hats (hook).
        """
        o = bb(bar)

        # Kick: beat 1, ghost on beat 3.5
        hit(o + 0.0, 36, 100)
        hit(o + 3.5, 36, 65)

        # Clap on beats 2 and 4 (half-time feel)
        hit(o + 1.0, 39, 94)
        hit(o + 3.0, 39, 98)
        # Layer snare with clap per tutorial
        hit(o + 1.0, 38, 72)
        hit(o + 3.0, 38, 76)

        # Hi-hats
        if pattern == 'B':
            # Hook pattern: 8th notes beats 1-2, triplet rolls beats 3-4
            for i in range(4):
                hit(o + i * 0.5, 42, random.randint(40, 58))
            # Triplet rolls on beats 3-4 (signature Opium bounce)
            for triplet in range(6):
                t = 2.0 + triplet * (2.0 / 6.0)
                hit(o + t, 42, random.randint(35, 55))
        else:
            # Verse pattern: straight 8th notes, sparse
            for i in range(8):
                vel = 48 if i % 2 == 0 else 32
                hit(o + i * 0.5, 42, vel)

        # Cowbell — offbeat hits (tutorials 2+3 both specified cowbell)
        hit(o + 0.5, 56, 52)
        hit(o + 2.5, 56, 48)

        # Open hat — velocity variation for "signature bounce" (tutorial 3)
        oh_vel = random.randint(45, 65)
        hit(o + 1.5, 46, oh_vel)
        hit(o + 3.5, 46, oh_vel - 10)

        if crash:
            hit(o + 0.0, 49, 88)

    # Intro: no drums (bars 0-7)

    # Hook A: full with triplet hats
    for bar in range(HOOKA_S, HOOKA_E):
        idx = bar - HOOKA_S
        drum_bar(bar, pattern='B', crash=(idx % 8 == 0))

    # Verse: sparse hats
    for bar in range(VERSE_S, VERSE_E):
        idx = bar - VERSE_S
        drum_bar(bar, pattern='A', crash=(idx == 0))

    # Bridge: no drums (contrast)

    # Hook B: full, intensified
    for bar in range(HOOKB_S, HOOKB_E):
        idx = bar - HOOKB_S
        drum_bar(bar, pattern='B', crash=(idx % 8 == 0))

    return part


# ============================================================================
# 808 BASS — root→UP→root pattern, two alternating patterns
# ============================================================================

def create_808():
    part = stream.Part()
    part.partName = '808 Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def bass_2bar(start_bar, roots, vel=94):
        """2-bar 808 pattern: root(1.5) → up(1.0) → root(1.0) per bar."""
        o = bb(start_bar)
        # Bar 1: root beat 1 (1.5 dur), up beat 3 (1.0 dur), root beat 4 (1.0 dur)
        n1 = note.Note(roots[0], quarterLength=1.5)
        n1.volume.velocity = vel
        part.insert(o, n1)

        n2 = note.Note(roots[1], quarterLength=1.0)
        n2.volume.velocity = int(vel * 0.75)
        part.insert(o + 2.0, n2)

        n3 = note.Note(roots[2], quarterLength=1.0)
        n3.volume.velocity = int(vel * 0.60)
        part.insert(o + 3.0, n3)

        # Bar 2: root sustained (2.0 dur), ghost on beat 3 (0.5 dur)
        o2 = bb(start_bar + 1)
        n4 = note.Note(roots[0], quarterLength=2.0)
        n4.volume.velocity = vel
        part.insert(o2, n4)

        n5 = note.Note(roots[2], quarterLength=0.5)
        n5.volume.velocity = int(vel * 0.30)
        part.insert(o2 + 3.0, n5)

    def write_section(s_bar, e_bar, vel=94):
        for bar in range(s_bar, e_bar, 4):
            bass_2bar(bar,     BASS_ROOTS_A, vel=vel)
            bass_2bar(bar + 2, BASS_ROOTS_B, vel=vel)

    # Hook A
    write_section(HOOKA_S, HOOKA_E, vel=96)

    # Verse
    write_section(VERSE_S, VERSE_E, vel=88)

    # Bridge: no 808

    # Hook B
    write_section(HOOKB_S, HOOKB_E, vel=98)

    return part


# ============================================================================
# SUPERSAW PAD — dissonant sustained chords
# ============================================================================
# SUPERSAW TIMBRE RULE: 7 voices detuned saw (±12 cents), NOT triangle/sine.
# Slow attack 0.4s, release 2.5s. Low velocity (40-50).

def create_pad():
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def pad_section(s_bar, e_bar, vel=45):
        for bar in range(s_bar, e_bar, 4):
            # 2-bar chord A: Ab4 Bb4 B4 Eb5
            ch_a = chord.Chord(PAD_CHORDS[0], quarterLength=8.0)
            ch_a.volume.velocity = vel
            part.insert(bb(bar), ch_a)
            # 2-bar chord B: C#4 Eb4 E4 Ab4
            ch_b = chord.Chord(PAD_CHORDS[1], quarterLength=8.0)
            ch_b.volume.velocity = vel
            part.insert(bb(bar + 2), ch_b)

    # Intro: pad enters (creepy atmosphere)
    pad_section(INTRO_S, INTRO_E, vel=38)

    # Hook A: pad present
    pad_section(HOOKA_S, HOOKA_E, vel=46)

    # Verse: no pad (sparse feel)

    # Bridge: pad returns (prominent)
    pad_section(BRIDGE_S, BRIDGE_E, vel=50)

    # Hook B: pad present
    pad_section(HOOKB_S, HOOKB_E, vel=48)

    return part


# ============================================================================
# LEAD SYNTH — comically repetitive 2-bar melody
# ============================================================================
# LEAD TIMBRE RULE: saw+square dark pluck, LPF 2500Hz, vel 38-55.
# Never louder than drums/808. Do NOT use bright bell or sine.

def create_lead():
    part = stream.Part()
    part.partName = 'Lead Synth'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_lead_section(s_bar, e_bar, vel_base=42, is_hook=False):
        for bar in range(s_bar, e_bar, 4):
            # Hook octave pattern: [base, base, +1, +1] per 4-cycle within 16 bars
            if is_hook:
                cycle_idx = (bar - s_bar) // 4
                oct = HOOK_OCT_PATTERN[cycle_idx] if cycle_idx < len(HOOK_OCT_PATTERN) else 0
            else:
                oct = 0

            # Phrase A over bars 0-1 (Abm chord)
            o_a = bb(bar)
            beat = 0.0
            for pitch, dur in LEAD_PHRASE_A:
                if pitch is None:
                    part.insert(o_a + beat, note.Rest(quarterLength=dur))
                else:
                    p = shift_pitch(pitch, oct)
                    nd = note.Note(p, quarterLength=dur)
                    nd.volume.velocity = vel_base + random.randint(-4, 4)
                    part.insert(o_a + beat, nd)
                beat += dur

            # Phrase B over bars 2-3 (Db chord)
            o_b = bb(bar + 2)
            beat = 0.0
            for pitch, dur in LEAD_PHRASE_B:
                if pitch is None:
                    part.insert(o_b + beat, note.Rest(quarterLength=dur))
                else:
                    p = shift_pitch(pitch, oct)
                    nd = note.Note(p, quarterLength=dur)
                    nd.volume.velocity = vel_base + random.randint(-4, 4)
                    part.insert(o_b + beat, nd)
                beat += dur

    # Intro: melody enters bar 4
    write_lead_section(4, INTRO_E, vel_base=36)

    # Hook A: full melody
    write_lead_section(HOOKA_S, HOOKA_E, vel_base=44, is_hook=True)

    # Verse: lead present (only melodic element)
    write_lead_section(VERSE_S, VERSE_E, vel_base=40)

    # Bridge: lead present (prominent over pad)
    write_lead_section(BRIDGE_S, BRIDGE_E, vel_base=46)

    # Hook B: full melody
    write_lead_section(HOOKB_S, HOOKB_E, vel_base=46, is_hook=True)

    return part


# ============================================================================
# REVERSED FX — pad swells before section transitions (MIDI placeholder)
# ============================================================================
# Actual reversed audio generated in render. MIDI marks transition points.

def create_reversed_fx():
    part = stream.Part()
    part.partName = 'Reversed FX'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Place marker notes before each transition
    transitions = [HOOKA_S, VERSE_S, BRIDGE_S, HOOKB_S]
    for target_bar in transitions:
        # 2-bar reverse swell ending at target bar
        start = bb(target_bar - 2)
        nd = note.Note('Ab4', quarterLength=8.0)
        nd.volume.velocity = 50
        part.insert(start, nd)

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
    'pad':    89,   # Pad 2 (warm)
    'lead':   81,   # Lead 2 (sawtooth) — dark pluck
    'reverse': 98,  # FX 3 (crystal)
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
    print(f'  Key: Ab minor  |  BPM: {BPM}  |  64 bars')

    parts = {
        'drums':       create_drums(),
        '808':         create_808(),
        'pad':         create_pad(),
        'lead':        create_lead(),
        'reversed_fx': create_reversed_fx(),
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
