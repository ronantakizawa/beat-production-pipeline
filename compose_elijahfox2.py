"""
Elijah Fox Song 2 — Solo Piano Compose
Key: D major | BPM: 68 | 28 bars (~1:38)

Style: Extended jazz chords (maj9, #11, min9, 13, min11, sus4),
two-hand ascending arpeggios, D major pentatonic fills, sparse LH,
rubato feel with fermatas at phrase boundaries.

Reference voicings: chordprogressions/GitHub Free Progressions/D/D Ninths/D Ninths.mid

Chord Progression:
  Section A  (bars 0-7,   2x4): Dmaj9 -> Gmaj7#11 -> Bmin9 -> A13
  Section B  (bars 8-15,  2x4): Emin11 -> F#min9 -> Gmaj9 -> A7sus4
  Section A' (bars 16-23, 2x4): Same chords as A, register variation
  Coda       (bars 24-27):      Gmaj7#11 (2 bars) -> Dmaj9 fermata (2 bars)

2 Parts: Right Hand, Left Hand — both GM program 0 (Acoustic Grand Piano)
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

BEAT_NAME  = 'ElijahFox2'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/ElijahFox_Piano2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 68
BPB = 4  # beats per bar


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat."""
    return float(bar * BPB + beat)


# ============================================================================
# CHORD VOICINGS — explicit pitches, both hands
# ============================================================================
# RH: 5-note voicings in D4-F#5 range (matching D Ninths reference density)
# LH: root + fifth in octaves 1-2

SECTION_A_CHORDS = [
    # Dmaj9: I
    {'lh': ['D2', 'A2'], 'rh': ['D4', 'F#4', 'A4', 'C#5', 'E5']},
    # Gmaj7#11: IV
    {'lh': ['G1', 'D2'], 'rh': ['F#4', 'G4', 'A4', 'B4', 'D5']},
    # Bmin9: vi
    {'lh': ['B1', 'F#2'], 'rh': ['A4', 'B4', 'D5', 'E5', 'F#5']},
    # A13: V
    {'lh': ['A1', 'E2'], 'rh': ['G4', 'A4', 'C#5', 'E5', 'F#5']},
]

SECTION_B_CHORDS = [
    # Emin11: ii
    {'lh': ['E2', 'B2'], 'rh': ['E4', 'G4', 'A4', 'B4', 'D5']},
    # F#min9: iii
    {'lh': ['F#1', 'C#2'], 'rh': ['F#4', 'A4', 'B4', 'C#5', 'E5']},
    # Gmaj9: IV
    {'lh': ['G1', 'D2'], 'rh': ['G4', 'A4', 'B4', 'D5', 'F#5']},
    # A7sus4: V
    {'lh': ['A1', 'E2'], 'rh': ['F#4', 'G4', 'A4', 'D5', 'E5']},
]

# A' — register variation: brighter top voicings
SECTION_AP_CHORDS = [
    # Dmaj9 high: omit low D4, add D5
    {'lh': ['D2', 'A2'], 'rh': ['F#4', 'A4', 'C#5', 'D5', 'E5']},
    # Gmaj7#11 high: F#4 -> F#5
    {'lh': ['G1', 'D2'], 'rh': ['G4', 'A4', 'B4', 'D5', 'F#5']},
    # Bmin9: same (already high)
    {'lh': ['B1', 'F#2'], 'rh': ['A4', 'B4', 'D5', 'E5', 'F#5']},
    # A13: same (already high)
    {'lh': ['A1', 'E2'], 'rh': ['G4', 'A4', 'C#5', 'E5', 'F#5']},
]

# Coda: Gmaj7#11 (2 bars) -> Dmaj9 fermata (2 bars)
CODA_CHORDS = [
    {'lh': ['G1', 'D2'], 'rh': ['F#4', 'G4', 'A4', 'B4', 'D5']},
    {'lh': ['G1', 'D2'], 'rh': ['F#4', 'G4', 'A4', 'B4', 'D5']},
    {'lh': ['D2', 'A2'], 'rh': ['D4', 'F#4', 'A4', 'C#5', 'E5']},
    {'lh': ['D2', 'A2'], 'rh': ['D4', 'F#4', 'A4', 'C#5', 'E5']},
]


# ============================================================================
# ARPEGGIO PATTERNS — 6-note ascending patterns through 5 RH chord tones
# ============================================================================
# Each pattern: list of (chord_tone_index, octave_shift).
# Indices refer to the 5 RH chord tones.

ARP_PATTERNS = [
    # Straight ascending + wrap
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1)],
    # Skip pattern: 0 2 4 3 1 0
    [(0, 0), (2, 0), (4, 0), (3, 0), (1, 0), (0, 0)],
    # Offset: 1 0 2 4 3 1+1
    [(1, 0), (0, 0), (2, 0), (4, 0), (3, 0), (1, 1)],
]

# D major pentatonic fills
PENTA_FILLS = {
    'A':      ['D4', 'E4', 'F#4', 'A4', 'B4'],
    'B':      ['D4', 'E4', 'F#4', 'A4', 'B4'],
    'A_high': ['D5', 'E5', 'F#5', 'A4', 'B4'],
    'coda':   ['D4', 'E4', 'F#4'],
}

# Fermata bars — extend last note
FERMATA_BARS = {7, 15, 23, 27}

# Breathing room: melody-only (no arp) and arp-only (no fill)
MELODY_ONLY_BARS = {1, 5, 9, 13, 17, 21}
ARP_ONLY_BARS = {0, 4, 8, 12, 16, 20, 24}


def get_chord_for_bar(bar):
    """Return the chord voicing dict for a given bar."""
    if bar < 8:
        return SECTION_A_CHORDS[bar % 4]
    elif bar < 16:
        return SECTION_B_CHORDS[(bar - 8) % 4]
    elif bar < 24:
        return SECTION_AP_CHORDS[(bar - 16) % 4]
    else:
        return CODA_CHORDS[(bar - 24) % 4]


def get_penta_pool(bar):
    """Return pentatonic fill pool for current section."""
    if bar < 8:
        return PENTA_FILLS['A']
    elif bar < 16:
        return PENTA_FILLS['B']
    elif bar < 24:
        return PENTA_FILLS['A_high']
    else:
        return PENTA_FILLS['coda']


# ============================================================================
# RIGHT HAND
# ============================================================================

def create_right_hand():
    part = stream.Part()
    part.partName = 'Right Hand'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    for bar in range(28):
        ch = get_chord_for_bar(bar)
        rh_tones = ch['rh']
        o = bb(bar)
        is_fermata = bar in FERMATA_BARS
        is_melody_only = bar in MELODY_ONLY_BARS
        is_arp_only = bar in ARP_ONLY_BARS

        # Coda bars 26-27: sustained Dmaj9 chord, no movement
        if bar >= 26:
            c = chord.Chord(rh_tones, quarterLength=4.0)
            c.volume.velocity = 38
            part.insert(o, c)
            continue

        beat = 0.0

        # --- Arpeggio section (first ~1.5 beats, 16th notes) ---
        if not is_melody_only:
            pat = ARP_PATTERNS[bar % 3]
            note_dur = 0.25  # 16th notes
            for tone_idx, oct_shift in pat:
                idx = min(tone_idx, len(rh_tones) - 1)
                p = note.Note(rh_tones[idx])
                if oct_shift != 0:
                    p.octave += oct_shift
                nd = note.Note(p.nameWithOctave, quarterLength=note_dur)
                vel = random.randint(40, 55)
                nd.volume.velocity = vel
                part.insert(o + beat, nd)
                beat += note_dur
            # Brief rest after arpeggio
            beat += 0.25

        # --- Pentatonic fill (2-4 notes) ---
        if not is_arp_only and beat < 3.0:
            penta = get_penta_pool(bar)
            n_fill = random.randint(2, 4)
            fill_notes = random.sample(penta, min(n_fill, len(penta)))
            fill_notes.sort(key=lambda p: note.Note(p).pitch.midi)
            for fp in fill_notes:
                dur = random.choice([0.5, 0.75, 1.0])
                if beat + dur > 3.5:
                    dur = max(0.25, 3.5 - beat)
                nd = note.Note(fp, quarterLength=dur)
                nd.volume.velocity = random.randint(35, 50)
                part.insert(o + beat, nd)
                beat += dur
                if beat >= 3.5:
                    break

        # --- Fermata: sustained chord tone at end of bar ---
        if is_fermata and beat > 0:
            p = rh_tones[-1]
            nd = note.Note(p, quarterLength=6.0)
            nd.volume.velocity = 35
            part.insert(o + beat, nd)

    return part


# ============================================================================
# LEFT HAND
# ============================================================================

def create_left_hand():
    part = stream.Part()
    part.partName = 'Left Hand'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    for bar in range(28):
        ch = get_chord_for_bar(bar)
        lh_tones = ch['lh']
        o = bb(bar)
        is_fermata = bar in FERMATA_BARS

        # Beat 1: always play bass root
        dur1 = 6.0 if is_fermata else random.choice([2.0, 3.0, 4.0])
        n1 = note.Note(lh_tones[0], quarterLength=dur1)
        n1.volume.velocity = random.randint(40, 55)
        part.insert(o, n1)

        # Beat 3: sometimes play fifth (60% chance)
        if not is_fermata and len(lh_tones) > 1 and random.random() < 0.6:
            n2 = note.Note(lh_tones[1], quarterLength=random.choice([1.0, 2.0]))
            n2.volume.velocity = random.randint(38, 48)
            part.insert(o + 2.0, n2)

        # Beat 4: occasional passing tone walk (25% chance)
        if not is_fermata and random.random() < 0.25:
            root = note.Note(lh_tones[0])
            walk = note.Note(root.pitch.midi - 2, quarterLength=0.5)
            walk.volume.velocity = random.randint(30, 40)
            part.insert(o + 3.5, walk)

    return part


# ============================================================================
# MIDI HELPERS — from skeleton
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
        # Both hands are piano — GM program 0
        insert_program(track, 0)


def save(score, filename):
    path = os.path.join(OUTPUT_DIR, filename)
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
    print(f'Composing {BEAT_NAME} ...')
    print(f'  Key: D major  |  BPM: {BPM}  |  28 bars')

    rh = create_right_hand()
    lh = create_left_hand()

    # Save individual stems
    print('Saving stems ...')
    save(solo(rh), f'{BEAT_NAME}_rh.mid')
    save(solo(lh), f'{BEAT_NAME}_lh.mid')

    # Save full arrangement
    print('\nSaving full arrangement ...')
    full = stream.Score()
    full.append(rh)
    full.append(lh)
    save(full, f'{BEAT_NAME}_FULL.mid')

    print(f'\nDone! MIDI files saved to:\n  {OUTPUT_DIR}')
