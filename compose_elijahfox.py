"""
Elijah Fox — Solo Piano Compose
Key: A major | BPM: 72 | 32 bars (~2:08)

Style: Extended jazz chords (maj9, 13#11, min11, sus, min6),
two-hand ascending arpeggios, pentatonic fills, sparse LH,
rubato feel with fermatas at phrase boundaries.

Chord Progression:
  Section A  (bars 0-11, 3x4): Amaj9 → Emin6/G → Dmaj7#11 → G13#11
  Section B  (bars 12-19, 2x4): F#min11 → Dmaj9 → Bmin9 → E7sus4
  Section A' (bars 20-27, 2x4): Amaj9 (high) → Emin6/G → Dmaj7#11 → G13#11
  Coda       (bars 28-31):      Dmaj7#11 → Amaj9 fermata

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

BEAT_NAME  = 'ElijahFox'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/ElijahFox_Piano'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 72
BPB = 4  # beats per bar


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat."""
    return float(bar * BPB + beat)


# ============================================================================
# CHORD VOICINGS — explicit pitches, both hands
# ============================================================================
# Each chord: { 'rh': [pitches high], 'lh': [pitches low] }

SECTION_A_CHORDS = [
    # Amaj9: A1 E2 | C#4 E4 G#4 B4
    {'lh': ['A1', 'E2'], 'rh': ['C#4', 'E4', 'G#4', 'B4']},
    # Emin6/G: G1 E2 | E3 G3 B3 C#4
    {'lh': ['G1', 'E2'], 'rh': ['E3', 'G3', 'B3', 'C#4']},
    # Dmaj7#11: D1 A2 | F#3 A3 C#4 G#4
    {'lh': ['D1', 'A2'], 'rh': ['F#3', 'A3', 'C#4', 'G#4']},
    # G13#11: G1 F2 | B3 E4 F#4 A4
    {'lh': ['G1', 'F2'], 'rh': ['B3', 'E4', 'F#4', 'A4']},
]

SECTION_B_CHORDS = [
    # F#min11: F#1 C#2 | A3 C#4 E4 F#4  (vi)
    {'lh': ['F#1', 'C#2'], 'rh': ['A3', 'C#4', 'E4', 'F#4']},
    # Dmaj9: D1 A2 | C#4 E4 F#4 A4  (IV)
    {'lh': ['D1', 'A2'], 'rh': ['C#4', 'E4', 'F#4', 'A4']},
    # Bmin9: B0 F#2 | B3 D4 F#4 A4  (ii)
    {'lh': ['B0', 'F#2'], 'rh': ['B3', 'D4', 'F#4', 'A4']},
    # E9sus4: E1 B2 | A3 D4 F#4 G#4  (V)
    {'lh': ['E1', 'B2'], 'rh': ['A3', 'D4', 'F#4', 'G#4']},
]

# A' — same register as A (no octave shift)
SECTION_AP_CHORDS = list(SECTION_A_CHORDS)

CODA_CHORDS = [
    SECTION_A_CHORDS[2],  # Dmaj7#11
    SECTION_A_CHORDS[0],  # Amaj9 fermata
    SECTION_A_CHORDS[0],  # held
    SECTION_A_CHORDS[0],  # held
]


# ============================================================================
# ARPEGGIO PATTERNS — 6-note ascending patterns through RH chord tones
# ============================================================================
# Each pattern is a list of (chord_tone_index, octave_shift) pairs.
# Indices refer to the 4 RH chord tones.

ARP_PATTERNS = [
    # Straight ascending: 0 1 2 3 0+1 1+1
    [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)],
    # Mirror: 0 1 2 / 2 1 0
    [(0, 0), (1, 0), (2, 0), (2, 0), (1, 0), (0, 0)],
    # Offset: 1 2 0 / 3 0+1 2
    [(1, 0), (2, 0), (0, 0), (3, 0), (0, 1), (2, 0)],
]

# Pentatonic fill notes (A major pentatonic, lower octave)
PENTA_FILLS = {
    'A': ['A3', 'B3', 'C#4', 'E4', 'F#4'],
    'A_high': ['A3', 'B3', 'C#4', 'E4', 'F#4'],
    'B': ['F#3', 'A3', 'B3', 'C#4', 'E4'],
}

# Fermata bars — extend last note
FERMATA_BARS = {3, 7, 11, 19, 27, 31}

# Bars that are melody-only (no arpeggio) for breathing room
MELODY_ONLY_BARS = {1, 5, 9, 14, 17, 22, 25}

# Bars that are arpeggio-only (no fill)
ARP_ONLY_BARS = {0, 4, 8, 12, 16, 20, 24, 28}


def get_chord_for_bar(bar):
    """Return the chord voicing dict for a given bar."""
    if bar < 12:
        return SECTION_A_CHORDS[bar % 4]
    elif bar < 20:
        return SECTION_B_CHORDS[(bar - 12) % 4]
    elif bar < 28:
        return SECTION_AP_CHORDS[(bar - 20) % 4]
    else:
        return CODA_CHORDS[(bar - 28) % 4]


def get_penta_pool(bar):
    """Return pentatonic fill pool for current section."""
    if bar < 12:
        return PENTA_FILLS['A']
    elif bar < 20:
        return PENTA_FILLS['B']
    elif bar < 28:
        return PENTA_FILLS['A_high']
    else:
        return PENTA_FILLS['A']


# ============================================================================
# RIGHT HAND
# ============================================================================

def create_right_hand():
    part = stream.Part()
    part.partName = 'Right Hand'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    for bar in range(32):
        ch = get_chord_for_bar(bar)
        rh_tones = ch['rh']
        o = bb(bar)
        is_fermata = bar in FERMATA_BARS
        is_melody_only = bar in MELODY_ONLY_BARS
        is_arp_only = bar in ARP_ONLY_BARS

        # Coda bars 29-31: sustained chord, no movement
        if bar >= 29:
            c = chord.Chord(rh_tones, quarterLength=4.0)
            c.volume.velocity = 38
            part.insert(o, c)
            continue

        beat = 0.0

        # --- Arpeggio section (first ~1.5 beats) ---
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

        # --- Fermata: extend last note ---
        if is_fermata and beat > 0:
            # Add a sustained chord tone at end of bar
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

    for bar in range(32):
        ch = get_chord_for_bar(bar)
        lh_tones = ch['lh']
        o = bb(bar)
        is_fermata = bar in FERMATA_BARS

        # Beat 1: always play bass root (first LH tone)
        dur1 = 4.0 if is_fermata else random.choice([2.0, 3.0, 4.0])
        if is_fermata:
            dur1 = 6.0  # fermata extension
        n1 = note.Note(lh_tones[0], quarterLength=dur1)
        n1.volume.velocity = random.randint(40, 55)
        part.insert(o, n1)

        # Beat 3: sometimes play second LH tone (fifth/octave)
        if not is_fermata and len(lh_tones) > 1 and random.random() < 0.6:
            n2 = note.Note(lh_tones[1], quarterLength=random.choice([1.0, 2.0]))
            n2.volume.velocity = random.randint(38, 48)
            part.insert(o + 2.0, n2)

        # Beat 4: occasional passing tone walk
        if not is_fermata and random.random() < 0.25:
            # Walk down a step from bass root
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
    print(f'  Key: A major  |  BPM: {BPM}  |  32 bars')

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
