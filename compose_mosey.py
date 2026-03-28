"""
Lil Mosey Type Beat -- MIDI Composition
Bright, happy, soulful melodic trap.

Layers:
  1. Flute Solo  -- main melody, chord tones + passing notes, C5-C6
  2. Harp        -- 16th note arp through chord tones
  3. Grand Piano -- sustained chord voicings, quiet
  4. Glockenspiel-- bell accents on hook downbeats
  5. Drums       -- 16th rolling hats, clap on 2&4, punchy kick
  6. 808 Bass    -- tuned to chord roots, tight ADSR

Chord Progressions:
  Key C: F major -> A minor -> E minor -> F major (bright)
  Key G: G major -> E minor -> C major -> D major (bouncy)

Song Structure (80 bars):
  Intro     0- 7: piano only, quiet
  Hook1     8-23: all layers (flute enters bar 8, harp bar 12)
  Verse    24-39: drums + 808 + harp + piano (flute sparse)
  Hook2    40-55: all layers
  Bridge   56-59: flute + glockenspiel only (4 bars max)
  Hook3    60-75: all layers
  Outro    76-79: piano + harp fade

Usage:
  python compose_mosey.py --name "Mosey_Happy" --key C --bpm 140
  python compose_mosey.py --name "Mosey_Bounce" --key G --bpm 138
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

OUTPUT_DIR = '/Users/ronantakizawa/Documents/LilMosey_Beat'

BPB = 4  # beats per bar

INTRO_S,  INTRO_E  =  0,  8
HOOK1_S,  HOOK1_E  =  8, 24
VERSE_S,  VERSE_E  = 24, 40
HOOK2_S,  HOOK2_E  = 40, 56
BRIDGE_S, BRIDGE_E = 56, 60
HOOK3_S,  HOOK3_E  = 60, 76
OUTRO_S,  OUTRO_E  = 76, 80


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat."""
    return float(bar * BPB + beat)


# ============================================================================
# CHORD PROGRESSIONS & VOICINGS
# ============================================================================

# Key of C: Fmaj7 -> Am7 -> Em7 -> Dm7 (bright, soulful — 7th chords throughout)
# Inspired by Blueberry Faygo's ii-V-I-VI7 jazz-influenced progression.
# All voicings use 7ths for warmth. Voice leading: move minimal notes between chords.
CHORDS_C = {
    'piano': [
        ['F3', 'A3', 'C4', 'E4'],     # Fmaj7  (IV)
        ['A3', 'C4', 'E4', 'G4'],     # Am7    (vi)
        ['E3', 'G3', 'B3', 'D4'],     # Em7    (iii)
        ['D3', 'F3', 'A3', 'C4'],     # Dm7    (ii) — resolves back to Fmaj7
    ],
    'harp_tones': [
        ['F4', 'A4', 'C5', 'E5', 'F5'],   # Fmaj7 arp (includes 7th)
        ['A4', 'C5', 'E5', 'G5', 'A5'],   # Am7 arp
        ['E4', 'G4', 'B4', 'D5', 'E5'],   # Em7 arp
        ['D4', 'F4', 'A4', 'C5', 'D5'],   # Dm7 arp
    ],
    # Melody tones: chord tones + passing tones from C major scale
    # Each list: [root_5, 3rd_5, 5th_5, 7th, passing_below_root, passing_above_3rd]
    'melody_tones': [
        ['F5', 'A5', 'C5', 'E5', 'E5', 'G5'],   # Fmaj7: F A C E + pass E G
        ['A4', 'C5', 'E5', 'G5', 'G4', 'D5'],   # Am7:   A C E G + pass G D
        ['E5', 'G5', 'B4', 'D5', 'D5', 'F5'],   # Em7:   E G B D + pass D F
        ['D5', 'F5', 'A5', 'C5', 'C5', 'E5'],   # Dm7:   D F A C + pass C E
    ],
    'bass_roots': ['F2', 'A1', 'E2', 'D2'],
    'glock_accents': ['E6', 'G5', 'D6', 'C6'],  # 7th of each chord for shimmer
}

# Key of G: Gmaj7 -> Em7 -> Cmaj7 -> D7 (bouncy, happy — dominant 7th on V for tension)
CHORDS_G = {
    'piano': [
        ['G3', 'B3', 'D4', 'F#4'],    # Gmaj7  (I)
        ['E3', 'G3', 'B3', 'D4'],     # Em7    (vi)
        ['C4', 'E4', 'G4', 'B4'],     # Cmaj7  (IV)
        ['D4', 'F#4', 'A4', 'C5'],    # D7     (V7) — dominant tension resolves to I
    ],
    'harp_tones': [
        ['G4', 'B4', 'D5', 'F#5', 'G5'],  # Gmaj7 arp
        ['E4', 'G4', 'B4', 'D5', 'E5'],   # Em7 arp
        ['C5', 'E5', 'G5', 'B5', 'C6'],   # Cmaj7 arp
        ['D5', 'F#5', 'A5', 'C6', 'D6'],  # D7 arp
    ],
    'melody_tones': [
        ['G5', 'B5', 'D5', 'F#5', 'F#5', 'A5'],  # Gmaj7 + pass
        ['E5', 'G5', 'B4', 'D5', 'D5', 'F#5'],   # Em7 + pass
        ['C5', 'E5', 'G5', 'B5', 'B4', 'D5'],    # Cmaj7 + pass
        ['D5', 'F#5', 'A5', 'C6', 'C5', 'E5'],   # D7 + pass
    ],
    'bass_roots': ['G2', 'E2', 'C2', 'D2'],
    'glock_accents': ['F#5', 'D6', 'B5', 'C6'],  # 7th of each chord
}

PROGRESSION = {'C': CHORDS_C, 'G': CHORDS_G}


# ============================================================================
# MARIMBA MELODY -- syncopated, call-and-response, chord tones + passing tones
# ============================================================================
# Techniques from research:
#   - Syncopation: notes on "and" beats (0.5, 1.5, 2.5, 3.5) for bounce
#   - Call-and-response: 2-bar call phrase, 2-bar answer with variation
#   - Passing tones: step-wise movement between chord tones
#   - Approach notes: half-step below target chord tone for tension/release
#   - Rhythmic motif: establish a pattern and repeat with pitch variation

# 2-bar rhythmic templates (beat_position, duration)
# Call: syncopated, ends with space for response
CALL_RHYTHM = [
    (0.0, 0.75),    # beat 1, dotted 8th (strong start)
    (1.5, 0.5),     # "and" of 2 (syncopated)
    (2.5, 0.5),     # "and" of 3 (syncopated)
    (3.5, 0.5),     # "and" of 4
    (4.0, 1.0),     # bar2 beat 1 (anchor)
    (6.0, 0.5),     # bar2 beat 3
    (6.5, 0.75),    # bar2 "and" of 3 (syncopated tail)
]

# Response: answers the call, slightly different rhythm
RESPONSE_RHYTHM = [
    (0.5, 0.5),     # bar3 "and" of 1 (pickup — doesn't start on 1)
    (1.0, 0.75),    # bar3 beat 2
    (2.5, 0.5),     # bar3 "and" of 3
    (3.0, 1.0),     # bar3 beat 4 (longer, resolving)
    (4.5, 0.5),     # bar4 "and" of 1
    (5.5, 0.5),     # bar4 "and" of 2
    (6.0, 1.5),     # bar4 beat 3, dotted quarter (resolves)
]


def create_flute(BPM, chords):
    part = stream.Part()
    part.partName = 'African Marimba'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    mt = chords['melody_tones']  # [root, 3rd, 5th, 7th, pass_below, pass_above]

    def _pick_pitch(bar_idx, note_idx, is_response=False):
        """Select pitch from chord tones + passing tones.
        Chord tones on strong positions, passing/approach on weak."""
        ct = mt[bar_idx % 4]
        # Strong positions: indices 0, 3, 4 in call; 3, 6 in response → chord tones
        # Weak positions: → passing tones or approach notes
        if note_idx in (0, 3, 6):
            # Chord tone: root, 3rd, or 5th
            return ct[note_idx % 3]
        elif note_idx in (1, 4):
            # 7th or passing tone
            return ct[3] if random.random() < 0.6 else ct[4]
        else:
            # Passing tone or approach note (half-step below next chord tone)
            return ct[5] if random.random() < 0.5 else ct[4]

    def melody_cycle(start_bar, vel_base=52):
        """Write a 4-bar call-and-response melodic cycle."""
        # Bars 0-1: Call
        o_call = bb(start_bar)
        call_chord = (start_bar) % 4
        for i, (beat, dur) in enumerate(CALL_RHYTHM):
            # Use chord from bar where the note falls
            bar_offset = int(beat // 4)
            ci = (start_bar + bar_offset) % 4
            pitch = _pick_pitch(ci, i, is_response=False)
            vel = vel_base + random.randint(-4, 4)
            # Accent syncopated notes slightly
            if beat != int(beat):
                vel = min(127, vel + 3)
            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = vel
            part.insert(o_call + beat, n)

        # Bars 2-3: Response (same rhythm template, different pitches)
        o_resp = bb(start_bar + 2)
        for i, (beat, dur) in enumerate(RESPONSE_RHYTHM):
            bar_offset = int(beat // 4)
            ci = (start_bar + 2 + bar_offset) % 4
            pitch = _pick_pitch(ci, i, is_response=True)
            vel = vel_base + random.randint(-4, 4)
            if beat != int(beat):
                vel = min(127, vel + 3)
            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = vel
            part.insert(o_resp + beat, n)

    # Hook1: melody enters bar 8, 4-bar cycles
    for cycle_start in range(HOOK1_S, HOOK1_E, 4):
        melody_cycle(cycle_start, vel_base=52)

    # Verse: sparse — only call phrase (no response), every 4 bars
    for cycle_start in range(VERSE_S, VERSE_E, 4):
        o = bb(cycle_start)
        ct = mt[cycle_start % 4]
        # Sparse: just 3 notes — root, 7th, root (sustained, dreamy)
        for pitch, beat, dur in [(ct[0], 0.0, 1.5), (ct[3], 2.0, 1.0), (ct[0], 3.5, 0.5)]:
            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = 42 + random.randint(-3, 3)
            part.insert(o + beat, n)

    # Hook2: full call-and-response
    for cycle_start in range(HOOK2_S, HOOK2_E, 4):
        melody_cycle(cycle_start, vel_base=54)

    # Bridge: prominent melody, single long phrases
    for bar in range(BRIDGE_S, BRIDGE_E):
        o = bb(bar)
        ct = mt[bar % 4]
        # Expressive: root held long, passing tone, resolve to 5th
        for pitch, beat, dur in [(ct[0], 0.0, 2.0), (ct[5], 2.5, 0.5), (ct[2], 3.0, 1.0)]:
            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = 56 + random.randint(-3, 3)
            part.insert(o + beat, n)

    # Hook3: full call-and-response
    for cycle_start in range(HOOK3_S, HOOK3_E, 4):
        melody_cycle(cycle_start, vel_base=52)

    return part


# ============================================================================
# HARP -- 16th note arp through chord tones
# ============================================================================

def create_harp(BPM, chords):
    part = stream.Part()
    part.partName = 'Classical Guitar'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    tones = chords['harp_tones']

    def harp_bar(bar, vel_base=40):
        o = bb(bar)
        ct = tones[bar % 4]
        # 16th note arp cycling through chord tones (16 notes per bar)
        for i in range(16):
            pitch = ct[i % len(ct)]
            beat_pos = i * 0.25
            vel = vel_base + random.randint(-3, 3)
            # Accent on-beat notes
            if i % 4 == 0:
                vel = min(127, vel + 6)
            n = note.Note(pitch, quarterLength=0.25)
            n.volume.velocity = vel
            part.insert(o + beat_pos, n)

    # Intro: no harp
    # Hook1: harp enters bar 12 (staggered, 4 bars after flute)
    for bar in range(HOOK1_S + 4, HOOK1_E):
        harp_bar(bar, vel_base=38)

    # Verse: harp continues, slightly quieter
    for bar in range(VERSE_S, VERSE_E):
        harp_bar(bar, vel_base=34)

    # Hook2
    for bar in range(HOOK2_S, HOOK2_E):
        harp_bar(bar, vel_base=40)

    # Bridge: no harp (flute + glock only)

    # Hook3
    for bar in range(HOOK3_S, HOOK3_E):
        harp_bar(bar, vel_base=40)

    # Outro: harp fades
    for bar in range(OUTRO_S, OUTRO_E):
        harp_bar(bar, vel_base=max(20, 36 - (bar - OUTRO_S) * 6))

    return part


# ============================================================================
# GRAND PIANO -- sustained chord voicings, quiet
# ============================================================================

def create_piano(BPM, chords):
    part = stream.Part()
    part.partName = 'Vibraphone'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    voicings = chords['piano']

    def piano_bar(bar, vel=36, dur=4.0):
        o = bb(bar)
        ch = chord.Chord(voicings[bar % 4], quarterLength=dur)
        ch.volume.velocity = vel
        part.insert(o, ch)

    # Intro: piano present, very quiet
    for bar in range(INTRO_S, INTRO_E):
        piano_bar(bar, vel=30)

    # Hook1
    for bar in range(HOOK1_S, HOOK1_E):
        piano_bar(bar, vel=36)

    # Verse
    for bar in range(VERSE_S, VERSE_E):
        piano_bar(bar, vel=34)

    # Hook2
    for bar in range(HOOK2_S, HOOK2_E):
        piano_bar(bar, vel=38)

    # Bridge: no piano

    # Hook3
    for bar in range(HOOK3_S, HOOK3_E):
        piano_bar(bar, vel=38)

    # Outro: piano fades
    for bar in range(OUTRO_S, OUTRO_E):
        piano_bar(bar, vel=max(18, 32 - (bar - OUTRO_S) * 5))

    return part


# ============================================================================
# GLOCKENSPIEL -- bell accents on hook downbeats
# ============================================================================

def create_glockenspiel(BPM, chords):
    part = stream.Part()
    part.partName = 'Glockenspiel'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    accents = chords['glock_accents']

    def glock_hit(bar, vel=44):
        o = bb(bar)
        pitch = accents[bar % 4]
        n = note.Note(pitch, quarterLength=0.5)
        n.volume.velocity = vel + random.randint(-3, 3)
        part.insert(o, n)

    # Hook sections only: accent every 2 bars on beat 1
    for bar in range(HOOK1_S, HOOK1_E, 2):
        glock_hit(bar, vel=44)

    for bar in range(HOOK2_S, HOOK2_E, 2):
        glock_hit(bar, vel=46)

    # Bridge: glockenspiel every bar
    for bar in range(BRIDGE_S, BRIDGE_E):
        glock_hit(bar, vel=50)

    for bar in range(HOOK3_S, HOOK3_E, 2):
        glock_hit(bar, vel=44)

    return part


# ============================================================================
# DRUMS -- simple 8th hats, clap on 2&4, punchy kick
# ============================================================================
# GM: 36=kick, 39=clap, 42=closed-HH, 46=open-HH, 49=crash
# SIMPLE: straight 8th hats, clean backbeat, no ghost notes or rolls.

def create_drums(BPM):
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def mosey_bar(bar, crash=False):
        o = bb(bar)

        # Kick: beats 1 and 3
        hit(o + 0.0, 36, 100)
        hit(o + 2.0, 36, 92)

        # Clap on beats 2 and 4, slightly behind the hat (+0.06 qn ≈ 25ms)
        hit(o + 1.06, 39, 94)
        hit(o + 3.06, 39, 96)

        # Hi-hats: straight 8th notes — on-beat louder, off-beat softer
        for i in range(8):
            beat_pos = i * 0.5
            vel = 56 if i % 2 == 0 else 38
            hit(o + beat_pos, 42, vel + random.randint(-3, 3))

        # Open hat on "and of 4" every other bar
        if bar % 2 == 1:
            hit(o + 3.5, 46, 45)

        if crash:
            hit(o + 0.0, 49, 85)

    # No drums in intro (bars 0-7)

    # Hook1
    for bar in range(HOOK1_S, HOOK1_E):
        idx = bar - HOOK1_S
        mosey_bar(bar, crash=(idx % 8 == 0))

    # Verse
    for bar in range(VERSE_S, VERSE_E):
        mosey_bar(bar, crash=(bar == VERSE_S))

    # Hook2
    for bar in range(HOOK2_S, HOOK2_E):
        idx = bar - HOOK2_S
        mosey_bar(bar, crash=(idx % 8 == 0))

    # Bridge: no drums

    # Hook3
    for bar in range(HOOK3_S, HOOK3_E):
        idx = bar - HOOK3_S
        mosey_bar(bar, crash=(idx % 8 == 0))

    # Outro
    for bar in range(OUTRO_S, OUTRO_E):
        mosey_bar(bar)

    return part


# ============================================================================
# 808 BASS -- tuned to chord roots, tight ADSR, slide notes
# ============================================================================

def create_808(BPM, chords):
    part = stream.Part()
    part.partName = '808 Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    roots = chords['bass_roots']

    def bass_bar(bar, vel=94, ghost=True):
        o = bb(bar)
        root = roots[bar % 4]

        # Main hit on beat 1 (2 beats = short-medium)
        n1 = note.Note(root, quarterLength=2.0)
        n1.volume.velocity = vel
        part.insert(o, n1)

        # Ghost bounce on beat 3 (short)
        if ghost:
            n2 = note.Note(root, quarterLength=0.75)
            n2.volume.velocity = int(vel * 0.35)
            part.insert(o + 2.0, n2)

    # Hook1
    for bar in range(HOOK1_S, HOOK1_E):
        bass_bar(bar, vel=96, ghost=True)

    # Verse
    for bar in range(VERSE_S, VERSE_E):
        bass_bar(bar, vel=90, ghost=(bar % 2 == 1))

    # Hook2
    for bar in range(HOOK2_S, HOOK2_E):
        bass_bar(bar, vel=98, ghost=True)

    # Bridge: no 808

    # Hook3
    for bar in range(HOOK3_S, HOOK3_E):
        bass_bar(bar, vel=96, ghost=True)

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
    'marimba': 12,  # Marimba
    'guitar':  24,  # Acoustic Guitar (nylon)
    'vibra':   11,  # Vibraphone
    'glock':   9,   # Glockenspiel
    '808':     38,  # Synth Bass 1
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
    parser = argparse.ArgumentParser(description='Lil Mosey type beat MIDI composer')
    parser.add_argument('--name', required=True, help='Beat name')
    parser.add_argument('--key', required=True, choices=['C', 'G'],
                        help='Key (C=bright F-Am-Em, G=bouncy G-Em-C-D)')
    parser.add_argument('--bpm', required=True, type=int, help='BPM (130-150)')
    args = parser.parse_args()

    BPM = min(args.bpm, 150)  # cap at 150 per workflow
    beat_name = args.name
    chords = PROGRESSION[args.key]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Composing {beat_name} ...')
    print(f'  Key: {args.key}  |  BPM: {BPM}  |  80 bars')

    parts = {
        'marimba':     create_flute(BPM, chords),
        'guitar':      create_harp(BPM, chords),
        'vibraphone':  create_piano(BPM, chords),
        'glockenspiel': create_glockenspiel(BPM, chords),
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
