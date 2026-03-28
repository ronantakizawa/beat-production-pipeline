"""
Shatter — Compose (Breakcore)
Key: D minor | BPM: 170 | 72 bars (~2:32)

Research: Path B — 4 YouTube tutorials on breakcore production
  - Complex breaks = simple chords/melody (2 chords repeating)
  - Amen break chopped, rearranged randomly with stutters (1/16, 1/32 rolls)
  - Heavy distortion/saturation on breaks
  - Reese bass (detuned saws + LPF + distortion)
  - Simple melody — piano/bell + delay, quiet behind breaks

Chord Progression (2-bar loop, D minor — i-VI):
  Bar 0 (even): Dm   (D3, F3, A3)   — i (tonic)
  Bar 1 (odd):  Bb   (Bb2, D3, F3)  — VI (submediant)

6 Sound Layers:
  1. Amen Breaks  — JSON slice map (NOT MIDI drums)
  2. Reese Bass   — FAUST synth (D1/Bb0 roots)
  3. Pad          — FAUST detuned saw (Dm/Bb chords)
  4. Melody       — sample-based (D minor pentatonic, sparse)
  5. Atmosphere   — noise/risers/impacts/downlifter
  6. Transition FX — reverse tails, snare rolls

Song Structure (72 bars):
  intro  bars  0- 7: pad + filtered amen ghost + noise
  build  bars  8-15: amen enters, bass creeps in, riser at 14-16
  drop1  bars 16-31: full chaos — chopped amen, reese bass, melody, impact
  break  bars 32-39: pad only, downlifter, breathing room
  drop2  bars 40-55: more complex chops, layered amens, higher energy
  outro  bars 56-71: breaks thin out, pad swells, fade
"""

import os
import json
import random
import numpy as np
from music21 import stream, note, chord, tempo, meter
from mido import MidiFile, Message

random.seed(42)
rng = random.Random(42)

# ============================================================================
# CONFIG
# ============================================================================

BEAT_NAME  = 'Shatter'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Shatter_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM   = 170
BPB   = 4
NBARS = 72

# Section boundaries
SECTIONS = {
    'intro':  (0,   8),
    'build':  (8,  16),
    'drop1':  (16, 32),
    'break':  (32, 40),
    'drop2':  (40, 56),
    'outro':  (56, 72),
}


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat (0-indexed)."""
    return float(bar * BPB + beat)


def section_for_bar(bar):
    """Return section name for a given bar."""
    for name, (s, e) in SECTIONS.items():
        if s <= bar < e:
            return name
    return 'outro'


# ============================================================================
# CHORD TABLES — D minor: i-VI (2-bar loop)
# ============================================================================

CHORDS = [
    ['D3',  'F3', 'A3'],     # bar 0 (even): Dm (i)
    ['Bb2', 'D3', 'F3'],     # bar 1 (odd):  Bb (VI)
]

BASS_ROOTS = ['D1', 'Bb0']   # even bars = D1, odd bars = Bb0


# ============================================================================
# AMEN SLICE MAP GENERATOR
# ============================================================================
# Slices 0-11 map to the 12 pre-chopped WAVs:
#   0-1: Kicks | 2,6: Snares | 3,4,7: Hat shuffles
#   5: Shuffle kick | 8-11: Pitch-shifted snares (+1 to +4 st)

KICK_SLICES  = [0, 1]
SNARE_SLICES = [2, 6]
HAT_SLICES   = [3, 4, 7]
SHUFFLE_KICK = [5]
PITCH_SNARES = [8, 9, 10, 11]
ALL_SLICES   = list(range(12))

# Standard amen order (classic breakbeat loop)
AMEN_ORDER = [0, 3, 2, 4, 1, 7, 6, 5]


def generate_slice_map(bar, section, rng):
    """Generate a list of slice events for one bar.

    Each event: {
        'slice': int (0-11),
        'beat': float (0.0-3.75),
        'dur': float (in beats, 0.125-1.0),
        'vel': float (0.0-1.0),
        'reverse': bool,
        'pitch_st': int (semitones shift, 0 default),
        'stutter': int (0=none, 2/4/8 = repeat count),
    }
    """
    events = []
    bar_in_section = bar - SECTIONS[section][0]

    if section == 'intro':
        # Ghost hats only, sparse, low velocity, some reversed
        for beat in [0.0, 1.0, 2.0, 3.0]:
            if rng.random() < 0.4:
                events.append({
                    'slice': rng.choice(HAT_SLICES),
                    'beat': beat,
                    'dur': 0.5,
                    'vel': rng.uniform(0.10, 0.25),
                    'reverse': rng.random() < 0.3,
                    'pitch_st': 0,
                    'stutter': 0,
                })

    elif section == 'build':
        # Standard amen order, 2-bar loop, increasing velocity
        progress = bar_in_section / 8.0
        base_vel = 0.35 + progress * 0.45
        for i, beat in enumerate(np.linspace(0, 3.5, 8)):
            sl = AMEN_ORDER[i % len(AMEN_ORDER)]
            events.append({
                'slice': sl,
                'beat': float(beat),
                'dur': 0.5,
                'vel': base_vel + rng.uniform(-0.05, 0.05),
                'reverse': False,
                'pitch_st': 0,
                'stutter': 0,
            })

    elif section == 'drop1':
        # Fixed rearranged amen, repeats every bar. Stutter only every 4th bar.
        DROP1_ORDER = [0, 2, 5, 3, 1, 6, 4, 7]
        for i in range(8):
            sl = DROP1_ORDER[i]
            beat = i * 0.5
            stutter = 0
            if bar_in_section % 4 == 3 and sl in SNARE_SLICES:
                stutter = 4
            events.append({
                'slice': sl,
                'beat': beat,
                'dur': 0.5 if stutter == 0 else 0.25,
                'vel': 0.80,
                'reverse': False,
                'pitch_st': 0,
                'stutter': stutter,
            })

    elif section == 'break':
        # Silence or very sparse ghost hats
        if rng.random() < 0.25:
            events.append({
                'slice': rng.choice(HAT_SLICES),
                'beat': rng.choice([0.0, 2.0]),
                'dur': 0.5,
                'vel': rng.uniform(0.08, 0.15),
                'reverse': True,
                'pitch_st': 0,
                'stutter': 0,
            })

    elif section == 'drop2':
        # Different fixed rearrangement, stutter every 2nd bar on snares
        DROP2_ORDER = [1, 6, 0, 4, 2, 7, 5, 3]
        for i in range(8):
            sl = DROP2_ORDER[i]
            beat = i * 0.5
            stutter = 0
            if bar_in_section % 2 == 1 and sl in SNARE_SLICES:
                stutter = 4
            events.append({
                'slice': sl,
                'beat': beat,
                'dur': 0.5 if stutter == 0 else 0.25,
                'vel': 0.85,
                'reverse': False,
                'pitch_st': 0,
                'stutter': stutter,
            })

    elif section == 'outro':
        # Standard amen order, velocity fades out over 16 bars
        fade = max(0.05, 1.0 - bar_in_section / 16.0)
        # First 8 bars: full loop fading. Last 8: just kicks + snares.
        if bar_in_section < 8:
            for i in range(8):
                events.append({
                    'slice': AMEN_ORDER[i],
                    'beat': i * 0.5,
                    'dur': 0.5,
                    'vel': 0.55 * fade,
                    'reverse': False,
                    'pitch_st': 0,
                    'stutter': 0,
                })
        else:
            # Sparse: kick on 1, snare on 3 only
            for beat, sl in [(0.0, 0), (2.0, 2)]:
                events.append({
                    'slice': sl,
                    'beat': beat,
                    'dur': 0.5,
                    'vel': 0.30 * fade,
                    'reverse': False,
                    'pitch_st': 0,
                    'stutter': 0,
                })

    return events


def create_break_schedule():
    """Generate the full 72-bar slice map and save as JSON."""
    print('Generating amen break slice map ...')
    schedule = {}
    for bar in range(NBARS):
        section = section_for_bar(bar)
        events = generate_slice_map(bar, section, rng)
        schedule[str(bar)] = {
            'section': section,
            'events': events,
        }
    return schedule


# ============================================================================
# REESE BASS (MIDI for FAUST synth)
# ============================================================================

def create_bass():
    """Reese bass following chord roots in D minor.
    Pattern: root on beat 1 (2.5 beats), hit on beat 3.5 (0.5 beats).
    Active in build (bars 12-15), drop1, drop2."""
    part = stream.Part()
    part.partName = 'Reese Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    bass_sections = [
        (12, 16, 75),    # build: creeps in, lower vel
        (16, 32, 95),    # drop1: full
        (40, 56, 100),   # drop2: full energy
    ]

    for s, e, base_vel in bass_sections:
        for bar in range(s, e):
            root = BASS_ROOTS[bar % 2]
            vel = int(base_vel + random.randint(-4, 4))
            # Main hit: beat 1, 2.5 beats long
            n1 = note.Note(root, quarterLength=2.5)
            n1.volume.velocity = min(127, max(1, vel))
            part.insert(bb(bar, 0.0), n1)
            # Syncopated hit: beat 3.5, 0.5 beats
            n2 = note.Note(root, quarterLength=0.5)
            n2.volume.velocity = min(127, max(1, int(vel * 0.55)))
            part.insert(bb(bar, 3.5), n2)

    return part


# ============================================================================
# PAD (Dm / Bb sustained chords)
# ============================================================================

def create_pad():
    """Sustained pad chords, active all sections.
    PAD TIMBRE RULE: 5x detuned saw, slow attack 0.5s, long release 3s,
    vel 30-50. Intro filter sweep 200->4000Hz."""
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    for bar in range(NBARS):
        section = section_for_bar(bar)
        c = chord.Chord(CHORDS[bar % 2], quarterLength=4.0)
        if section == 'intro':
            c.volume.velocity = 35 + random.randint(-3, 3)
        elif section == 'break':
            c.volume.velocity = 45 + random.randint(-3, 3)
        elif section in ('drop1', 'drop2'):
            c.volume.velocity = 38 + random.randint(-3, 3)
        elif section == 'outro':
            progress = (bar - SECTIONS['outro'][0]) / 16.0
            c.volume.velocity = int(38 + progress * 12) + random.randint(-2, 2)
        else:
            c.volume.velocity = 40 + random.randint(-3, 3)
        part.insert(bb(bar), c)

    return part


# ============================================================================
# MELODY (D minor pentatonic, sparse)
# ============================================================================

def create_melody():
    """Sparse melody in D minor pentatonic (D5, F5, G5, A5).
    Active in drop1 and drop2 only. 4-bar phrase, vel 35-48.
    MELODY TIMBRE RULE: bell/piano oneshot, vel 35-48.
    Never louder than amen breaks."""
    part = stream.Part()
    part.partName = 'Melody'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    SCALE = ['D5', 'F5', 'G5', 'A5']

    # 4-bar sparse phrase patterns (bar_offset, beat, pitch_idx, dur)
    PHRASE_A = [
        (0, 0.0,  3, 1.0),   # A5 on beat 1
        (0, 2.5,  1, 0.5),   # F5 syncopated
        (1, 1.0,  2, 1.5),   # G5 held
        (2, 0.0,  0, 0.5),   # D5 quick
        (2, 3.0,  3, 1.0),   # A5 end of bar
        (3, 2.0,  1, 1.0),   # F5
    ]

    PHRASE_B = [
        (0, 0.5,  2, 1.0),   # G5
        (1, 0.0,  3, 0.5),   # A5 quick
        (1, 3.0,  0, 1.0),   # D5
        (2, 1.5,  1, 0.5),   # F5
        (3, 0.0,  2, 1.5),   # G5 held
        (3, 3.0,  3, 0.5),   # A5 pickup
    ]

    melody_sections = [
        (16, 32),   # drop1
        (40, 56),   # drop2
    ]

    for s, e in melody_sections:
        for cycle_start in range(s, e, 4):
            phrase = PHRASE_A if ((cycle_start - s) // 4) % 2 == 0 else PHRASE_B
            for bar_off, beat, pitch_idx, dur in phrase:
                bar = cycle_start + bar_off
                if bar >= e:
                    break
                nd = note.Note(SCALE[pitch_idx], quarterLength=dur)
                nd.volume.velocity = random.randint(35, 48)
                part.insert(bb(bar, beat), nd)

    return part


# ============================================================================
# ARP (D minor, fills non-drop sections)
# ============================================================================

def create_arp():
    """Simple arp in D minor (D4-F4-A4-D5), active in intro, build, break, outro.
    Fills the sections where the main melody is absent.
    ARP TIMBRE RULE: bell/FM, vel 30-42. Quiet texture, not a lead."""
    part = stream.Part()
    part.partName = 'Arp'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    NOTES = ['D4', 'F4', 'A4', 'D5']   # Dm arp on even bars
    NOTES_BB = ['Bb3', 'D4', 'F4', 'Bb4']  # Bb arp on odd bars

    arp_sections = [
        (0,  8,  28),   # intro: very quiet
        (8,  16, 35),   # build: emerging
        (32, 40, 38),   # break: present
        (56, 72, 32),   # outro: fading
    ]

    for s, e, base_vel in arp_sections:
        for bar in range(s, e):
            pitches = NOTES if bar % 2 == 0 else NOTES_BB
            # 4 quarter-note hits per bar
            for i in range(4):
                nd = note.Note(pitches[i], quarterLength=0.75)
                nd.volume.velocity = base_vel + random.randint(-3, 3)
                part.insert(bb(bar, float(i)), nd)

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
    'reese': 38,   # Synth Bass 1
    'bass':  38,
    'pad':   89,   # Pad 2 (warm)
    'melody': 14,  # Tubular Bells
    'arp':   14,   # Tubular Bells
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
    print(f'  Key: D minor  |  BPM: {BPM}  |  {NBARS} bars')
    print(f'  Genre: breakcore  |  Chords: Dm-Bb (i-VI)')

    # Generate amen slice map (JSON, not MIDI)
    slice_schedule = create_break_schedule()
    slice_path = os.path.join(OUTPUT_DIR, f'{BEAT_NAME}_slices.json')
    with open(slice_path, 'w') as f:
        json.dump(slice_schedule, f, indent=2)
    total_events = sum(len(b['events']) for b in slice_schedule.values())
    print(f'  {BEAT_NAME}_slices.json  ({total_events} slice events across {NBARS} bars)')

    # Create MIDI parts (bass, pad, melody, arp — NOT drums)
    parts = {
        'bass':   create_bass(),
        'pad':    create_pad(),
        'melody': create_melody(),
        'arp':    create_arp(),
    }

    print('Saving individual stems ...')
    for stem_name, part in parts.items():
        save(solo(part), f'{BEAT_NAME}_{stem_name}.mid')

    print('\nSaving full arrangement ...')
    full = stream.Score()
    for part in parts.values():
        full.append(part)
    save(full, f'{BEAT_NAME}_FULL.mid')

    print(f'\nDone! Files saved to:\n  {OUTPUT_DIR}')
    print(f'  Slice map: {slice_path}')
