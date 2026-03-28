"""
Tainy Clone v2 — Compose
Key: Ab minor (1A Camelot) | BPM: 112 | 105 bars (~3:45)

Research: Path A — ref_analysis.json (corrected key from metadata)
  - Key: Ab minor (Spotify metadata confirms 1A Camelot)
  - BPM: 112 (confirmed by both analysis and metadata)
  - Duration: ~3:45

Chord Progression (4-bar loop, Ab minor — i-VI-VII-III):
  Bar 0: Abm  (Ab3, B3, Eb4)   — i (tonic)
  Bar 1: E    (E3, Ab3, B3)    — VI (submediant, enharmonic Fb)
  Bar 2: Gb   (Gb3, Bb3, Db4)  — VII (subtonic)
  Bar 3: B    (B3, Eb4, Gb4)   — III (mediant, enharmonic Cb)

  Source: Free-Chord-Progressions-main/GitHub Free Progressions/Ab/Ab Minor/
  Classic reggaeton cadence: i-VI-VII-III

4 Sound Layers:
  1. Drums      — pattern from ref_analysis.json
  2. Sub bass   — sub_bass following chord roots Ab1/E1/Gb1/B1
  3. Pad        — sustained string chords
  4. Lead       — Ab minor lead synth motif (Eb5-Db5-B4)

Song Structure (9 sections, 105 bars — from ref_analysis.json):
  intro    bars   0-34:  pad+lead only, filter sweep in
  bridge1  bars  34-38:  drums+bass enter
  verse1   bars  38-55:  full kit + vocals section
  verse2   bars  55-59:  full kit
  verse3   bars  59-64:  full kit
  verse4   bars  64-72:  full kit
  bridge2  bars  72-81:  pad+lead only (drop)
  verse5   bars  81-85:  drums+bass re-enter
  outro    bars  85-105: bass+pad+lead, fade out

Kit: reggaetondrums + REGGAETON 4
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

BEAT_NAME  = 'Tainy'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Tainy_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM   = 100
BPB   = 4
NBARS = 105

# Section boundaries (from ref_analysis.json sections)
SECTIONS = {
    'intro':   (0,   34),
    'bridge1': (34,  38),
    'verse1':  (38,  55),
    'verse2':  (55,  59),
    'verse3':  (59,  64),
    'verse4':  (64,  72),
    'bridge2': (72,  81),
    'verse5':  (81,  85),
    'outro':   (85, 105),
}

# Energy curve (105 values from ref_analysis.json)
ENERGY_CURVE = [
    0.0, 0.0, 0.0, 0.1977, 0.2448, 0.1964, 0.1912, 0.1428,
    0.0379, 0.0322, 0.0707, 0.0721, 0.0351, 0.057, 0.0571, 0.0904,
    0.1321, 0.0304, 0.0807, 0.0928, 0.1499, 0.0705, 0.022, 0.0849,
    0.1858, 0.2232, 0.2315, 0.3241, 0.2658, 0.1487, 0.2382, 0.3006,
    0.2196, 0.2547, 0.0002, 0.1639, 0.6349, 0.1378, 0.617, 0.1027,
    0.4247, 0.3315, 0.5963, 0.7851, 1.0, 0.9693, 0.991, 0.722,
    0.6465, 0.7944, 0.3778, 0.538, 0.2326, 0.5419, 0.1184, 0.0005,
    0.5401, 0.4363, 0.6357, 0.5057, 0.35, 0.3679, 0.2018, 0.6243,
    0.0739, 0.0004, 0.0006, 0.455, 0.0921, 0.5242, 0.6161, 0.9323,
    0.0442, 0.1111, 0.1929, 0.0629, 0.0955, 0.1559, 0.1684, 0.103,
    0.0804, 0.0001, 0.3528, 0.4954, 0.4279, 0.606, 0.628, 0.3964,
    0.385, 0.2583, 0.6217, 0.2861, 0.2173, 0.1672, 0.2948, 0.2605,
    0.0867, 0.0449, 0.0736, 0.0555, 0.0, 0.0, 0.0, 0.0,
    0.0,
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
# CHORD TABLES — Ab minor: i-VI-VII-III
# Voicings derived from Free-Chord-Progressions-main Ab Minor MIDI files
# ============================================================================

CHORDS = [
    ['Ab3', 'B3',  'Eb4'],     # bar 0: Abm (i)     — Cb=B enharmonic
    ['E3',  'Ab3', 'B3'],      # bar 1: E   (VI)    — Fb=E enharmonic
    ['Gb3', 'Bb3', 'Db4'],     # bar 2: Gb  (VII)
    ['B3',  'Eb4', 'Gb4'],     # bar 3: B   (III)   — Cb=B enharmonic
]

# Bass roots following chord progression in Ab minor
BASS_ROOTS = ['Ab1', 'E1', 'Gb1', 'B1']


# ============================================================================
# DRUMS — pattern from ref_analysis.json
# ============================================================================
# GM drum mapping: 36=kick 37=rim 38=snare 39=clap
#                  42=closed-HH 46=open-HH 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    # Dembow patterns — A (basic) and B (variation)
    KICK_A = [0.0, 2.0]                                        # beats 1, 3
    KICK_B = [0.0, 0.75, 2.0]                                  # bouncy: extra kick
    CLAP_A = [1.0, 1.75, 3.0, 3.75]                            # dembow syncopation
    CLAP_B = [1.0, 1.75, 2.5, 3.75]                            # shifted snare
    HH_8TH = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]       # straight 8ths
    HH_16TH = [i * 0.25 for i in range(16)]                    # 16th notes
    CRASH_EVERY = 16

    def drum_bar(bar, crash=False, fill=False):
        o = bb(bar)
        vs = energy_vel(bar, 100) / 100.0
        bar_in_cycle = bar % 4

        # Alternate kick pattern every other bar
        kicks = KICK_B if bar_in_cycle in (1, 3) else KICK_A
        for b in kicks:
            hit(o + b, 36, 100 * vs)

        # Fill bar: snare roll on beats 3-4
        if fill:
            for b in [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]:
                hit(o + b, 39, int(70 * vs * (0.7 + 0.3 * b / 4.0)))
            for b in CLAP_A[:2]:  # keep beats 1-2 normal
                hit(o + b, 39, 88 * vs)
        else:
            # Alternate clap pattern on bar 3 of cycle
            claps = CLAP_B if bar_in_cycle == 2 else CLAP_A
            for b in claps:
                hit(o + b, 39, 88 * vs)

        # Hi-hats: 16ths in verses, 8ths elsewhere (quieter ghost notes)
        in_verse = any(SECTIONS[s][0] <= bar < SECTIONS[s][1]
                       for s in ('verse1', 'verse2', 'verse3', 'verse4', 'verse5'))
        if in_verse and bar_in_cycle in (2, 3):
            for b in HH_16TH:
                accent = b in (0.0, 1.0, 2.0, 3.0)
                vel = (55 if accent else 35) * vs
                hit(o + b, 42, vel)
        else:
            for b in HH_8TH:
                hit(o + b, 42, 55 * vs)

        # Open hat on odd bars
        if bar % 2 == 1:
            hit(o + 3.5, 46, 50 * vs)

        if crash:
            hit(o + 0.0, 49, 80 * vs)

    # Drums drop out in intro, bridge2, outro for contrast
    drum_sections = ['bridge1', 'verse1', 'verse2', 'verse3', 'verse4',
                     'verse5']
    for sec_name in drum_sections:
        s, e = SECTIONS[sec_name]
        for bar in range(s, e):
            is_fill = (bar == e - 1 and sec_name not in ('outro',))
            drum_bar(bar, crash=(bar % CRASH_EVERY == 0 or bar == s),
                     fill=is_fill)

    return part


# ============================================================================
# SUB BASS — Ab minor roots
# ============================================================================

def create_bass():
    """Sub bass following chord roots in Ab minor.
    SUB BASS TIMBRE RULE: sine osc, LPF 200Hz, no harmonics above 200Hz."""
    part = stream.Part()
    part.partName = 'Sub Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Simplified bass pattern from ref_analysis.json (key events)
    BASS_EVENTS = [
        (0.0,  2.0),    # main hit, sustained
        (2.5,  0.75),   # syncopated pickup
        (3.5,  0.5),    # anticipation
    ]

    # Bass active in: bridge1, verse1-4, verse5, outro
    bass_sections = ['bridge1', 'verse1', 'verse2', 'verse3', 'verse4',
                     'verse5', 'outro']
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
# LEAD MELODY — Ab minor lead synth, Eb5-B4 range, 2-bar cycle
# Derived from chord top notes in Free-Chord-Progressions Ab Minor MIDI files
# ============================================================================

def create_lead():
    """Lead synth in Ab minor. 2-bar phrase using Ab minor scale.
    Scale: Ab Bb B(Cb) Db Eb E(Fb) Gb. Centered around B4-Eb5.
    LEAD TIMBRE RULE: sine+FM, medium attack, vel 35-48.
    Never louder than drums/bass. Do NOT use saw waves."""
    part = stream.Part()
    part.partName = 'Lead'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # 2-bar motif derived from Ab minor chord top notes (Eb5, B4, Db5, Gb4)
    # with passing tones from Ab minor scale
    LEAD_PHRASE = [
        # Bar 0: descending from Eb5 through chord tones
        [('Eb5', 0.5), ('Db5', 0.5), ('B4', 0.5), (None, 0.5),
         ('Db5', 0.5), ('B4', 0.5), ('Ab4', 0.5), (None, 0.5)],
        # Bar 1: response with wider intervals, resolving to B4
        [('Eb5', 0.5), ('B4', 0.5), ('Db5', 0.75), (None, 0.25),
         ('B4', 0.5), ('Gb4', 0.5), ('Ab4', 0.5), (None, 0.5)],
    ]

    for bar in range(NBARS):
        phrase_bar = LEAD_PHRASE[bar % 2]
        base_vel = 42
        if bar < SECTIONS['intro'][1]:
            base_vel = 30  # quieter in intro
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
    print(f'Composing {BEAT_NAME} v2 ...')
    print(f'  Key: Ab minor (1A)  |  BPM: {BPM}  |  {NBARS} bars')
    print(f'  Genre: reggaeton  |  Chords: Abm-E-Gb-B (i-VI-VII-III)')

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
