"""
WarCry — Compose (Che-style)
A Phrygian | 160 BPM | 96 bars (~2:24)

Research: 4 YouTube tutorials on Che beat production (CXO, Snapp, adanmade)
  - 150-170 BPM, Phrygian/minor keys
  - Layered processed melodies, sliding 808s
  - Gross Beat-style gate effects, chaotic hi-hats, snare rolls
  - Half-time kick (NOT four-on-the-floor)

Chord Progression (4-bar loop, A Phrygian — i-bII-iv-V):
  Bar 0: Am   (A3, C4, E4)       — i   (root)
  Bar 1: Bb   (Bb3, D4, F4)      — bII (root) — Phrygian hallmark
  Bar 2: Dm   (D4, F4, A4)       — iv  (1st inv, smooth voice leading)
  Bar 3: E    (E3, G#3, B3)      — V   (Phrygian dominant)

8 Sound Layers:
  1. Drums       — kick/snare/hat_closed/hat_open (half-time)
  2. Perc        — snare rolls, tom fills
  3. 808         — sliding glide bass (FAUST in render)
  4. Lead synth  — aggressive staccato gated melody
  5. Lead2 (arp) — arp layer following chord tones
  6. Pad/texture — dark sustained pad
  7. Siren FX    — slow pitch-bent background tone
  8. Transition  — risers, impacts, reverse crashes

Song Structure (96 bars):
  Intro        bars  0– 7:  filtered melody + atmosphere, no drums
  Verse 1      bars  8–23:  drums + 808 + melody (basic)
  Hook 1       bars 24–39:  full energy — all layers, snare rolls, FX
  Bridge       bars 40–47:  breakdown — atmosphere + filtered lead only
  Verse 2      bars 48–63:  drums + 808 + melody (variation)
  Hook 2       bars 64–87:  full energy — all layers + extra chaos
  Outro        bars 88–95:  strip down, fade
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

BEAT_NAME  = 'WarCry'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/WarCry_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 160
BPB = 4

# Section boundaries
INTRO_S,  INTRO_E  =  0,   8
VERSE1_S, VERSE1_E =  8,  24
HOOK1_S,  HOOK1_E  = 24,  40
BRIDGE_S, BRIDGE_E = 40,  48
VERSE2_S, VERSE2_E = 48,  64
HOOK2_S,  HOOK2_E  = 64,  88
OUTRO_S,  OUTRO_E  = 88,  96


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
# CHORD TABLES — A Phrygian i-bII-iv-V
# ============================================================================

CHORDS = [
    ['A3',  'C4',  'E4'],      # bar 0: Am (root)
    ['B-3', 'D4',  'F4'],      # bar 1: Bb (root) — bII Phrygian hallmark
    ['D4',  'F4',  'A4'],      # bar 2: Dm (1st inv, smooth voice leading)
    ['E3',  'G#3', 'B3'],      # bar 3: E  (Phrygian dominant V)
]

BASS_ROOTS = ['A1', 'B-1', 'D2', 'E1']

# Pad voicings — one octave above CHORDS
PAD_CHORDS = [
    ['A4',  'C5',  'E5'],      # bar 0: Am
    ['B-4', 'D5',  'F5'],      # bar 1: Bb
    ['D5',  'F5',  'A5'],      # bar 2: Dm
    ['E4',  'G#4', 'B4'],      # bar 3: E
]

# ============================================================================
# MELODY — derived from A Phrygian soprano pool [E5, F5, G5, A5]
# ============================================================================
# A Phrygian scale: A Bb C D E F G
# Contour: aggressive staccato — short hits with rests (gate effect in render)

LEAD_MELODY = [
    # Bar 0 (Am): staccato E5 hits + drop to C5
    [('E5', 0.25), (None, 0.25), ('E5', 0.25), (None, 0.25),
     ('C5', 0.5), (None, 0.5), ('A4', 0.5), (None, 0.5)],
    # Bar 1 (Bb): Bb stab — bII tension
    [('F5', 0.25), (None, 0.25), ('F5', 0.25), (None, 0.75),
     ('D5', 0.5), (None, 0.5), ('B-4', 0.5), (None, 0.5)],
    # Bar 2 (Dm): descending run
    [('A5', 0.25), (None, 0.25), ('G5', 0.25), (None, 0.25),
     ('F5', 0.5), (None, 0.5), ('D5', 0.5), (None, 0.5)],
    # Bar 3 (E): resolve tension
    [(None, 0.5), ('E5', 0.5), (None, 0.25), ('G#4', 0.5),
     (None, 0.25), ('B4', 0.5), (None, 1.0)],
]

# Lead2: arp layer following chord tones (eighth notes)
LEAD2_MELODY = [
    # Bar 0 (Am): A4->C5->E5 arp
    [('A4', 0.5), ('C5', 0.5), ('E5', 0.5), (None, 0.5),
     ('A4', 0.5), ('C5', 0.5), ('E5', 0.5), (None, 0.5)],
    # Bar 1 (Bb): Bb4->D5->F5
    [('B-4', 0.5), ('D5', 0.5), ('F5', 0.5), (None, 0.5),
     ('B-4', 0.5), ('D5', 0.5), ('F5', 0.5), (None, 0.5)],
    # Bar 2 (Dm): D4->F4->A4
    [('D4', 0.5), ('F4', 0.5), ('A4', 0.5), (None, 0.5),
     ('D4', 0.5), ('F4', 0.5), ('A4', 0.5), (None, 0.5)],
    # Bar 3 (E): E4->G#4->B4
    [('E4', 0.5), ('G#4', 0.5), ('B4', 0.5), (None, 0.5),
     ('E4', 0.5), ('G#4', 0.5), ('B4', 0.5), (None, 0.5)],
]

# Siren: slow pitch-bent note (MIDI for FAUST rendering)
SIREN_NOTES = ['A2', 'B-2', 'D3', 'E2']


# ============================================================================
# DRUMS — Che-style: half-time kick, hard snare, 16th hats + rolls
# ============================================================================
# GM mapping: 36=kick 38=snare 42=closed-HH 46=open-HH 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def che_groove(bar, crash=False):
        """Che groove: half-time kick (1,3), hard snare (2,4), 16th hats."""
        o = bb(bar)
        # Kick on beats 1 and 3 (half-time feel)
        hit(o + 0.0, 36, 105)
        hit(o + 2.0, 36, 100)
        # Snare on beats 2 and 4
        hit(o + 1.0, 38, 95)
        hit(o + 3.0, 38, 100)
        # 16th-note closed hi-hats with velocity accents
        for i in range(16):
            beat_pos = i * 0.25
            if i % 4 == 0:
                vel = 65
            elif i % 2 == 0:
                vel = 50
            else:
                vel = 38 + random.randint(-3, 3)
            hit(o + beat_pos, 42, vel)
        # Open hat on beat 4.5
        hit(o + 3.5, 46, 55)
        if crash:
            hit(o + 0.0, 49, 90)

    def hat_roll_32nd(bar, start_beat=3.0, length_beats=1.0):
        """32nd-note hi-hat fill."""
        o = bb(bar, start_beat)
        steps = int(length_beats / 0.125)
        for i in range(steps):
            vel = 45 + int((i / steps) * 30) + random.randint(-3, 3)
            hit(o + i * 0.125, 42, min(110, vel))

    def triplet_hats(bar):
        """Triplet hi-hat pattern for variation bars."""
        o = bb(bar)
        # Replace regular 16ths with triplet groupings on beats 3-4
        trip_dur = 1.0 / 3.0
        for beat in [2.0, 3.0]:
            for i in range(3):
                vel = 50 + random.randint(-5, 10)
                hit(o + beat + i * trip_dur, 42, vel)

    def outro_bar(bar):
        """Outro: thins out, velocity drops."""
        o = bb(bar)
        bar_idx = bar - OUTRO_S
        vel_drop = bar_idx * 8
        if bar_idx < 4:
            hit(o + 0.0, 36, max(40, 100 - vel_drop))
            hit(o + 2.0, 36, max(35, 95 - vel_drop))
            hit(o + 1.0, 38, max(35, 90 - vel_drop))
            hit(o + 3.0, 38, max(35, 90 - vel_drop))
            for i in range(16):
                vel = max(20, 40 - vel_drop + (8 if i % 4 == 0 else 0))
                hit(o + i * 0.25, 42, vel)
        elif bar_idx < 6:
            hit(o + 0.0, 36, max(30, 70 - vel_drop))
            for i in range(8):
                vel = max(15, 30 - bar_idx * 3)
                hit(o + i * 0.5, 42, vel)

    # Intro (bars 0-7): no drums

    # Verse 1 (bars 8-23): basic groove
    for bar in range(VERSE1_S, VERSE1_E):
        che_groove(bar)
        # Hat rolls at end of 4-bar phrases
        if bar % 4 == 3:
            hat_roll_32nd(bar, start_beat=3.0, length_beats=1.0)
        # Triplet hats on variation bars
        if bar % 8 == 7:
            triplet_hats(bar)

    # Hook 1 (bars 24-39): full energy + more rolls
    for bar in range(HOOK1_S, HOOK1_E):
        che_groove(bar, crash=(bar == HOOK1_S))
        if bar % 4 == 3:
            hat_roll_32nd(bar, start_beat=2.0, length_beats=2.0)
        if bar % 4 == 1:
            triplet_hats(bar)

    # Bridge (bars 40-47): no drums

    # Verse 2 (bars 48-63): groove with variation
    for bar in range(VERSE2_S, VERSE2_E):
        che_groove(bar)
        if bar % 4 == 3:
            hat_roll_32nd(bar, start_beat=3.0, length_beats=1.0)
        if bar % 8 == 3:
            triplet_hats(bar)

    # Hook 2 (bars 64-87): full chaos — more hat rolls, crashes
    for bar in range(HOOK2_S, HOOK2_E):
        che_groove(bar, crash=(bar == HOOK2_S or bar == HOOK2_S + 12))
        # More frequent hat rolls
        if bar % 2 == 1:
            hat_roll_32nd(bar, start_beat=3.0, length_beats=1.0)
        if bar % 4 == 3:
            hat_roll_32nd(bar, start_beat=2.0, length_beats=2.0)
        if bar % 4 == 1:
            triplet_hats(bar)

    # Outro (bars 88-95): strip down
    for bar in range(OUTRO_S, OUTRO_E):
        outro_bar(bar)

    return part


# ============================================================================
# PERC — snare rolls + tom fills at transitions
# ============================================================================
# GM mapping: 38=snare 45=tom-low 47=tom-mid 50=tom-hi

def create_perc():
    part = stream.Part()
    part.partName = 'Perc'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def hit(offset, note_num, vel=70):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def snare_roll(bar, start_beat=0.0, length_beats=4.0):
        """Triplet snare roll with velocity crescendo."""
        o = bb(bar, start_beat)
        trip_dur = 1.0 / 3.0
        steps = int(length_beats / trip_dur)
        for i in range(steps):
            vel = 40 + int((i / steps) * 50) + random.randint(-3, 3)
            hit(o + i * trip_dur, 38, min(120, vel))

    def tom_fill(bar):
        """Descending tom pattern: hi->mid->low."""
        o = bb(bar, 2.0)  # last 2 beats
        toms = [50, 47, 45, 50, 47, 45, 47, 45]
        for i, tom in enumerate(toms):
            vel = 75 - i * 3 + random.randint(-3, 3)
            hit(o + i * 0.25, tom, vel)

    # Snare rolls 2 bars before hooks
    for target in [HOOK1_S, HOOK2_S]:
        snare_roll(target - 2, start_beat=0.0, length_beats=4.0)
        snare_roll(target - 1, start_beat=0.0, length_beats=4.0)

    # Tom fills at transitions
    for bar in [VERSE1_E - 1, HOOK1_E - 1, VERSE2_E - 1, HOOK2_E - 1]:
        tom_fill(bar)

    # Extra snare rolls in hooks (every 8 bars at bar 7)
    for hook_s, hook_e in [(HOOK1_S, HOOK1_E), (HOOK2_S, HOOK2_E)]:
        for bar in range(hook_s, hook_e):
            if (bar - hook_s) % 8 == 7:
                snare_roll(bar, start_beat=2.0, length_beats=2.0)

    return part


# ============================================================================
# 808 BASS — sliding glide (MIDI for FAUST rendering)
# ============================================================================

def create_808():
    part = stream.Part()
    part.partName = '808'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def bass_note(bar, pitch, dur=3.5, vel=95):
        n = note.Note(pitch, quarterLength=dur)
        n.volume.velocity = vel
        part.insert(bb(bar), n)

    def glide_pattern(bar, root, vel=95):
        """Glide pattern: main note + short connecting note for portamento."""
        # Main note (3 beats)
        n1 = note.Note(root, quarterLength=3.0)
        n1.volume.velocity = vel
        part.insert(bb(bar), n1)
        # Short connecting note to next root (overlaps for glide)
        next_root = BASS_ROOTS[(bar + 1) % 4]
        n2 = note.Note(next_root, quarterLength=1.0)
        n2.volume.velocity = int(vel * 0.7)
        part.insert(bb(bar, 3.0), n2)

    # Active in verse + hooks
    for s_bar, e_bar in [(VERSE1_S, VERSE1_E), (HOOK1_S, HOOK1_E),
                          (VERSE2_S, VERSE2_E), (HOOK2_S, HOOK2_E)]:
        for bar in range(s_bar, e_bar):
            root = BASS_ROOTS[bar % 4]
            # Use glide pattern in hooks for sliding effect
            if s_bar in (HOOK1_S, HOOK2_S):
                glide_pattern(bar, root, vel=100)
            else:
                bass_note(bar, root, dur=3.5, vel=90)

    return part


# ============================================================================
# LEAD MELODY — aggressive staccato
# ============================================================================

def create_lead():
    part = stream.Part()
    part.partName = 'Lead'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(50, 65)):
        # Hook octave pattern: [base, base, +1, +1] per workflow.md MANDATORY
        for bar in range(s_bar, e_bar):
            bar_idx = bar % 4
            cycle_in_section = ((bar - s_bar) % 16) // 4
            oct_shift = 1 if cycle_in_section >= 2 else 0
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

    # Intro: filtered melody (quiet)
    write_phrase(INTRO_S, INTRO_E, vel_range=(30, 40))

    # Verse 1: basic melody
    write_phrase(VERSE1_S, VERSE1_E, vel_range=(50, 62))

    # Hook 1: full energy
    write_phrase(HOOK1_S, HOOK1_E, vel_range=(60, 75))

    # Bridge: filtered (quiet)
    write_phrase(BRIDGE_S, BRIDGE_E, vel_range=(30, 40))

    # Verse 2: variation
    write_phrase(VERSE2_S, VERSE2_E, vel_range=(52, 65))

    # Hook 2: max energy
    write_phrase(HOOK2_S, HOOK2_E, vel_range=(65, 80))

    return part


# ============================================================================
# LEAD2 — arp layer following chord tones
# ============================================================================

def create_lead2():
    part = stream.Part()
    part.partName = 'Lead2'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(35, 50)):
        for bar in range(s_bar, e_bar):
            bar_idx = bar % 4
            motif = LEAD2_MELODY[bar_idx]
            beat = 0.0
            for pitch, dur in motif:
                if pitch is None:
                    part.insert(bb(bar, beat), note.Rest(quarterLength=dur))
                else:
                    n = note.Note(pitch, quarterLength=dur)
                    n.volume.velocity = random.randint(vel_range[0], vel_range[1])
                    part.insert(bb(bar, beat), n)
                beat += dur

    # Staggered entry within hooks (per melody layering MANDATORY):
    # Hook bars 9-16: + lead2 arp
    write_phrase(HOOK1_S + 8, HOOK1_E, vel_range=(38, 50))
    write_phrase(HOOK2_S + 8, HOOK2_E, vel_range=(40, 55))

    return part


# ============================================================================
# PAD — dark sustained chords
# ============================================================================

def create_pad():
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    sections = {
        'intro':   (INTRO_S,        INTRO_E,  35),
        'bridge':  (BRIDGE_S,       BRIDGE_E, 40),
        'hook1':   (HOOK1_S + 8,    HOOK1_E,  48),
        'hook2':   (HOOK2_S + 8,    HOOK2_E,  52),
        'outro':   (OUTRO_S,        OUTRO_E,  30),
    }

    for section_key, (s_bar, e_bar, vel) in sections.items():
        for bar in range(s_bar, e_bar):
            c = chord.Chord(PAD_CHORDS[bar % 4], quarterLength=4.0)
            c.volume.velocity = vel
            part.insert(bb(bar), c)

    return part


# ============================================================================
# SIREN FX — slow pitch-bent background tone (MIDI for FAUST rendering)
# ============================================================================

def create_siren():
    part = stream.Part()
    part.partName = 'Siren'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Active in hooks only (background, low vol)
    for s_bar, e_bar in [(HOOK1_S + 8, HOOK1_E), (HOOK2_S + 8, HOOK2_E)]:
        for bar in range(s_bar, e_bar):
            root = SIREN_NOTES[bar % 4]
            n = note.Note(root, quarterLength=4.0)
            n.volume.velocity = 45
            part.insert(bb(bar), n)

    return part


# ============================================================================
# TRANSITION FX — risers + impacts (MIDI markers for render)
# ============================================================================

def create_transition():
    part = stream.Part()
    part.partName = 'Transition'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Risers before hooks (2 bars)
    for target_bar in [HOOK1_S, HOOK2_S]:
        for i in range(8):  # 2 bars of quarter notes as riser markers
            n = note.Note('C5', quarterLength=0.5)
            n.volume.velocity = 40 + i * 8
            part.insert(bb(target_bar - 2, i * 0.5), n)

    # Impact hits at hook starts
    for target_bar in [HOOK1_S, HOOK2_S]:
        n = note.Note('C3', quarterLength=1.0)
        n.volume.velocity = 100
        part.insert(bb(target_bar), n)

    # Downlifter at bridge
    for i in range(4):
        n = note.Note('C4', quarterLength=1.0)
        n.volume.velocity = max(20, 60 - i * 12)
        part.insert(bb(BRIDGE_S, i), n)

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
    '808':    38,
    'bass':   38,
    'lead':   80,
    'lead2':  80,
    'pad':    89,
    'siren':  80,
}


def fix_instruments(mid, part_names):
    for i, track in enumerate(mid.tracks):
        if i == 0:
            continue
        pidx = i - 1
        if pidx >= len(part_names):
            break
        name = part_names[pidx].lower()
        if 'drum' in name or 'perc' in name:
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
    print(f'  Key: A Phrygian  |  BPM: {BPM}  |  96 bars')

    parts = {
        'drums':      create_drums(),
        'perc':       create_perc(),
        '808':        create_808(),
        'lead':       create_lead(),
        'lead2':      create_lead2(),
        'pad':        create_pad(),
        'siren':      create_siren(),
        'transition': create_transition(),
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
