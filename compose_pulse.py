"""
Pulse — Compose (House)
G minor | 124 BPM | 128 bars (~4:08)

Research: Path B — 6 YouTube tutorials on house production
  - Two energy levels: sustained pad (low) vs rhythmic stab/pluck (high)
  - Swing on drums, bass, stabs (~15ms off-beat delay)
  - Bass follows stab chord rhythm (root, shortened)
  - Shadow percussion: shaker/tamb/conga behind main drums
  - Dub chords: filtered pad with movement
  - 8-bar rule: new element halfway through every 16-bar section

Chord Progression (4-bar loop, G minor — i-VII-III-VI):
  Bar 0: Gm  (G3, Bb3, D4)      — i   (root)
  Bar 1: F   (F3, A3, C4)       — VII (root)
  Bar 2: Bb  (Bb3, D4, F4)      — III (root)
  Bar 3: Eb  (Bb3, Eb4, G4)     — VI  (2nd inv, keeps ±7st)

10 Sound Layers:
  1. Drums       — kick/clap/hat_closed/hat_open/crash
  2. Shadow perc — shaker/tamb/conga
  3. Sub bass    — FAUST sine (render only), MIDI roots here
  4. Mid bass    — rhythmic, follows stab pattern
  5. Chord stab  — rhythmic chord hits in drops
  6. Lead melody — G minor, chord-derived, syncopated arch
  7. Lead2       — same melody, softer (Drop 2 only)
  8. Pluck       — rhythmic chord pattern (drops)
  9. Pad         — sustained chords (dub chord in render)

Song Structure (128 bars — 4-bar buildups, 32-bar drops):
  Intro       bars   0– 15: drums only (kick + hat + shaker)
  Verse 1     bars  16– 31: pad + bass, low energy
  Buildup 1   bars  32– 35: hat crescendo, kicks intensify
  Drop 1      bars  36– 67: staggered entry (8-bar rule):
                  36–43: drums + bass + chord stab
                  44–51: + lead melody
                  52–59: + lead2 counter-melody + pluck + pad
                  60–67: full energy
  Breakdown   bars  68– 75: pad + filtered melody only
  Buildup 2   bars  76– 79: bigger hat crescendo
  Drop 2      bars  80–111: staggered entry:
                  80–87: drums + bass + stab + lead
                  88–95: + lead2 + pluck + pad
                  96–111: full energy
  Outro       bars 112–127: strip down, fade
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

BEAT_NAME  = 'Pulse'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Pulse_Beat'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BPM = 124
BPB = 4

# Section boundaries
INTRO_S,  INTRO_E  =   0,  16
VERSE_S,  VERSE_E  =  16,  32
BUILD1_S, BUILD1_E =  32,  36
DROP1_S,  DROP1_E  =  36,  68
BREAK_S,  BREAK_E  =  68,  76
BUILD2_S, BUILD2_E =  76,  80
DROP2_S,  DROP2_E  =  80, 112
OUTRO_S,  OUTRO_E  = 112, 128

# Swing: delay off-beat 8th notes by ~15ms (in quarter-note units at 124 BPM)
SWING_AMOUNT = 0.03  # ~15ms at 124 BPM


def bb(bar, beat=0.0):
    """Absolute quarter-note offset from bar (0-indexed) + beat (0-indexed)."""
    return float(bar * BPB + beat)


def swing(beat_pos):
    """Return swing offset for off-beat 8th notes."""
    frac = beat_pos % 1.0
    if abs(frac - 0.5) < 0.01:
        return SWING_AMOUNT
    return 0.0


def shift_pitch(pitch_str, oct_offset):
    if pitch_str is None or oct_offset == 0:
        return pitch_str
    n = note.Note(pitch_str)
    n.octave += oct_offset
    return n.nameWithOctave


# ============================================================================
# CHORD TABLES — G minor i-VII-III-VI
# All voicings within ±7 semitones of C4 (MIDI 60)
# Reference: chordprogressions/GitHub Free Progressions/G/G Minor/
# ============================================================================

CHORDS = [
    ['G3',  'B-3', 'D4'],    # bar 0: G minor (root) — shifts from stab root 60: -5, -2, +2
    ['F3',  'A3',  'C4'],    # bar 1: F major (root) — shifts: -7, -3, 0
    ['B-3', 'D4',  'F4'],    # bar 2: Bb major (root) — shifts: -2, +2, +5
    ['B-3', 'E-4', 'G4'],    # bar 3: Eb major (2nd inv) — shifts: -2, +3, +7
]

BASS_ROOTS = ['G1', 'F1', 'B-1', 'E-1']

# Pad voicings — one octave above CHORDS (pad root=72/C5, all within ±7)
PAD_CHORDS = [
    ['G4',  'B-4', 'D5'],    # bar 0: G minor
    ['F4',  'A4',  'C5'],    # bar 1: F major
    ['B-4', 'D5',  'F5'],    # bar 2: Bb major
    ['B-4', 'E-5', 'G5'],    # bar 3: Eb major (2nd inv)
]

# ============================================================================
# MELODY — derived from G Minor chord TOP NOTES (chordprogressions/ reference)
# ============================================================================
# Source: chordprogressions/GitHub Free Progressions/G/G Minor/
#   Gm top notes: D5(74), Bb4(70)
#   F  top notes: F5(77), Bb4(70)  — using C5(72) as lower anchor
#   Bb top notes: F5(77), D5(74)
#   Eb top notes: G5(79), Eb5(75)
#
# G minor scale: G A Bb C D Eb F
# Contour: "double-hit hook" — repeated staccato note launches each bar,
# then descends to lower chord tone. Call (bars 0-1) / response (bars 2-3).
# Different from Horizon (ascending arpeggios) and Elevate (descending sustained).

LEAD_MELODY = [
    # Bar 0 (Gm i): double-hit Bb4 (Gm 3rd), descend through G4 to D4
    [('B-4', 0.5), ('B-4', 0.5), (None, 0.5), ('G4', 1.0), (None, 0.5), ('D4', 1.0)],
    # Bar 1 (F VII): double-hit A4 (F 3rd), settle on F4
    [('A4', 0.5), ('A4', 0.5), (None, 0.5), ('F4', 1.5), (None, 1.0)],
    # Bar 2 (Bb III): same hook rhythm, descend through F4 to D4
    [('B-4', 0.5), ('B-4', 0.5), (None, 0.5), ('F4', 0.5), ('D4', 1.0), (None, 1.0)],
    # Bar 3 (Eb VI): break pattern — held Bb4 (Eb 5th), resolve to G4
    [(None, 0.5), ('B-4', 1.5), (None, 0.5), ('G4', 1.0), (None, 0.5)],
]

# Lead2 = counter-melody (answer notes on beat 4, fills lead's rests)
# Sample: Nylon Guitar d4, verified root=62 (D4). All shifts within ±7.
# Shift table: G4(67)=+5, C4(60)=-2, F4(65)=+3, G3(55)=-7
LEAD2_MELODY = [
    # Bar 0 (Gm): answer with G4 (root) on beat 4
    [(None, 3.0), ('G4', 1.0)],
    # Bar 1 (F): answer with C4 (5th) on beat 4
    [(None, 3.0), ('C4', 1.0)],
    # Bar 2 (Bb): answer with F4 (5th) on beat 4
    [(None, 3.0), ('F4', 1.0)],
    # Bar 3 (Eb): answer with G3 (3rd) on beat 4
    [(None, 3.0), ('G3', 1.0)],
]

# Pluck = chord arpeggios in octave 3 (not lead -1 octave, avoids low register shifts)
PLUCK_MELODY = [
    # Bar 0 (Gm): G3→Bb3→D4 up-arpeggio, repeated
    [('G3', 0.5), ('B-3', 0.5), ('D4', 0.5), (None, 0.5),
     ('G3', 0.5), ('B-3', 0.5), ('D4', 0.5), (None, 0.5)],
    # Bar 1 (F): F3→A3→C4
    [('F3', 0.5), ('A3', 0.5), ('C4', 0.5), (None, 0.5),
     ('F3', 0.5), ('A3', 0.5), ('C4', 0.5), (None, 0.5)],
    # Bar 2 (Bb): F3→Bb3→D4 (root position, stays within ±7 of pluck root 54)
    [('F3', 0.5), ('B-3', 0.5), ('D4', 0.5), (None, 0.5),
     ('F3', 0.5), ('B-3', 0.5), ('D4', 0.5), (None, 0.5)],
    # Bar 3 (Eb): Eb3→G3→Bb3, then rest
    [('E-3', 0.5), ('G3', 0.5), ('B-3', 0.5), (None, 0.5),
     ('E-3', 0.5), ('G3', 0.5), ('B-3', 0.5), (None, 0.5)],
]


# ============================================================================
# DRUMS
# ============================================================================
# GM mapping: 36=kick 39=clap 42=closed-HH 46=open-HH 49=crash

def create_drums():
    part = stream.Part()
    part.partName = 'Drums'
    part.insert(0, tempo.MetronomeMark(number=BPM))
    part.insert(0, meter.TimeSignature('4/4'))

    def hit(offset, note_num, vel=90):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def house_groove(bar, crash=False):
        """House groove: four-on-the-floor kick, clap 2+4, swung hats."""
        o = bb(bar)
        # Kick on every beat
        for b in range(4):
            hit(o + b, 36, 100)
        # Clap on 2 and 4
        hit(o + 1.0, 39, 88)
        hit(o + 3.0, 39, 92)
        # Closed hi-hat on 8th notes with swing on off-beats
        for i in range(8):
            beat_pos = i * 0.5
            vel = 55 if i % 2 == 0 else 40
            hit(o + beat_pos + swing(beat_pos), 42, vel)
        # Open hat on beat 4.5 (swung)
        hit(o + 3.5 + SWING_AMOUNT, 46, 50)
        if crash:
            hit(o + 0.0, 49, 85)

    def intro_groove(bar):
        """Intro: kick + hats only, minimal."""
        o = bb(bar)
        # Kick on every beat
        for b in range(4):
            hit(o + b, 36, 90)
        # Closed hi-hat on 8th notes with swing
        for i in range(8):
            beat_pos = i * 0.5
            vel = 48 if i % 2 == 0 else 32
            hit(o + beat_pos + swing(beat_pos), 42, vel)

    def buildup_hats(bar, bar_in_build, total_bars):
        """Buildup: 16th-note closed hi-hats with velocity crescendo."""
        o = bb(bar)
        progress = (bar_in_build + 0.5) / total_bars
        base_vel = int(35 + progress * 50)
        for i in range(16):
            accent = 8 if (i % 4 == 0) else 0
            vel = min(110, base_vel + accent + random.randint(-3, 3))
            hit(o + i * 0.25, 42, vel)

    def buildup_bar(bar, phase='early'):
        """Buildup: kicks (half-time early, every beat late)."""
        o = bb(bar)
        if phase == 'early':
            hit(o + 0.0, 36, 90)
            hit(o + 2.0, 36, 90)
        else:
            for b in range(4):
                hit(o + b, 36, 95)

    def outro_bar(bar):
        """Outro: groove thins out, velocity drops."""
        o = bb(bar)
        bar_idx = bar - OUTRO_S
        if bar_idx < 8:
            # Kick thins
            for b in range(4):
                hit(o + b, 36, max(40, 90 - bar_idx * 6))
            # Hats thin
            for i in range(8):
                beat_pos = i * 0.5
                vel = max(20, (45 if i % 2 == 0 else 30) - bar_idx * 3)
                hit(o + beat_pos + swing(beat_pos), 42, vel)
        elif bar_idx < 12:
            # Just kick
            for b in range(4):
                hit(o + b, 36, max(30, 70 - bar_idx * 4))

    # Intro (bars 0-15): kick + hats
    for bar in range(INTRO_S, INTRO_E):
        intro_groove(bar)

    # Verse (bars 16-31): add claps to groove
    for bar in range(VERSE_S, VERSE_E):
        house_groove(bar)

    # Buildup 1 (bars 32-35): hat crescendo
    for bar in range(BUILD1_S, BUILD1_E):
        idx = bar - BUILD1_S
        buildup_bar(bar, 'early' if idx < 2 else 'late')
        buildup_hats(bar, idx, BUILD1_E - BUILD1_S)

    # Drop 1 (bars 36-67): full house groove
    for bar in range(DROP1_S, DROP1_E):
        house_groove(bar, crash=(bar == DROP1_S))

    # Breakdown (bars 68-75): no drums

    # Buildup 2 (bars 76-79): bigger hat crescendo
    for bar in range(BUILD2_S, BUILD2_E):
        idx = bar - BUILD2_S
        buildup_bar(bar, 'early' if idx < 1 else 'late')
        buildup_hats(bar, idx, BUILD2_E - BUILD2_S)

    # Drop 2 (bars 80-111): full house groove
    for bar in range(DROP2_S, DROP2_E):
        house_groove(bar, crash=(bar == DROP2_S))

    # Outro (bars 112-127): thin out
    for bar in range(OUTRO_S, OUTRO_E):
        outro_bar(bar)

    return part


# ============================================================================
# SHADOW PERCUSSION — shaker/tamb/conga (layered behind main drums)
# ============================================================================
# GM mapping: 70=maracas(shaker) 54=tambourine 63=conga-open-hi

def create_shadow_perc():
    part = stream.Part()
    part.partName = 'Shadow Perc'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def hit(offset, note_num, vel=40):
        n = note.Note(note_num, quarterLength=0.25)
        n.volume.velocity = min(127, max(1, int(vel)))
        part.insert(offset, n)

    def shaker_pattern(bar, vel_base=28):
        """16th-note shaker, very quiet — adds shuffle."""
        o = bb(bar)
        for i in range(16):
            vel = vel_base + (5 if i % 4 == 0 else 0) + random.randint(-3, 3)
            beat_pos = i * 0.25
            hit(o + beat_pos + swing(beat_pos), 70, vel)

    def tamb_pattern(bar, vel_base=35):
        """Tambourine on beats 2 and 4, layered behind clap."""
        o = bb(bar)
        hit(o + 1.0, 54, vel_base + random.randint(-3, 3))
        hit(o + 3.0, 54, vel_base + random.randint(-3, 3))

    def conga_pattern(bar, vel_base=38):
        """Syncopated conga: beats 1.5, 3, 4.5 — adds groove."""
        o = bb(bar)
        hit(o + 0.5 + SWING_AMOUNT, 63, vel_base + random.randint(-3, 3))
        hit(o + 2.0, 63, vel_base - 5 + random.randint(-3, 3))
        hit(o + 3.5 + SWING_AMOUNT, 63, vel_base + random.randint(-3, 3))

    # Drop 1: shaker + tamb (shaker from bar 36, tamb enters at 8-bar mark = bar 44)
    for bar in range(DROP1_S, DROP1_E):
        shaker_pattern(bar, vel_base=25)
        if bar >= DROP1_S + 8:  # 8-bar rule: tamb enters halfway
            tamb_pattern(bar, vel_base=30)

    # Drop 2: full shadow perc (shaker + tamb + conga)
    for bar in range(DROP2_S, DROP2_E):
        shaker_pattern(bar, vel_base=28)
        tamb_pattern(bar, vel_base=35)
        if bar >= DROP2_S + 8:  # 8-bar rule: conga enters halfway
            conga_pattern(bar, vel_base=38)

    # Intro: light shaker only (bars 8-15 — 8-bar rule)
    for bar in range(INTRO_S + 8, INTRO_E):
        shaker_pattern(bar, vel_base=20)

    return part


# ============================================================================
# SUB BASS (MIDI for FAUST rendering — roots on beat 1, sustained)
# ============================================================================

def create_sub_bass():
    part = stream.Part()
    part.partName = 'Sub Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Active in verse + drops
    for s_bar, e_bar in [(VERSE_S, VERSE_E), (DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar):
            root = BASS_ROOTS[bar % 4]
            n = note.Note(root, quarterLength=3.5)
            n.volume.velocity = 90
            part.insert(bb(bar), n)

    return part


# ============================================================================
# MID BASS — follows chord stab rhythm (tutorial tip: bass copies chord rhythm)
# ============================================================================

def create_mid_bass():
    part = stream.Part()
    part.partName = 'Mid Bass'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    # Same rhythmic pattern as chord stab but playing root notes (shorter)
    # Stab pattern: hits at 0.0, 0.5+swing, 2.5+swing, 3.0
    # Bass copies this but with shorter durations
    for s_bar, e_bar in [(DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar):
            root = BASS_ROOTS[bar % 4]
            hits = [0.0, 0.5 + SWING_AMOUNT, 2.5 + SWING_AMOUNT, 3.0]
            durs = [0.4, 0.4, 0.4, 0.8]
            vels = [88, 55, 55, 75]
            for h, d, v in zip(hits, durs, vels):
                n = note.Note(root, quarterLength=d)
                n.volume.velocity = v
                part.insert(bb(bar, h), n)

    return part


# ============================================================================
# CHORD STAB — rhythmic chord hits in drops (replaces piano)
# ============================================================================

def create_chord_stab():
    part = stream.Part()
    part.partName = 'Chord Stab'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def rhythmic_stab(s_bar, e_bar, vel=65):
        """Off-beat chord stab pattern with swing."""
        for bar in range(s_bar, e_bar):
            tones = CHORDS[bar % 4]
            # House stab pattern: hit on 1, &of1 (swung), &of2 (swung), 3
            hits = [0.0, 0.5 + SWING_AMOUNT, 2.5 + SWING_AMOUNT, 3.0]
            durs = [0.4, 0.4, 0.4, 0.8]
            for h, d in zip(hits, durs):
                c = chord.Chord(tones, quarterLength=d)
                c.volume.velocity = vel + random.randint(-4, 4)
                part.insert(bb(bar, h), c)

    # Drop 1: chord stabs
    rhythmic_stab(DROP1_S, DROP1_E, vel=62)

    # Drop 2: chord stabs (slightly louder)
    rhythmic_stab(DROP2_S, DROP2_E, vel=68)

    return part


# ============================================================================
# LEAD MELODY
# ============================================================================

def create_lead():
    part = stream.Part()
    part.partName = 'Lead'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(40, 55)):
        # Hook octave pattern: [base, base, +1, +1] per workflow.md MANDATORY
        for bar in range(s_bar, e_bar):
            bar_idx = bar % 4
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

    # Drop 1: lead enters 8 bars in (staggered entry — 8-bar rule)
    write_phrase(DROP1_S + 8, DROP1_E, vel_range=(50, 65))

    # Breakdown: quiet, filtered melody
    write_phrase(BREAK_S, BREAK_E, vel_range=(35, 45))

    # Buildup 2: lead continues through
    write_phrase(BUILD2_S, BUILD2_E, vel_range=(48, 58))

    # Drop 2: lead from the start (hits harder)
    write_phrase(DROP2_S, DROP2_E, vel_range=(55, 70))

    return part


# ============================================================================
# LEAD2 — counter-melody (answer notes filling lead's rests), both drops
# ============================================================================

def create_lead2():
    part = stream.Part()
    part.partName = 'Lead2'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    def write_phrase(s_bar, e_bar, vel_range=(32, 45)):
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

    # Drop 1: counter-melody enters 16 bars in (staggered — after lead)
    write_phrase(DROP1_S + 16, DROP1_E, vel_range=(32, 42))

    # Drop 2: counter-melody enters 8 bars in
    write_phrase(DROP2_S + 8, DROP2_E, vel_range=(35, 48))

    return part


# ============================================================================
# PLUCK — rhythmic chord melody in drops (same as stab rhythm, melody notes)
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

    # Drop 1: pluck enters 16 bars in (staggered with lead + lead2)
    write_phrase(DROP1_S + 16, DROP1_E, vel_range=(36, 46))
    # Drop 2: pluck enters 8 bars in
    write_phrase(DROP2_S + 8, DROP2_E, vel_range=(38, 48))

    return part


# ============================================================================
# PAD — sustained chords (dub chord filtering done in render)
# ============================================================================

def create_pad():
    part = stream.Part()
    part.partName = 'Pad'
    part.insert(0, tempo.MetronomeMark(number=BPM))

    sections = {
        'verse':  (VERSE_S,  VERSE_E,  45),       # Low energy: sustained pad
        'break':  (BREAK_S,  BREAK_E,  40),       # Atmospheric
        'drop1':  (DROP1_S + 16, DROP1_E, 50),    # Enters 16 bars into Drop1 (staggered)
        'drop2':  (DROP2_S + 8,  DROP2_E, 55),    # Enters 8 bars into Drop2
        'outro':  (OUTRO_S,  OUTRO_E,  35),       # Fade out
    }

    for section_key, (s_bar, e_bar, vel) in sections.items():
        for bar in range(s_bar, e_bar):
            c = chord.Chord(PAD_CHORDS[bar % 4], quarterLength=4.0)
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
    'stab':  81,
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
        if 'drum' in name or 'shadow' in name or 'perc' in name:
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
    print(f'  Key: G minor  |  BPM: {BPM}  |  128 bars')

    parts = {
        'drums':       create_drums(),
        'shadow_perc': create_shadow_perc(),
        'sub_bass':    create_sub_bass(),
        'mid_bass':    create_mid_bass(),
        'chord_stab':  create_chord_stab(),
        'lead':        create_lead(),
        'lead2':       create_lead2(),
        'pluck':       create_pluck(),
        'pad':         create_pad(),
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
