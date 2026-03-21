"""
render_beat.py -- Unified Sample-Based Beat Renderer

Single renderer for all genres (trap, boombap, jazzhouse). Genre-specific
behavior is driven by inline config dicts — no separate files needed.

Uses shared modules: audio_utils, sample_analysis, gross_beat, mix_master,
chord_detect, lofi_fx.

Usage:
  python render_beat.py --sample <path.wav> --name "MyBeat" --genre trap
  python render_beat.py --sample <path.wav> --name "MyBeat" --genre boombap --lofi 0.7
  python render_beat.py --sample <path.wav> --name "MyBeat" --genre jazzhouse
"""

import argparse
import os
import sys
import json
import numpy as np
from math import gcd
from scipy import signal
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import pedalboard as pb
import pyroomacoustics as pra
import pyloudnorm as pyln
import librosa
import glob as _glob

sys.path.insert(0, '/Users/ronantakizawa/Documents/instruments')
from audio_utils import (load_sample, place, apply_pb, bar_to_s, stereo_widen,
                         create_sample_bed, lpf_sweep, auto_gain_sample,
                         adaptive_hpf, add_metronome)
from sample_analysis import (analyze_sample_character, detect_sample_tempo,
                             detect_sample_key, detect_loop_period,
                             extract_loop_auto, extract_loop_at)
from gross_beat import apply_gross_beat
from mix_master import mix_analysis, get_version
from chord_detect import detect_chords, chords_to_bass_pattern
from lofi_fx import (lofi_process, vinyl_noise, vinyl_crackle,
                     parallel_drum_compress, generate_sub_bass)

SR = 44100
INSTR = '/Users/ronantakizawa/Documents/instruments'

# =============================================================================
# Genre Configs (inline, not JSON)
# =============================================================================

GENRE_CONFIGS = {
    'trap': {
        'bpm_default': 148, 'bpm_range': (120, 165), 'bars': 96,
        'jitter_ms': 4,
        'kick_beats': [0, 2], 'kick_ghost': 3.5, 'kick_ghost_pct': 0.25, 'kick_gain': 1.0,
        'snare_beats': [2], 'snare_ghost_pct': 0.35, 'snare_gain': 0.65,
        'clap_layer': True, 'clap_gain': 0.45,
        'hat_pattern': '16th_rolling', 'hat_gain': 0.38, 'hat_32nd_rolls': True,
        'ride': False,
        'bass_type': '808', 'bass_lpf': 1200, 'bass_gain_db': 3.0,
        'bass_comp': (-10, 4.0, 2, 80),
        'mix': {'drums': 0.70, 'bass': 0.80, 'sample': 0.40, 'perc': 0.15, 'fx': 0.22, 'vinyl': 0.0},
        'sc_depth': 0.75,
        'master': {'hpf': 25, 'lpf': 18000, 'comp_thresh': -14, 'comp_ratio': 2.0},
        'sample_hpf': (150, 220),
        'arrangement': [
            ('intro', 0, 8, False, 'lpf_sweep_up'),
            ('hook1', 8, 24, True, 'full'),
            ('verse', 24, 40, True, 'slight_lpf'),
            ('hook2', 40, 56, True, 'full'),
            ('bridge', 56, 64, False, 'underwater'),
            ('hook3', 64, 80, True, 'full'),
            ('outro', 80, 96, True, 'lpf_sink'),
        ],
        'lofi': False, 'drum_break': False,
        'room': (5.0, 4.0, 3.5, 0.35, 2, 0.15, 0.06),
        'album': 'Trap Beats', 'genre_tag': 'Trap',
    },
    'boombap': {
        'bpm_default': 90, 'bpm_range': (70, 110), 'bars': 64,
        'jitter_ms': 20,
        'kick_beats': [0, 2], 'kick_ghost': 3.5, 'kick_ghost_pct': 0.30, 'kick_gain': 0.70,
        'snare_beats': [1, 3], 'snare_ghost_pct': 0.40, 'snare_gain': 0.60,
        'clap_layer': False, 'clap_gain': 0.0,
        'hat_pattern': '8th', 'hat_gain': 0.30, 'hat_32nd_rolls': False,
        'ride': False,
        'bass_type': 'sine_sub', 'bass_lpf': 200, 'bass_gain_db': 3.0,
        'bass_comp': (-12, 3.0, 5, 100),
        'mix': {'drums': 0.75, 'bass': 0.55, 'sample': 0.45, 'perc': 0.12, 'fx': 0.0, 'vinyl': 0.03},
        'sc_depth': 0.60,
        'master': {'hpf': 30, 'lpf': 16000, 'comp_thresh': -12, 'comp_ratio': 2.5},
        'sample_hpf': (120, 200),
        'arrangement': [
            ('intro', 0, 8, False, 'lpf_sweep_up'),
            ('hook1', 8, 24, True, 'full'),
            ('verse', 24, 40, True, 'slight_lpf'),
            ('hook2', 40, 56, True, 'full'),
            ('outro', 56, 64, False, 'lpf_sink'),
        ],
        'lofi': True, 'lofi_intensity': 0.5, 'drum_break': True,
        'room': (6.0, 5.0, 3.0, 0.25, 3, 0.25, 0.12),
        'album': 'Old School Beats', 'genre_tag': 'Hip-Hop',
    },
    'jazzhouse': {
        'bpm_default': 122, 'bpm_range': (115, 130), 'bars': 80,
        'jitter_ms': 3,
        'kick_beats': [0, 1, 2, 3], 'kick_ghost': None, 'kick_ghost_pct': 0.0, 'kick_gain': 0.55,
        'snare_beats': [1, 3], 'snare_ghost_pct': 0.30, 'snare_gain': 0.40,
        'clap_layer': False, 'clap_gain': 0.0,
        'hat_pattern': '16th_quiet', 'hat_gain': 0.13, 'hat_32nd_rolls': False,
        'ride': True, 'ride_gain': 0.18,
        'bass_type': 'sine_sub', 'bass_lpf': 180, 'bass_gain_db': 2.0,
        'bass_comp': (-14, 2.5, 8, 120),
        'mix': {'drums': 0.65, 'bass': 0.15, 'sample': 0.50, 'perc': 0.0, 'fx': 0.0, 'vinyl': 0.02},
        'sc_depth': 0.40,
        'master': {'hpf': 25, 'lpf': 18000, 'comp_thresh': -14, 'comp_ratio': 2.0},
        'sample_hpf': (100, 180),
        'arrangement': [
            ('intro', 0, 8, False, 'lpf_sweep_up'),
            ('laid_back1', 8, 24, False, 'full'),
            ('upbeat1', 24, 40, True, 'full'),
            ('laid_back2', 40, 48, False, 'slight_lpf'),
            ('upbeat2', 48, 64, True, 'full'),
            ('outro', 64, 80, False, 'lpf_sink'),
        ],
        'lofi': False, 'drum_break': False,
        'room': (8.0, 6.0, 3.5, 0.20, 4, 0.40, 0.15),
        'album': 'Jazz House Beats', 'genre_tag': 'House',
    },
    'melodic_trap': {
        'bpm_default': 140, 'bpm_range': (130, 150), 'bars': 96,
        'jitter_ms': 4,
        'kick_beats': [0, 2], 'kick_ghost': 3.5, 'kick_ghost_pct': 0.25, 'kick_gain': 1.0,
        'snare_beats': [1, 3], 'snare_ghost_pct': 0.0, 'snare_gain': 0.0,
        'clap_layer': True, 'clap_gain': 0.50, 'clap_offset_ms': 25,
        'hat_pattern': '8th', 'hat_gain': 0.32, 'hat_32nd_rolls': False,
        'ride': False,
        'bass_type': '808', 'bass_lpf': 1200, 'bass_gain_db': 3.0,
        'bass_dur_beats': 3.5,
        'bass_comp': (-10, 4.0, 2, 80),
        'mix': {'drums': 0.65, 'bass': 0.75, 'sample': 0.50, 'perc': 0.12, 'fx': 0.20, 'vinyl': 0.0},
        'sc_depth': 0.70,
        'master': {'hpf': 25, 'lpf': 19000, 'comp_thresh': -14, 'comp_ratio': 2.0},
        'sample_hpf': (140, 210),
        'arrangement': [
            ('intro', 0, 8, False, 'lpf_sweep_up'),
            ('hook1', 8, 24, True, 'full'),
            ('verse1', 24, 40, True, 'slight_lpf'),
            ('hook2', 40, 56, True, 'full'),
            ('bridge', 56, 60, False, 'underwater'),
            ('hook3', 60, 80, True, 'full'),
            ('outro', 80, 96, True, 'lpf_sink'),
        ],
        'lofi': False, 'drum_break': False,
        'room': (6.0, 5.0, 3.5, 0.25, 3, 0.25, 0.10),
        'album': 'Melodic Trap Beats', 'genre_tag': 'Trap',
    },
    'breakcore': {
        'bpm_default': 100, 'bpm_range': (85, 120), 'bars': 72,
        'jitter_ms': 2,
        'drum_mode': 'amen_chop',
        'kick_beats': [], 'kick_ghost': None, 'kick_ghost_pct': 0, 'kick_gain': 0,
        'snare_beats': [], 'snare_ghost_pct': 0, 'snare_gain': 0,
        'clap_layer': False, 'clap_gain': 0,
        'hat_pattern': 'none', 'hat_gain': 0, 'hat_32nd_rolls': False,
        'ride': False,
        'bass_type': 'sine_sub', 'bass_lpf': 200, 'bass_gain_db': 1.0,
        'bass_comp': (-10, 3.0, 3, 80),
        'mix': {'drums': 0.85, 'bass': 0.20, 'sample': 0.30, 'perc': 0.0, 'fx': 0.0, 'vinyl': 0.0},
        'sc_depth': 0.50,
        'master': {'hpf': 30, 'lpf': 18000, 'comp_thresh': -10, 'comp_ratio': 3.0},
        'sample_hpf': (200, 280),
        'arrangement': [
            ('intro', 0, 8, False, 'lpf_sweep_up'),
            ('drop1', 8, 24, True, 'full'),
            ('break', 24, 32, False, 'underwater'),
            ('drop2', 32, 48, True, 'full'),
            ('break2', 48, 56, False, 'slight_lpf'),
            ('drop3', 56, 68, True, 'full'),
            ('outro', 68, 72, False, 'lpf_sink'),
        ],
        'lofi': True, 'lofi_intensity': 0.7, 'drum_break': False,
        'target_lufs': -12,
        'room': (4.0, 3.0, 2.5, 0.40, 2, 0.10, 0.04),
        'album': 'Breakcore Beats', 'genre_tag': 'Breakcore',
    },
}


# =============================================================================
# Kit Selection (per-genre, kept inline)
# =============================================================================

VIRION = os.path.join(INSTR, 'VIRION - BLESSDEEKIT [JERK DRUMKIT]')
MODTRAP = os.path.join(INSTR, 'Obie - ALL GENRE KIT PT 2 ', '1. TRAP_NEWAGE_ETC', 'MODERN TRAP')
METRO808 = os.path.join(INSTR, 'Metro Boomin - #MetroWay Sound Kit [Nexus XP]', '808s')
RAP2_808 = os.path.join(INSTR, 'rap2', '808s')
OSHH = os.path.join(INSTR, 'oldschoolhiphop')
BROOTLE = os.path.join(INSTR, 'Studio Brootle Free House Kick Sample Pack')
HOUSE_OS = os.path.join(INSTR, 'house', 'One Shots')
CYM_HOUSE = os.path.join(INSTR, 'Cymatics - House - Starter Pack', 'Drums', 'Drum One Shots')


def _glob_wavs(*patterns):
    result = []
    for p in patterns:
        result.extend(sorted(_glob.glob(p)))
    return result


def select_kit(genre, char, track_name):
    rng = np.random.RandomState(hash(track_name) % (2**31))

    if genre == 'trap':
        kicks = _glob_wavs(f'{VIRION}/Kick/*.wav', f'{MODTRAP}/Kicks/*.wav')
        bass_808s = _glob_wavs(f'{VIRION}/808/*.wav', f'{METRO808}/*.wav', f'{RAP2_808}/*.wav')
        snares = _glob_wavs(f'{VIRION}/Snare/*.wav', f'{MODTRAP}/Snares/*.wav')
        claps = _glob_wavs(f'{MODTRAP}/Claps/*.wav')
        hats = _glob_wavs(f'{VIRION}/Hi-Hat/*.wav', f'{MODTRAP}/Closed Hats/*.wav')
        ohs = _glob_wavs(f'{VIRION}/Open Hat/*.wav', f'{MODTRAP}/Open Hats/*.wav')
        percs = _glob_wavs(f'{VIRION}/Perc/*.wav', f'{MODTRAP}/Percs/*.wav')
        crashes = _glob_wavs(f'{VIRION}/Crash/*.wav')
        return {
            'kick': kicks[rng.randint(len(kicks))] if kicks else None,
            'bass_808': bass_808s[rng.randint(len(bass_808s))] if bass_808s else None,
            'snare': snares[rng.randint(len(snares))] if snares else None,
            'clap': claps[rng.randint(len(claps))] if claps else None,
            'hat': hats[rng.randint(len(hats))] if hats else None,
            'hat_open': ohs[rng.randint(len(ohs))] if ohs else None,
            'perc': percs[rng.randint(len(percs))] if percs else None,
            'crash': crashes[0] if crashes else None,
        }

    elif genre == 'melodic_trap':
        kicks = _glob_wavs(f'{VIRION}/Kick/*.wav', f'{MODTRAP}/Kicks/*.wav')
        bass_808s = _glob_wavs(f'{VIRION}/808/*.wav', f'{METRO808}/*.wav', f'{RAP2_808}/*.wav')
        snares = _glob_wavs(f'{VIRION}/Snare/*.wav', f'{MODTRAP}/Snares/*.wav')
        claps = _glob_wavs(f'{MODTRAP}/Claps/*.wav')
        hats = _glob_wavs(f'{VIRION}/Hi-Hat/*.wav', f'{MODTRAP}/Closed Hats/*.wav')
        ohs = _glob_wavs(f'{VIRION}/Open Hat/*.wav', f'{MODTRAP}/Open Hats/*.wav')
        percs = _glob_wavs(f'{VIRION}/Perc/*.wav', f'{MODTRAP}/Percs/*.wav')
        crashes = _glob_wavs(f'{VIRION}/Crash/*.wav')
        return {
            'kick': kicks[rng.randint(len(kicks))] if kicks else None,
            'bass_808': bass_808s[rng.randint(len(bass_808s))] if bass_808s else None,
            'snare': snares[rng.randint(len(snares))] if snares else None,
            'clap': claps[rng.randint(len(claps))] if claps else None,
            'hat': hats[rng.randint(len(hats))] if hats else None,
            'hat_open': ohs[rng.randint(len(ohs))] if ohs else None,
            'perc': percs[rng.randint(len(percs))] if percs else None,
            'crash': crashes[0] if crashes else None,
        }

    elif genre == 'boombap':
        kicks = _glob_wavs(f'{OSHH}/One Shots/Kicks/*.wav')
        snares = _glob_wavs(f'{OSHH}/One Shots/Snares_Rims_Claps/*.wav')
        hats = _glob_wavs(f'{OSHH}/One Shots/Hats/*.wav')
        percs = _glob_wavs(f'{OSHH}/One Shots/Perc/*.wav')
        hat2 = hats[rng.randint(len(hats))]
        oh = hats[rng.randint(len(hats))]
        while oh == hat2 and len(hats) > 1:
            oh = hats[rng.randint(len(hats))]
        return {
            'kick': kicks[rng.randint(len(kicks))] if kicks else None,
            'snare': snares[rng.randint(len(snares))] if snares else None,
            'hat': hat2, 'hat_open': oh,
            'perc': percs[rng.randint(len(percs))] if percs else None,
        }

    elif genre == 'jazzhouse':
        kicks = _glob_wavs(f'{BROOTLE}/*.wav', f'{HOUSE_OS}/Kicks/*.wav')
        rides = _glob_wavs(f'{CYM_HOUSE}/Cymbals/Rides/*.wav')
        claps = _glob_wavs(f'{HOUSE_OS}/Claps/*.wav', f'{CYM_HOUSE}/Claps/*.wav')
        hh_cl = _glob_wavs(f'{CYM_HOUSE}/Cymbals/Hihats - Closed/*.wav')
        hh_op = _glob_wavs(f'{CYM_HOUSE}/Cymbals/Hihats - Open/*.wav')
        percs = _glob_wavs(f'{CYM_HOUSE}/Percussion/*.wav')
        return {
            'kick': kicks[rng.randint(len(kicks))] if kicks else None,
            'ride': rides[rng.randint(len(rides))] if rides else None,
            'clap': claps[rng.randint(len(claps))] if claps else None,
            'hat': hh_cl[rng.randint(len(hh_cl))] if hh_cl else None,
            'hat_open': hh_op[rng.randint(len(hh_op))] if hh_op else None,
            'perc': percs[rng.randint(len(percs))] if percs else None,
        }

    elif genre == 'breakcore':
        # Breakcore uses chopped amen, no traditional kit needed
        return {'kick': None, 'snare': None, 'hat': None, 'hat_open': None, 'perc': None}

    raise ValueError(f'Unknown genre: {genre}')


def select_drum_break(track_name, target_bpm):
    """Pick and time-stretch a drum break for boom-bap."""
    rng = np.random.RandomState(hash(track_name + '_break') % (2**31))
    loops = sorted(_glob.glob(os.path.join(OSHH, 'Loops', 'Drum Loops', '*.wav')))
    if not loops:
        return None

    scored = []
    for lp in loops:
        fname = os.path.basename(lp)
        bpm_val = 90
        for part in fname.replace('_', ' ').split():
            if 'BPM' in part:
                try: bpm_val = int(part.replace('BPM', ''))
                except ValueError: pass
        if bpm_val > 130:
            bpm_val /= 2
        scored.append((abs(bpm_val - target_bpm), bpm_val, lp))
    scored.sort()

    pick = scored[rng.randint(min(3, len(scored)))]
    audio = load_sample(pick[2])
    if abs(pick[1] - target_bpm) > 1:
        audio = librosa.effects.time_stretch(audio, rate=target_bpm / pick[1]).astype(np.float32)
    print(f'  Break: {os.path.basename(pick[2])} ({pick[1]:.0f} → {target_bpm:.0f} BPM)')
    return audio


# =============================================================================
# Breakcore: Amen Chop Engine + Reese Bass
# =============================================================================

AMEN_PATH = os.path.join(INSTR, 'breaks', 'amen', 'amen_break.mp3')
BREAKCORE2 = os.path.join(INSTR, 'breakcore2', 'Breaks')


def load_amen_slices(target_bpm, sr=SR):
    """Load the amen break, detect beats, return individual slices time-stretched."""
    amen = load_sample(AMEN_PATH, sr)
    # Amen break is ~136-138 BPM, ~7s
    _, beat_frames = librosa.beat.beat_track(y=amen, sr=sr, start_bpm=136)
    beat_samps = librosa.frames_to_samples(beat_frames)
    # Create slices at each beat
    slices = []
    for i in range(len(beat_samps)):
        start = int(beat_samps[i])
        end = int(beat_samps[i + 1]) if i + 1 < len(beat_samps) else len(amen)
        slc = amen[start:end].copy()
        if len(slc) < 100:
            continue
        # Time-stretch slice to target BPM
        ratio = target_bpm / 136.0
        slc = librosa.effects.time_stretch(slc, rate=ratio).astype(np.float32)
        slices.append(slc)
    if not slices:
        # Fallback: chop into 16 equal parts
        chunk = len(amen) // 16
        for i in range(16):
            slc = amen[i * chunk:(i + 1) * chunk].copy()
            ratio = target_bpm / 136.0
            slc = librosa.effects.time_stretch(slc, rate=ratio).astype(np.float32)
            slices.append(slc)
    print(f'  Amen: {len(slices)} slices at {target_bpm:.0f} BPM')
    return slices


def load_extra_breaks(target_bpm, rng, n=3, sr=SR):
    """Load a few extra break loops from breakcore2 collection."""
    breaks = _glob_wavs(f'{BREAKCORE2}/*.wav', f'{BREAKCORE2}/*.WAV')
    if not breaks:
        return []
    picked = rng.choice(breaks, size=min(n, len(breaks)), replace=False)
    extras = []
    for bp in picked:
        try:
            audio = load_sample(bp, sr)
            # Rough time-stretch (assume ~140 BPM source)
            ratio = target_bpm / 140.0
            audio = librosa.effects.time_stretch(audio, rate=ratio).astype(np.float32)
            extras.append(audio)
        except Exception:
            pass
    return extras


def program_breakbeats(cfg, nbars, BAR, BEAT, NSAMP, rng):
    """Chop amen break into slices, rearrange with stutters/FX per bar.
    Returns stereo drum buffer + kick_env."""
    bpm = 60.0 / BEAT
    slices = load_amen_slices(bpm)
    extras = load_extra_breaks(bpm, rng)

    drum_L = np.zeros(NSAMP, dtype=np.float32)
    drum_R = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)

    arr = cfg['arrangement']
    sixteenth = BEAT / 4  # duration of 1/16th note in seconds
    sixteenth_samp = int(sixteenth * SR)

    def section_at(bar):
        for name, s, e, active, _ in arr:
            if s <= bar < e:
                return name, active
        return 'none', False

    for bar in range(nbars):
        sec_name, active = section_at(bar)
        if not active:
            # Sparse breaks in non-active sections (intro/break/outro)
            if sec_name in ('intro', 'outro'):
                continue
            # In "break" sections: play occasional sparse hits
            if rng.random() > 0.3:
                continue

        bar_start = int(bar * BAR * SR)

        # Fill bar with 16th note grid of amen slices
        for step in range(16):
            pos = bar_start + step * sixteenth_samp
            if pos >= NSAMP:
                break

            # Pick a random slice
            if extras and rng.random() < 0.15:
                # Occasionally use extra break
                src = extras[rng.randint(len(extras))]
                slc = src[:sixteenth_samp].copy() if len(src) > sixteenth_samp else src.copy()
            else:
                slc = slices[rng.randint(len(slices))].copy()
                # Trim to 1/16th note
                slc = slc[:sixteenth_samp] if len(slc) > sixteenth_samp else slc

            # Pad if too short
            if len(slc) < sixteenth_samp:
                padded = np.zeros(sixteenth_samp, dtype=np.float32)
                padded[:len(slc)] = slc
                slc = padded

            # === Per-slice FX (simplified — less chaos) ===
            roll = rng.random()

            if roll < 0.08:
                # Reverse (rare)
                slc = slc[::-1].copy()
            elif roll < 0.15:
                # Stutter: repeat a fragment (occasional)
                frag_len = max(128, sixteenth_samp // rng.randint(3, 6))
                frag = slc[:frag_len].copy()
                fade = min(16, frag_len // 4)
                if fade > 0:
                    frag[-fade:] *= np.linspace(1, 0, fade).astype(np.float32)
                slc = np.tile(frag, (sixteenth_samp // frag_len) + 1)[:sixteenth_samp]
            # else: play slice clean (85% of the time)

            # Velocity variation
            vel = rng.uniform(0.6, 1.0)
            # Accent on beats 1 and 3
            if step % 4 == 0:
                vel *= 1.2

            # Density control: in "break" sections, skip ~60% of steps
            if not active and rng.random() < 0.6:
                continue

            # Place
            end = min(pos + len(slc), NSAMP)
            chunk = slc[:end - pos]
            # Slight stereo variation
            pan = rng.uniform(0.35, 0.65)
            drum_L[pos:end] += chunk * vel * pan
            drum_R[pos:end] += chunk * vel * (1 - pan)

            # Kick envelope for sidechain (rough: every 4 steps)
            if step % 4 == 0:
                env_len = min(int(0.05 * SR), NSAMP - pos)
                if env_len > 0:
                    kick_env[pos:pos + env_len] = np.maximum(
                        kick_env[pos:pos + env_len],
                        np.linspace(1, 0, env_len).astype(np.float32))

    # Light saturation on the break bus (not heavy distortion)
    drum_L = np.tanh(drum_L * 1.3).astype(np.float32)
    drum_R = np.tanh(drum_R * 1.3).astype(np.float32)

    print(f'  Breakcore drums: {nbars} bars of chopped amen')

    return {
        'kick_L': drum_L, 'kick_R': drum_R, 'kick_env': kick_env,
        'snare_L': np.zeros(NSAMP, dtype=np.float32),
        'snare_R': np.zeros(NSAMP, dtype=np.float32),
        'hh_L': np.zeros(NSAMP, dtype=np.float32),
        'hh_R': np.zeros(NSAMP, dtype=np.float32),
        'ride_L': np.zeros(NSAMP, dtype=np.float32),
        'ride_R': np.zeros(NSAMP, dtype=np.float32),
        'perc_L': np.zeros(NSAMP, dtype=np.float32),
        'perc_R': np.zeros(NSAMP, dtype=np.float32),
    }


def generate_reese_bass(freq_hz, duration_s, sr=SR):
    """Reese bass: 2 detuned saws + LPF + distortion."""
    n = int(duration_s * sr)
    if n < 1:
        return np.zeros(1, dtype=np.float32)
    t = np.arange(n) / sr
    # Two detuned sawtooth waves
    saw1 = signal.sawtooth(2 * np.pi * freq_hz * t).astype(np.float32)
    saw2 = signal.sawtooth(2 * np.pi * freq_hz * 1.01 * t).astype(np.float32)
    reese = (saw1 + saw2) * 0.4
    # LPF at 800Hz
    sos = butter(4, 800, btype='low', fs=sr, output='sos')
    reese = sosfilt(sos, reese).astype(np.float32)
    # Soft distortion
    reese = np.tanh(reese * 1.8).astype(np.float32)
    # ADSR
    attack = min(int(0.005 * sr), n // 4)
    release = min(int(0.040 * sr), n // 3)
    env = np.ones(n, dtype=np.float32) * 0.85
    if attack > 0:
        env[:attack] = np.linspace(0, 0.85, attack)
    if release > 0:
        env[-release:] = np.linspace(0.85, 0, release)
    return (reese * env).astype(np.float32)


# =============================================================================
# Generic Drum Programmer
# =============================================================================

def program_drums(cfg, nbars, BAR, BEAT, NSAMP, kit, rng):
    """Program all drums from genre config. Returns dict of stereo buffers + kick_env."""
    jitter = int(cfg['jitter_ms'] / 1000 * SR)
    bufs = {}

    # Helper to check if a bar is in an active section
    arr = cfg['arrangement']
    def section_at(bar):
        for name, s, e, kick_on, fx in arr:
            if s <= bar < e:
                return name, kick_on
        return 'none', False

    def is_outro(bar):
        for name, s, e, _, _ in arr:
            if name == 'outro' and s <= bar < e:
                return (bar - s) / max(e - s - 1, 1)
        return -1

    def b2s(bar, beat=0.0):
        return bar_to_s(bar, beat, BAR)

    # --- KICK ---
    kick_L = np.zeros(NSAMP, dtype=np.float32)
    kick_R = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)
    KICK = load_sample(kit['kick']) if kit.get('kick') else None
    kick_count = 0

    if KICK is not None:
        intro_end = arr[0][2] if arr else 8
        for bar in range(nbars):
            _, kick_on = section_at(bar)
            if not kick_on:
                continue
            outro_prog = is_outro(bar)
            if outro_prog > 0.6:
                continue

            beats = list(cfg['kick_beats'])
            if cfg['kick_ghost'] and rng.random() < cfg['kick_ghost_pct']:
                beats.append(cfg['kick_ghost'])

            for beat in beats:
                pos = int(b2s(bar, beat) * SR) + rng.randint(-jitter, jitter + 1)
                pos = max(0, pos)
                vel = rng.uniform(0.85, 1.0)
                if beat == 0:
                    vel *= 1.05
                if beat == cfg.get('kick_ghost'):
                    vel *= 0.60
                g = vel * cfg['kick_gain']
                place(kick_L, kick_R, KICK, pos, g, g)
                env_len = min(int(0.07 * SR), NSAMP - pos)
                if env_len > 0 and pos + env_len <= NSAMP:
                    kick_env[pos:pos + env_len] = np.maximum(
                        kick_env[pos:pos + env_len],
                        np.linspace(1, 0, env_len).astype(np.float32))
                kick_count += 1

    print(f'  Kicks: {kick_count}')

    # --- SNARE / CLAP ---
    snare_L = np.zeros(NSAMP, dtype=np.float32)
    snare_R = np.zeros(NSAMP, dtype=np.float32)
    SNARE = load_sample(kit.get('snare') or kit.get('clap')) if kit.get('snare') or kit.get('clap') else None
    CLAP = load_sample(kit['clap']) if cfg['clap_layer'] and kit.get('clap') else None
    snare_count = 0

    if SNARE is not None:
        for bar in range(nbars):
            _, kick_on = section_at(bar)
            # Snare plays in all active sections (kick_on sections + some non-kick)
            sec_name, _ = section_at(bar)
            if sec_name in ('intro', 'none'):
                continue
            # For boombap: snare enters at bar 6
            if cfg.get('jitter_ms', 0) >= 15 and bar < 6:
                continue
            outro_prog = is_outro(bar)
            if outro_prog > 0.5:
                continue

            beats = list(cfg['snare_beats'])
            if rng.random() < cfg['snare_ghost_pct']:
                ghost = rng.choice([0.5, 1.5, 2.5, 3.5])
                beats.append(ghost)

            for beat in beats:
                pos = int(b2s(bar, beat) * SR) + rng.randint(-jitter, jitter + 1)
                pos = max(0, pos)
                vel = rng.uniform(0.80, 1.0)
                if beat not in cfg['snare_beats']:
                    vel *= 0.30
                g = vel * cfg['snare_gain']
                place(snare_L, snare_R, SNARE, pos, g, g)
                if CLAP is not None and cfg.get('clap_layer') and beat in cfg['snare_beats']:
                    clap_ms = cfg.get('clap_offset_ms', 5)
                    clap_offset = int(clap_ms / 1000.0 * SR) + rng.randint(0, max(1, int(0.002 * SR)))
                    place(snare_L, snare_R, CLAP, pos + clap_offset,
                          vel * cfg['clap_gain'] * 0.95, vel * cfg['clap_gain'] * 1.05)
                snare_count += 1

    print(f'  Snare/Clap: {snare_count}')

    # --- HI-HATS ---
    hh_L = np.zeros(NSAMP, dtype=np.float32)
    hh_R = np.zeros(NSAMP, dtype=np.float32)
    HH = load_sample(kit['hat']) if kit.get('hat') else None
    HH_OP = load_sample(kit['hat_open']) if kit.get('hat_open') else None
    hh_count, oh_count = 0, 0
    hh_jitter = min(jitter, int(0.003 * SR)) if cfg['hat_pattern'] in ('16th_quiet',) else jitter

    if HH is not None:
        for bar in range(nbars):
            sec_name, _ = section_at(bar)
            if sec_name in ('intro', 'none'):
                continue
            outro_prog = is_outro(bar)
            if outro_prog > 0.7:
                continue

            if cfg['hat_pattern'] == '8th':
                steps, step_size = 8, 0.5
            else:
                steps, step_size = 16, 0.25

            for i in range(steps):
                beat = i * step_size
                pos = int(b2s(bar, beat) * SR) + rng.randint(-hh_jitter, hh_jitter + 1)
                pos = max(0, pos)

                if cfg['hat_pattern'] == '8th':
                    vel = rng.uniform(0.55, 0.80) if i % 2 == 0 else rng.uniform(0.30, 0.55)
                elif cfg['hat_pattern'] == '16th_quiet':
                    if i % 4 == 0: vel = rng.uniform(0.35, 0.45)
                    elif i % 2 == 0: vel = rng.uniform(0.22, 0.32)
                    else: vel = rng.uniform(0.12, 0.20)
                else:  # 16th_rolling
                    if i % 4 == 0: vel = rng.uniform(0.80, 1.0)
                    elif i % 2 == 0: vel = rng.uniform(0.50, 0.70)
                    else: vel = rng.uniform(0.25, 0.45)

                g = vel * cfg['hat_gain']
                pan_l = 0.47 if i % 2 == 0 else 0.53
                place(hh_L, hh_R, HH, pos, g * pan_l / 0.5, g * (1 - pan_l) / 0.5)
                hh_count += 1

            # 32nd rolls (trap only)
            if cfg['hat_32nd_rolls'] and bar % 4 in [2, 3]:
                sec_name, kick_on = section_at(bar)
                if kick_on:
                    for i in range(8, 16):
                        beat = i * 0.25
                        for sub in [0.125]:
                            pos = int(b2s(bar, beat + sub) * SR)
                            pos = max(0, pos)
                            vel = rng.uniform(0.20, 0.40)
                            place(hh_L, hh_R, HH, pos, vel * cfg['hat_gain'] * 0.7,
                                  vel * cfg['hat_gain'] * 0.7)
                            hh_count += 1

    # Open hat accents
    if HH_OP is not None:
        oh_interval = 2 if cfg.get('ride') else rng.randint(2, 5)
        oh_beat = 1.5 if cfg.get('ride') else 3.5
        for bar in range(arr[1][1] if len(arr) > 1 else 8, nbars - 8, oh_interval):
            pos = int(b2s(bar, oh_beat) * SR) + rng.randint(-hh_jitter, hh_jitter + 1)
            pos = max(0, pos)
            place(hh_L, hh_R, HH_OP, pos, 0.10, 0.10)
            oh_count += 1

    print(f'  HH: {hh_count}  OH: {oh_count}')

    # --- RIDE (jazz house only) ---
    ride_L = np.zeros(NSAMP, dtype=np.float32)
    ride_R = np.zeros(NSAMP, dtype=np.float32)
    ride_count = 0

    if cfg.get('ride') and kit.get('ride'):
        RIDE = load_sample(kit['ride'])
        ride_jitter = int(0.003 * SR)
        for bar in range(nbars):
            sec_name, _ = section_at(bar)
            if sec_name in ('intro', 'none'):
                continue
            outro_prog = is_outro(bar)
            if outro_prog > 0.7:
                continue
            for eighth in range(8):
                beat = eighth * 0.5
                pos = int(b2s(bar, beat) * SR) + rng.randint(-ride_jitter, ride_jitter + 1)
                pos = max(0, pos)
                vel = rng.uniform(0.55, 0.65) if eighth % 2 == 0 else rng.uniform(0.40, 0.50)
                g = vel * cfg.get('ride_gain', 0.18)
                place(ride_L, ride_R, RIDE, pos, g * 0.90, g * 1.10)
                ride_count += 1
        print(f'  Ride: {ride_count}')

    # --- PERCUSSION ---
    perc_L = np.zeros(NSAMP, dtype=np.float32)
    perc_R = np.zeros(NSAMP, dtype=np.float32)
    PERC = load_sample(kit['perc']) if kit.get('perc') else None
    perc_count = 0

    if PERC is not None and cfg['mix'].get('perc', 0) > 0:
        for bar in range(nbars):
            sec_name, kick_on = section_at(bar)
            if sec_name in ('intro', 'outro', 'none'):
                continue
            if rng.random() < 0.50:
                pos = int(b2s(bar, 1.5) * SR) + rng.randint(-jitter, jitter + 1)
                pos = max(0, pos)
                vel = rng.uniform(0.25, 0.45)
                pan = rng.uniform(0.35, 0.65)
                place(perc_L, perc_R, PERC, pos, vel * pan, vel * (1 - pan))
                perc_count += 1
        print(f'  Perc: {perc_count}')

    bufs = {
        'kick_L': kick_L, 'kick_R': kick_R, 'kick_env': kick_env,
        'snare_L': snare_L, 'snare_R': snare_R,
        'hh_L': hh_L, 'hh_R': hh_R,
        'ride_L': ride_L, 'ride_R': ride_R,
        'perc_L': perc_L, 'perc_R': perc_R,
    }
    return bufs


# =============================================================================
# Generic Arrangement
# =============================================================================

def arrange_sample(sample_bed, cfg, BAR, NSAMP, name):
    """Apply section-specific FX from arrangement config."""
    samp_out = np.zeros(NSAMP, dtype=np.float32)
    arr_rng = np.random.RandomState(hash(name + '_arr') % (2**31))

    def b2s_local(bar):
        return bar_to_s(bar, 0.0, BAR)

    for sec_name, s_bar, e_bar, _, fx in cfg['arrangement']:
        s = int(b2s_local(s_bar) * SR)
        e = min(int(b2s_local(e_bar) * SR), NSAMP)
        seg_len = e - s
        if seg_len <= 0:
            continue

        if fx == 'lpf_sweep_up':
            start_cut = arr_rng.randint(250, 500)
            audio = lpf_sweep(sample_bed[s:e], SR, start_cutoff=start_cut, end_cutoff=14000)
            vol = np.linspace(0.25, 0.85, seg_len).astype(np.float32)
            samp_out[s:e] = audio * vol
        elif fx == 'full':
            samp_out[s:e] = sample_bed[s:e]
        elif fx == 'slight_lpf':
            styles = ['lpf', 'vol_dip', 'full_quiet']
            style = styles[arr_rng.randint(len(styles))]
            if style == 'lpf':
                audio = lpf_sweep(sample_bed[s:e], SR, arr_rng.randint(6000, 9000), arr_rng.randint(9000, 11000))
                samp_out[s:e] = audio * 0.85
            elif style == 'vol_dip':
                samp_out[s:e] = sample_bed[s:e] * 0.80
            else:
                samp_out[s:e] = sample_bed[s:e] * 0.90
        elif fx == 'underwater':
            mid = seg_len // 2
            first = lpf_sweep(sample_bed[s:s + mid], SR, 10000, 350)
            second = lpf_sweep(sample_bed[s + mid:e], SR, 350, 12000)
            samp_out[s:s + mid] = first * 1.1
            samp_out[s + mid:e] = second * 1.1
        elif fx == 'lpf_sink':
            audio = lpf_sweep(sample_bed[s:e], SR, 12000, 250)
            vol = np.linspace(0.85, 0.0, seg_len).astype(np.float32)
            samp_out[s:e] = audio * vol

        print(f'  {sec_name}: {fx}')

    return samp_out


# =============================================================================
# Main Render
# =============================================================================

def render(sample_path, name, genre='trap', bpm_hint=None, nbars=None,
           loop_start=None, loop_end=None, metronome=False, lofi=None,
           no_bass=False):

    cfg = GENRE_CONFIGS[genre]
    if bpm_hint is None:
        bpm_hint = cfg['bpm_default']
    if nbars is None:
        nbars = cfg['bars']

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f'{genre.title()}_Beats')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'\n=== render_beat.py — {genre.upper()} ===')
    print(f'\nStep 0: Loading & analyzing sample ...')
    raw_sample = load_sample(sample_path)
    print(f'  Raw: {len(raw_sample)/SR:.1f}s')

    char = analyze_sample_character(raw_sample, SR)
    print(f'  Centroid: {int(char["centroid"])} Hz  |  Warmth: {char["warmth"]:.2f}  |  Brightness: {char["brightness"]:.2f}')

    detected_bpm = detect_sample_tempo(raw_sample, SR, target_bpm=bpm_hint,
                                       bpm_range=cfg['bpm_range'])
    key = detect_sample_key(raw_sample, SR)
    print(f'  Detected: {detected_bpm:.1f} BPM, key={key}')

    bpm = detected_bpm
    lo, hi = cfg['bpm_range']
    for mult in [1.0, 0.5, 2.0]:
        adj = detected_bpm * mult
        if lo <= adj <= hi:
            bpm = adj
            break
    print(f'  BPM estimate: {bpm:.1f}')

    ver = get_version(output_dir, name)
    vstr = f'v{ver}'
    OUT_WAV = os.path.join(output_dir, f'{name}_{vstr}.wav')
    OUT_MP3 = os.path.join(output_dir, f'{name}_{vstr}.mp3')
    print(f'Output: {OUT_MP3}')

    # --- Loop extraction ---
    if loop_start is not None:
        loop, n_loop_bars = extract_loop_at(raw_sample, loop_start, SR, loop_end_s=loop_end)
    else:
        loop, n_loop_bars = extract_loop_auto(raw_sample, SR, target_bpm=bpm)

    # BPM sanity check
    loop_dur = len(loop) / SR
    best_bars, best_bpm, best_dist = n_loop_bars, n_loop_bars * 4 * 60.0 / loop_dur if loop_dur > 0 else bpm_hint, 999
    for try_bars in range(1, max(n_loop_bars, 1) + 1):
        try_bpm = try_bars * 4 * 60.0 / loop_dur if loop_dur > 0 else bpm_hint
        dist = abs(try_bpm - bpm_hint)
        if dist < best_dist:
            best_dist, best_bars, best_bpm = dist, try_bars, try_bpm
    if best_bars != n_loop_bars:
        print(f'  Bar count adjusted: {n_loop_bars} → {best_bars} (BPM {n_loop_bars*4*60.0/loop_dur:.0f} → {best_bpm:.0f})')
    n_loop_bars = best_bars
    bpm = best_bpm

    BEAT = 60.0 / bpm
    BAR = BEAT * 4
    SONG = nbars * BAR
    NSAMP = int((SONG + 4.0) * SR)
    print(f'  Loop: {loop_dur:.2f}s  |  Locked BPM: {bpm:.2f}  ({n_loop_bars} bars)')

    sample_bed = create_sample_bed(loop, NSAMP)

    # Auto-gain + adaptive HPF
    sample_bed, gain_db = auto_gain_sample(sample_bed)
    print(f'  Sample auto-gain: {gain_db:+.1f} dB')
    hpf_base, hpf_max = cfg['sample_hpf']
    hpf_cutoff, low_ratio = adaptive_hpf(loop, SR, baseline_hz=hpf_base, max_hz=hpf_max)
    print(f'  Low-end ratio: {low_ratio:.2f} → HPF at {hpf_cutoff} Hz')

    tmp = np.stack([sample_bed, sample_bed], axis=1)
    lpf_cut = 12000 if genre == 'boombap' else 16000
    tmp = apply_pb(tmp, pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=hpf_cutoff),
        pb.LowpassFilter(cutoff_frequency_hz=lpf_cut),
    ]))
    sample_bed = tmp[:, 0]

    # --- Chord detection ---
    print('\nStep 1: Detecting chords ...')
    chords = detect_chords(loop, SR, n_loop_bars, key=key)
    for i, ch in enumerate(chords):
        print(f'  Bar {i}: {ch["root_name"]} {ch["quality"]}')
    bass_pattern = chords_to_bass_pattern(chords, nbars)

    # --- Kit selection ---
    print(f'\nStep 2: Selecting {genre} kit ...')
    kit = select_kit(genre, char, name)
    for role, path in kit.items():
        if path:
            print(f'  {role:<10} {os.path.basename(path)}')

    rng = np.random.RandomState(hash(name) % (2**31))

    # Room IR
    rx, ry, rz, mat, order, ir_len, _ = cfg['room']
    room = pra.ShoeBox([rx, ry, rz], fs=SR, materials=pra.Material(mat), max_order=order)
    room.add_source([rx/2, ry/2, 1.5])
    room.add_microphone(np.array([[rx/2+0.2, ry/2+0.2, 1.6]]).T)
    room.compute_rir()
    room_ir = np.array(room.rir[0][0], dtype=np.float32)[:int(SR * ir_len)]
    room_ir /= (np.abs(room_ir).max() + 1e-9)

    # --- Program drums ---
    print(f'\nStep 3: Programming drums ...')
    if cfg.get('drum_mode') == 'amen_chop':
        bufs = program_breakbeats(cfg, nbars, BAR, BEAT, NSAMP, rng)
    else:
        bufs = program_drums(cfg, nbars, BAR, BEAT, NSAMP, kit, rng)

    # Drum break (boombap)
    break_buf = np.zeros((NSAMP, 2), dtype=np.float32)
    if cfg.get('drum_break'):
        break_audio = select_drum_break(name, bpm)
        if break_audio is not None:
            break_bed = create_sample_bed(break_audio, NSAMP)
            break_2ch = np.stack([break_bed, break_bed], axis=1)
            break_2ch = apply_pb(break_2ch, pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=8000)]))
            for sec_name, s, e, kick_on, _ in cfg['arrangement']:
                if kick_on:
                    ss, ee = int(bar_to_s(s, 0, BAR) * SR), min(int(bar_to_s(e, 0, BAR) * SR), NSAMP)
                    break_buf[ss:ee] = break_2ch[ss:ee] * 0.25

    # Combine drums
    print('\nStep 4: Combining drums ...')
    drum_L = bufs['kick_L'] + bufs['snare_L'] + bufs['hh_L'] + bufs['ride_L'] + bufs['perc_L']
    drum_R = bufs['kick_R'] + bufs['snare_R'] + bufs['hh_R'] + bufs['ride_R'] + bufs['perc_R']

    # Reverb (non-kick elements for jazzhouse, all for others)
    rev_wet = cfg['room'][6]
    if genre == 'jazzhouse':
        mel_L = bufs['snare_L'] + bufs['hh_L'] + bufs['ride_L'] + bufs['perc_L']
        mel_R = bufs['snare_R'] + bufs['hh_R'] + bufs['ride_R'] + bufs['perc_R']
    else:
        mel_L, mel_R = drum_L, drum_R
    rev_L = fftconvolve(mel_L, room_ir)[:NSAMP].astype(np.float32)
    rev_R = fftconvolve(mel_R, room_ir)[:NSAMP].astype(np.float32)
    drum_L += rev_L * rev_wet
    drum_R += rev_R * rev_wet

    drum_stereo = np.stack([drum_L, drum_R], axis=1).astype(np.float32)
    if genre == 'boombap':
        drum_stereo = parallel_drum_compress(drum_stereo, SR)
    else:
        drum_stereo = apply_pb(drum_stereo, pb.Pedalboard([
            pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=15, release_ms=200)]))
    drum_stereo += break_buf

    # --- Bass ---
    bass_L = np.zeros(NSAMP, dtype=np.float32)
    bass_R = np.zeros(NSAMP, dtype=np.float32)
    if no_bass:
        print('\nStep 5: Bass disabled (--no-bass)')
        bass_buf = np.zeros((NSAMP, 2), dtype=np.float32)
    else:
        print('\nStep 5: Programming bass ...')
        bass_count = 0
        for bar in range(nbars):
            sec_name, kick_on = None, False
            for sn, s, e, ko, _ in cfg['arrangement']:
                if s <= bar < e:
                    sec_name, kick_on = sn, ko
                    break
            if sec_name in ('intro', None):
                continue
            if sec_name == 'outro':
                continue
            root_midi = bass_pattern[bar]
            freq = 440.0 * (2.0 ** ((root_midi - 69) / 12.0))
            if cfg['bass_type'] == '808' and kit.get('bass_808'):
                bass_808 = load_sample(kit['bass_808'])
                bass_808_clip = bass_808[:SR]
                spec = np.abs(np.fft.rfft(bass_808_clip))
                freqs_arr = np.fft.rfftfreq(len(bass_808_clip), 1.0 / SR)
                sub_mask = (freqs_arr > 30) & (freqs_arr < 200)
                bass_root_freq = freqs_arr[sub_mask][np.argmax(spec[sub_mask])] if sub_mask.any() else 55.0
                semitones = 12 * np.log2(freq / bass_root_freq)
                from audio_utils import pitch_shift_sample
                shifted = pitch_shift_sample(bass_808, semitones)
                bass_dur = cfg.get('bass_dur_beats', 2) * BEAT
                max_len = int(bass_dur * SR)
                shifted = shifted[:max_len]
                rel = min(int(0.08 * SR), len(shifted) // 3)
                if rel > 0:
                    shifted[-rel:] *= np.linspace(1, 0, rel).astype(np.float32)
                pos = int(bar_to_s(bar, 0, BAR) * SR)
                vel = rng.uniform(0.85, 1.0)
                place(bass_L, bass_R, shifted, pos, vel, vel)
            elif cfg['bass_type'] == 'reese':
                dur = BEAT * 2
                pos = int(bar_to_s(bar, 0, BAR) * SR)
                note = generate_reese_bass(freq, dur, SR)
                vel = rng.uniform(0.70, 0.90)
                place(bass_L, bass_R, note, pos, vel, vel)
            else:
                dur = BEAT * 1.5
                pos = int(bar_to_s(bar, 0, BAR) * SR)
                note = generate_sub_bass(freq, dur, SR)
                vel = rng.uniform(0.75, 0.90)
                place(bass_L, bass_R, note, pos, vel, vel)
            bass_count += 1
        print(f'  Bass hits: {bass_count}')
        bass_buf = np.stack([bass_L, bass_R], axis=1)
        ct, cr, ca, crel = cfg['bass_comp']
        bass_buf = apply_pb(bass_buf, pb.Pedalboard([
            pb.LowpassFilter(cutoff_frequency_hz=cfg['bass_lpf']),
            pb.Compressor(threshold_db=ct, ratio=cr, attack_ms=ca, release_ms=crel),
            pb.Gain(gain_db=cfg['bass_gain_db']),
        ]))

    # --- Sidechain ---
    print('\nStep 6: Sidechain ...')
    sc_env = bufs['kick_env'][:NSAMP]
    sc_depth = cfg['sc_depth']
    sc_hard = np.clip(1.0 - sc_env * sc_depth, 1.0 - sc_depth, 1.0).astype(np.float32)

    # --- Arrange sample ---
    print('\nStep 7: Arranging sample ...')
    samp_out = arrange_sample(sample_bed, cfg, BAR, NSAMP, name)

    # --- Gross Beat FX ---
    print('\nStep 8: Gross Beat FX ...')
    transition_bars = []
    for sec_name, s, e, _, _ in cfg['arrangement']:
        if sec_name not in ('intro',):
            transition_bars.append(e - 1)
    # Remove duplicates and last bar
    transition_bars = sorted(set(b for b in transition_bars if b < nbars - 1))
    samp_out, gb_log = apply_gross_beat(samp_out, SR, BAR, nbars, transition_bars, track_name=name)
    for entry in gb_log:
        print(entry)

    # Sidechain sample
    samp_out *= sc_hard

    # --- Lo-fi processing ---
    if cfg.get('lofi') or lofi is not None:
        intensity = lofi if lofi is not None else cfg.get('lofi_intensity', 0.5)
        print(f'\nStep 9: Lo-fi processing (intensity={intensity:.1f}) ...')
        samp_out = lofi_process(samp_out, SR, intensity=intensity)

    # Process sample bus
    samp_buf = np.stack([samp_out, samp_out], axis=1)
    reverb_wet = 0.08 if genre == 'boombap' else 0.12 if genre == 'jazzhouse' else 0.06
    samp_buf = apply_pb(samp_buf, pb.Pedalboard([
        pb.Compressor(threshold_db=-16, ratio=2.5, attack_ms=10, release_ms=200),
        pb.Reverb(room_size=0.30, damping=0.50, wet_level=reverb_wet, dry_level=1.0 - reverb_wet, width=0.60),
    ]))
    samp_buf = stereo_widen(samp_buf, 12)

    # Perc bus
    perc_buf = np.stack([bufs['perc_L'], bufs['perc_R']], axis=1).astype(np.float32)

    # Vinyl
    vinyl_gain = cfg['mix'].get('vinyl', 0)
    if vinyl_gain > 0:
        if genre == 'jazzhouse':
            v = vinyl_crackle(NSAMP, SR)
        else:
            v = vinyl_noise(NSAMP, SR)
        vinyl_buf = np.stack([v, v * 0.85], axis=1).astype(np.float32)
    else:
        vinyl_buf = np.zeros((NSAMP, 2), dtype=np.float32)

    # --- Mix ---
    print('\nStep 10: Mixing ...')
    mx = cfg['mix']
    mix = (drum_stereo * mx['drums'] +
           bass_buf * mx['bass'] +
           samp_buf * mx['sample'] +
           perc_buf * mx.get('perc', 0) +
           vinyl_buf * mx.get('vinyl', 0))

    if metronome:
        mix = add_metronome(mix, len(loop), n_loop_bars, NSAMP)

    print(f'  Mix peak: {np.abs(mix).max():.3f}')

    # --- Master ---
    print('\nStep 11: Master chain ...')
    mc = cfg['master']
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=mc['hpf']),
        pb.LowpassFilter(cutoff_frequency_hz=mc['lpf']),
        pb.Compressor(threshold_db=mc['comp_thresh'], ratio=mc['comp_ratio'],
                      attack_ms=20, release_ms=250),
        pb.Gain(gain_db=-1.0),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix = mix[:trim]

    # Fade out
    fade_start = int(bar_to_s(nbars - 4, 0, BAR) * SR)
    fade_len = trim - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:trim, 0] *= fade_curve
        mix[fade_start:trim, 1] *= fade_curve

    # --- LUFS ---
    print('\nStep 12: LUFS normalization ...')
    meter = pyln.Meter(SR, block_size=0.400)
    # Measure on first full-energy section
    measure_bar = None
    for sec_name, s, e, kick_on, _ in cfg['arrangement']:
        if kick_on:
            measure_bar = (s, e)
            break
    if measure_bar is None:
        measure_bar = (8, 24)
    ms = int(bar_to_s(measure_bar[0], 0, BAR) * SR)
    me = int(bar_to_s(measure_bar[1], 0, BAR) * SR)
    lufs = meter.integrated_loudness(mix[ms:me])
    print(f'  Pre-norm LUFS: {lufs:.1f}')
    if np.isfinite(lufs):
        target_lufs = cfg.get('target_lufs', -14.0)
        gain = target_lufs - lufs
        mix = mix * (10 ** (gain / 20.0))
        print(f'  Applied {gain:+.1f} dB')
    peak = np.abs(mix).max()
    if peak > 0.95:
        print(f'  Peak: {peak:.3f} — soft clipping')
        mix = np.tanh(mix * (1.0 / peak) * 1.2) * 0.95
    else:
        print(f'  Peak: {peak:.3f} — clean')

    # --- Export ---
    print('\nStep 13: Exporting ...')
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title': f'{name} {vstr}', 'artist': 'Claude Code',
        'album': cfg['album'], 'genre': cfg['genre_tag'],
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # --- Analysis ---
    def b2s_fn(bar):
        return bar_to_s(bar, 0, BAR)
    sections = [(sn, s, e) for sn, s, e, _, _ in cfg['arrangement']]
    mix_analysis(mix, name, sections, b2s_fn)

    print(f'\nDone!  ->  {OUT_MP3}')
    return OUT_MP3


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified beat renderer')
    parser.add_argument('--sample', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--genre', required=True, choices=list(GENRE_CONFIGS.keys()))
    parser.add_argument('--bpm', type=float, default=None)
    parser.add_argument('--bars', type=int, default=None)
    parser.add_argument('--loop-start', type=float, default=None)
    parser.add_argument('--loop-end', type=float, default=None)
    parser.add_argument('--lofi', type=float, default=None)
    parser.add_argument('--metronome', action='store_true')
    parser.add_argument('--no-bass', action='store_true', help='Disable bass')
    args = parser.parse_args()
    render(args.sample, args.name, args.genre, args.bpm, args.bars,
           args.loop_start, args.loop_end, args.metronome, args.lofi,
           no_bass=args.no_bass)
