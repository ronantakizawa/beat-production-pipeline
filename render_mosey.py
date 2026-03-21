"""
Lil Mosey Type Beat -- Render Script
Renders MIDI from compose_mosey.py using GarageBand sampler instruments.

MIDI track indices in *_FULL.mid:
  0: tempo/metadata  1: flute  2: harp
  3: piano           4: glockenspiel
  5: drums           6: 808 bass

GarageBand sampler instruments:
  Flute Solo   -> /Apple GarageBand Sampler/Flute Solo/Flute_LV_na_sus_mf/
  Harp         -> /Apple GarageBand Sampler/Harp/Harp_ES_mf/
  Grand Piano  -> /Apple GarageBand Sampler/Grand Piano/
  Glockenspiel -> /Apple GarageBand Sampler/Glockenspiel/Glockenspiel_Pla_mf1/

Usage:
  python render_mosey.py --midi Mosey_Happy_FULL.mid --name "Mosey_Happy" --bpm 140
"""

import argparse
import os
import sys
import re
import json
import numpy as np
from math import gcd
from scipy import signal
from scipy.signal import fftconvolve
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import mido
import pedalboard as pb
import pyroomacoustics as pra
import glob as _glob
import pyloudnorm as pyln

SR = 44100
INST = '/Users/ronantakizawa/Documents/instruments'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/LilMosey_Beat'
GB_SAMPLER = os.path.join(INST, 'Apple GarageBand Sampler')

# Drum kit paths (same as melodic_trap in render_beat.py)
VIRION = os.path.join(INST, 'VIRION - BLESSDEEKIT [JERK DRUMKIT]')
MODTRAP = os.path.join(INST, 'Obie - ALL GENRE KIT PT 2 ', '1. TRAP_NEWAGE_ETC', 'MODERN TRAP')
METRO808 = os.path.join(INST, 'Metro Boomin - #MetroWay Sound Kit [Nexus XP]', '808s')
RAP2_808 = os.path.join(INST, 'rap2', '808s')

# Section boundaries (must match compose_mosey.py)
INTRO_S,  INTRO_E  =  0,  8
HOOK1_S,  HOOK1_E  =  8, 24
VERSE_S,  VERSE_E  = 24, 40
HOOK2_S,  HOOK2_E  = 40, 56
BRIDGE_S, BRIDGE_E = 56, 60
HOOK3_S,  HOOK3_E  = 60, 76
OUTRO_S,  OUTRO_E  = 76, 80
NBARS = 80


# ============================================================================
# HELPERS
# ============================================================================

def load_sample(path):
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0, nsamp=0):
    n = nsamp
    e = min(start_s + len(snd), n)
    if e <= start_s:
        return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R


def apply_pb(arr2ch, board):
    out = board(arr2ch.T.astype(np.float32), SR)
    return out.T.astype(np.float32)


def midi_to_hz(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))


def note_name_to_midi(name):
    """Convert e.g. 'C#4' -> MIDI 61."""
    m = re.match(r'^([A-G]#?)(\d+)$', name)
    if not m:
        return None
    note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    pc = note_map.get(m.group(1))
    if pc is None:
        return None
    octave = int(m.group(2))
    return (octave + 1) * 12 + pc


def parse_track(mid_path, track_idx):
    mid = mido.MidiFile(mid_path)
    tpb = mid.ticks_per_beat
    tempo_val = 500000
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo_val = msg.tempo
            break
    active, result = {}, []
    ticks = 0
    for msg in mid.tracks[track_idx]:
        ticks += msg.time
        t = mido.tick2second(ticks, tpb, tempo_val)
        if msg.type == 'note_on' and msg.velocity > 0:
            active[msg.note] = (t, msg.velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active:
                s, v = active.pop(msg.note)
                if t - s > 0:
                    result.append((s, msg.note, v, t - s))
    return result


def humanize_notes(notes, timing_ms=8, vel_range=5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    result = []
    jitter_samp = timing_ms / 1000.0
    for start, note_num, vel, dur in notes:
        t_jitter = rng.uniform(-jitter_samp, jitter_samp)
        v_jitter = rng.randint(-vel_range, vel_range + 1)
        result.append((max(0, start + t_jitter), note_num,
                        int(np.clip(vel + v_jitter, 1, 127)), dur))
    return result


def bar_to_s(bar, beat=0.0, BAR_DUR=0):
    return (bar + beat / 4.0) * BAR_DUR


def pan_stereo(buf, position):
    angle = (position + 1) * np.pi / 4
    result = buf.copy()
    result[:, 0] *= np.cos(angle)
    result[:, 1] *= np.sin(angle)
    return result


def stereo_widen(buf, delay_ms=12):
    d = int(delay_ms / 1000.0 * SR)
    if d <= 0 or d >= len(buf):
        return buf.copy()
    result = buf.copy()
    result[d:, 1] = buf[:-d, 1]
    result[:d, 1] *= 0.3
    return result


# ============================================================================
# GARAGEBAND SAMPLER LOADER
# ============================================================================

def scan_sampler_dir(instrument_dir):
    """Scan a GarageBand sampler directory and return {midi_note: wav_path}."""
    samples = {}
    for root, dirs, files in os.walk(instrument_dir):
        for f in files:
            if not f.lower().endswith('.wav'):
                continue
            full_path = os.path.join(root, f)

            # Pattern 1: note name at end of filename — e.g. FL1_sus_mf_A#4.wav
            m = re.search(r'_([A-G]#?\d+)\.wav$', f, re.IGNORECASE)
            if m:
                midi = note_name_to_midi(m.group(1))
                if midi is not None:
                    samples[midi] = full_path
                continue

            # Pattern 2: MIDI number prefix — e.g. 072_C4KM56_H.wav
            m2 = re.match(r'^(\d{3})_', f)
            if m2:
                # Skip pedal noise files
                if 'ped' in f.lower():
                    continue
                midi = int(m2.group(1))
                if midi not in samples:  # prefer first match (usually Hard velocity)
                    samples[midi] = full_path
    return samples


def build_sampler_pitched(sample_map, target_midis):
    """Pre-render pitch-shifted versions of samples for all target MIDI notes."""
    pitched = {}
    available = sorted(sample_map.keys())
    if not available:
        return pitched

    for target in target_midis:
        # Find nearest available sample
        nearest = min(available, key=lambda x: abs(x - target))
        snd = load_sample(sample_map[nearest])
        st = target - nearest
        if st != 0:
            board_p = pb.Pedalboard([pb.PitchShift(semitones=st)])
            snd = board_p(snd[np.newaxis, :], SR)[0].astype(np.float32)
        pitched[target] = snd
    return pitched


# ============================================================================
# STEP 0 — Fix MIDI
# ============================================================================

def fix_midi(full_mid, fixed_mid):
    print('\nStep 0: Fixing MIDI ...')
    mid = mido.MidiFile(full_mid)
    new_t0 = mido.MidiTrack()
    seen_tempo = False
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            if not seen_tempo:
                new_t0.append(mido.MetaMessage('set_tempo', tempo=msg.tempo, time=0))
                seen_tempo = True
        else:
            new_t0.append(msg)
    mid.tracks[0] = new_t0
    mid.save(fixed_mid)
    print(f'  FIXED ({len(mid.tracks)} tracks)')


# ============================================================================
# STEP 1 — Load drum samples
# ============================================================================

def load_drum_kit(beat_name):
    print('\nStep 1: Loading drum kit ...')
    rng = np.random.RandomState(hash(beat_name) % (2**31))

    def glob_wavs(*patterns):
        result = []
        for p in patterns:
            result.extend(sorted(_glob.glob(p)))
        return result

    kicks = glob_wavs(f'{VIRION}/Kick/*.wav', f'{MODTRAP}/Kicks/*.wav')
    bass_808s = glob_wavs(f'{VIRION}/808/*.wav', f'{METRO808}/*.wav', f'{RAP2_808}/*.wav')
    snares = glob_wavs(f'{VIRION}/Snare/*.wav', f'{MODTRAP}/Snares/*.wav')
    claps = glob_wavs(f'{MODTRAP}/Claps/*.wav')
    hats = glob_wavs(f'{VIRION}/Hi-Hat/*.wav', f'{MODTRAP}/Closed Hats/*.wav')
    ohs = glob_wavs(f'{VIRION}/Open Hat/*.wav', f'{MODTRAP}/Open Hats/*.wav')
    crashes = glob_wavs(f'{VIRION}/Crash/*.wav')

    kit = {}
    for name, pool in [('kick', kicks), ('bass_808', bass_808s), ('snare', snares),
                        ('clap', claps), ('hat', hats), ('hat_open', ohs), ('crash', crashes)]:
        if pool:
            kit[name] = load_sample(pool[rng.randint(len(pool))])
            print(f'  {name}: {os.path.basename(pool[0])}')
        else:
            kit[name] = np.zeros(int(SR * 0.1), dtype=np.float32)
    return kit


# ============================================================================
# STEP 2 — Build Drums
# ============================================================================

def build_drums(drum_events, kit, NSAMP, BAR_DUR, rng):
    print('\nStep 2: Building drums ...')

    room = pra.ShoeBox([4.0, 3.5, 3.0], fs=SR, materials=pra.Material(0.30), max_order=2)
    room.add_source([2.0, 1.5, 1.5])
    room.add_microphone(np.array([[2.5, 2.0, 1.8]]).T)
    room.compute_rir()
    room_ir = np.array(room.rir[0][0], dtype=np.float32)
    room_ir = room_ir[:int(SR * 0.18)]
    room_ir /= (np.abs(room_ir).max() + 1e-9)

    kick_L = np.zeros(NSAMP, dtype=np.float32)
    kick_R = np.zeros(NSAMP, dtype=np.float32)
    nk_L = np.zeros(NSAMP, dtype=np.float32)
    nk_R = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)

    KICK, CLAP, SNARE = kit['kick'], kit['clap'], kit['snare']
    HH_CL, HH_OP, CRASH = kit['hat'], kit['hat_open'], kit['crash']

    samples = {
        36: (KICK,  1.10, 'kick'),
        39: (CLAP,  0.90, 'clap'),
        38: (SNARE, 0.65, 'snare'),
        42: (HH_CL, 0.48, 'hh'),
        46: (HH_OP, 0.70, 'oh'),
        49: (CRASH, 0.55, 'crash'),
    }

    MAX_JITTER = int(0.005 * SR)
    pan_toggle = False

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        # Skip drums in intro and bridge
        if sec < bar_to_s(INTRO_E, BAR_DUR=BAR_DUR):
            continue
        if bar_to_s(BRIDGE_S, BAR_DUR=BAR_DUR) <= sec < bar_to_s(BRIDGE_E, BAR_DUR=BAR_DUR):
            continue

        g = vel / 127.0
        if note_num not in samples:
            continue
        snd_raw, gain, label = samples[note_num]

        if note_num == 36:  # Kick — centered, punchy
            snd = snd_raw * g * gain
            chunk = snd[:min(len(snd), NSAMP - bs)]
            e = bs + len(chunk)
            kick_env[bs:e] += np.abs(chunk)
            kick_L[bs:e] += chunk * 0.96
            kick_R[bs:e] += chunk * 0.96
        elif note_num == 42:  # Hi-hat — stereo panned
            pan_toggle = not pan_toggle
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s = int(np.clip(bs + jitter, 0, NSAMP - 1))
            v = g * rng.uniform(0.68, 1.00)
            snd = snd_raw * v * gain
            pr = 0.62 if pan_toggle else 0.38
            e = min(s + len(snd), NSAMP)
            ch = snd[:e - s]
            nk_L[s:e] += ch * (1 - pr) * 2
            nk_R[s:e] += ch * pr * 2
        elif note_num in (46, 49):  # Open hat, crash — slight pan
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56, nsamp=NSAMP)
        else:  # Clap, snare
            snd = snd_raw * g * rng.uniform(0.90, 1.10) * gain
            place(nk_L, nk_R, snd, bs, 0.50, 0.50, nsamp=NSAMP)

    drum_L = kick_L + nk_L
    drum_R = kick_R + nk_R
    dl_room = fftconvolve(drum_L, room_ir, mode='full')[:NSAMP]
    dr_room = fftconvolve(drum_R, room_ir, mode='full')[:NSAMP]
    drum_L = drum_L * 0.92 + dl_room * 0.06
    drum_R = drum_R * 0.92 + dr_room * 0.06

    drum_stereo = np.stack([drum_L, drum_R], axis=1)
    drum_board = pb.Pedalboard([
        pb.Compressor(threshold_db=-10, ratio=4.0, attack_ms=2, release_ms=80),
        pb.Gain(gain_db=2.0),
        pb.Limiter(threshold_db=-0.8),
    ])
    drum_stereo = apply_pb(drum_stereo, drum_board)
    print(f'  {len(drum_events)} drum events')
    return drum_stereo, kick_env


# ============================================================================
# STEP 3 — Sidechain
# ============================================================================

def build_sidechain(kick_env, NSAMP):
    print('\nStep 3: Sidechain ...')
    smooth = int(SR * 0.010)
    sc_env = np.convolve(kick_env, np.ones(smooth) / smooth, mode='same')
    sc_env /= sc_env.max() + 1e-9
    sc_gain = np.clip(1.0 - sc_env * 0.55, 0.45, 1.0)
    return sc_gain


# ============================================================================
# STEP 4 — 808 Bass (pitch-shifted sample)
# ============================================================================

def build_808(bass_notes, kit, sc_gain, NSAMP, BEAT, BAR_DUR, rng):
    print('\nStep 4: Building 808 bass ...')
    BASS_808 = kit['bass_808']

    # Detect root pitch of 808 sample via FFT
    _fft = np.abs(np.fft.rfft(BASS_808 * np.hanning(len(BASS_808))))
    _freq = np.fft.rfftfreq(len(BASS_808), 1 / SR)
    _mask = (_freq > 20) & (_freq < 500)
    _peak = _freq[_mask][np.argmax(_fft[_mask])]
    root_midi = int(round(12 * np.log2(_peak / 440) + 69))
    print(f'  808 root: MIDI {root_midi} ({_peak:.1f} Hz)')

    target_midis = sorted(set(n[1] for n in bass_notes))
    pitched = {}
    for t in target_midis:
        st = t - root_midi
        if st == 0:
            pitched[t] = BASS_808.copy()
        else:
            board_p = pb.Pedalboard([pb.PitchShift(semitones=st)])
            pitched[t] = board_p(BASS_808[np.newaxis, :], SR)[0].astype(np.float32)

    bass_L = np.zeros(NSAMP, dtype=np.float32)
    bass_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, midi_note, vel, dur_sec in bass_notes:
        s = int(sec * SR)
        if s >= NSAMP:
            continue
        if bar_to_s(BRIDGE_S, BAR_DUR=BAR_DUR) <= sec < bar_to_s(BRIDGE_E, BAR_DUR=BAR_DUR):
            continue
        target = min(pitched.keys(), key=lambda t: abs(t - midi_note))
        snd = pitched[target]
        g = (vel / 127.0) * rng.uniform(0.92, 1.08)
        max_dur = min(dur_sec, 3 * BEAT)
        trim = min(int(max_dur * SR), len(snd))
        chunk = snd[:trim].copy() * g
        fade_n = max(1, int(SR * 0.015))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        e = min(s + len(chunk), NSAMP)
        bass_L[s:e] += chunk[:e - s] * 0.75
        bass_R[s:e] += chunk[:e - s] * 0.75

    bass_buf = np.stack([bass_L, bass_R], axis=1)
    bass_buf[:, 0] *= (sc_gain * 0.70 + 0.30)
    bass_buf[:, 1] *= (sc_gain * 0.70 + 0.30)

    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=1200),
        pb.Distortion(drive_db=3.0),
        pb.Compressor(threshold_db=-10, ratio=3.5, attack_ms=4, release_ms=130),
        pb.Gain(gain_db=1.5),
        pb.Limiter(threshold_db=-2.0),
    ])
    bass_buf = apply_pb(bass_buf, bass_board)
    print(f'  808 events={len(bass_notes)}  max={np.abs(bass_buf).max():.3f}')
    return bass_buf


# ============================================================================
# STEP 5 — Melodic Instruments (GarageBand Sampler)
# ============================================================================

def build_sampler_track(name, instrument_dir, track_notes, sc_gain, NSAMP, BEAT,
                        effects_board, mix_gain=0.40, widen_ms=0, rng=None):
    """Render a melodic track using GarageBand sampler WAVs."""
    print(f'\n  Building {name} ...')
    if rng is None:
        rng = np.random.RandomState(42)

    sample_map = scan_sampler_dir(instrument_dir)
    if not sample_map:
        print(f'    WARNING: No samples found in {instrument_dir}')
        return np.zeros((NSAMP, 2), dtype=np.float32)

    print(f'    {len(sample_map)} samples loaded')

    target_midis = sorted(set(n[1] for n in track_notes))
    pitched = build_sampler_pitched(sample_map, target_midis)

    buf_L = np.zeros(NSAMP, dtype=np.float32)
    buf_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, midi_note, vel, dur_sec in track_notes:
        s = int(sec * SR)
        if s >= NSAMP or midi_note not in pitched:
            continue
        snd = pitched[midi_note]
        g = (vel / 127.0) * rng.uniform(0.92, 1.08) * mix_gain

        # Trim to note duration + small release
        max_samp = min(int((dur_sec + 0.1) * SR), len(snd))
        chunk = snd[:max_samp].copy() * g

        # Fade out at end of note
        fade_n = max(1, min(int(SR * 0.02), len(chunk) // 4))
        if len(chunk) > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)

        e = min(s + len(chunk), NSAMP)
        buf_L[s:e] += chunk[:e - s]
        buf_R[s:e] += chunk[:e - s]

    buf = np.stack([buf_L, buf_R], axis=1)

    # Sidechain
    buf[:, 0] *= (sc_gain * 0.20 + 0.80)
    buf[:, 1] *= (sc_gain * 0.20 + 0.80)

    # Effects
    buf = apply_pb(buf, effects_board)

    # Stereo widen
    if widen_ms > 0:
        buf = stereo_widen(buf, delay_ms=widen_ms)

    print(f'    {name}: {len(track_notes)} notes  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# TRANSITION FX
# ============================================================================

def build_transition_fx(kit, NSAMP, BAR_DUR, rng):
    print('\nBuilding transition FX ...')
    fx_L = np.zeros(NSAMP, dtype=np.float32)
    fx_R = np.zeros(NSAMP, dtype=np.float32)
    SNARE = kit['snare']
    CRASH = kit['crash']

    def snare_roll(target_bar, bars_build=1.0):
        n_beats = int(bars_build * 4)
        densities = [2, 3, 4, 6, 8]
        for beat_i in range(n_beats):
            progress = beat_i / n_beats
            d_idx = min(int(progress * len(densities)), len(densities) - 1)
            n_hits = densities[d_idx]
            for h in range(n_hits):
                t = bar_to_s(target_bar - bars_build, BAR_DUR=BAR_DUR) + (beat_i + h / n_hits) * (BAR_DUR / 4)
                vel = (0.25 + 0.65 * progress) * rng.uniform(0.90, 1.10)
                s = int(t * SR)
                if 0 <= s < NSAMP:
                    place(fx_L, fx_R, SNARE * vel * 0.55, s, 0.50, 0.50, nsamp=NSAMP)

    def reverse_tail(target_bar, length_beats=3):
        beat_dur = BAR_DUR / 4
        tail_len = int(length_beats * beat_dur * SR)
        padded = np.zeros(tail_len, dtype=np.float32)
        sn_len = min(len(SNARE), tail_len)
        padded[:sn_len] = SNARE[:sn_len] * 0.5
        tail_verb = pb.Pedalboard([
            pb.Reverb(room_size=0.92, damping=0.25, wet_level=0.95,
                      dry_level=0.0, width=1.0),
        ])
        wet = tail_verb(padded[np.newaxis, :], SR)[0]
        rev = wet[::-1].copy()
        fade = int(0.04 * SR)
        rev[:fade] *= np.linspace(0, 1, fade)
        end_s = int(bar_to_s(target_bar, BAR_DUR=BAR_DUR) * SR)
        start_s = max(0, end_s - len(rev))
        chunk = rev[:end_s - start_s]
        place(fx_L, fx_R, chunk, start_s, 0.48, 0.52, nsamp=NSAMP)

    # Intro -> Hook1
    snare_roll(HOOK1_S, bars_build=1.0)
    reverse_tail(HOOK1_S, length_beats=3)
    place(fx_L, fx_R, CRASH * 0.72,
          int(bar_to_s(HOOK1_S, BAR_DUR=BAR_DUR) * SR), 0.44, 0.56, nsamp=NSAMP)

    # Hook1 -> Verse
    place(fx_L, fx_R, CRASH * 0.45,
          int(bar_to_s(VERSE_S, BAR_DUR=BAR_DUR) * SR), 0.44, 0.56, nsamp=NSAMP)

    # Verse -> Hook2
    snare_roll(HOOK2_S, bars_build=1.0)
    place(fx_L, fx_R, CRASH * 0.65,
          int(bar_to_s(HOOK2_S, BAR_DUR=BAR_DUR) * SR), 0.44, 0.56, nsamp=NSAMP)

    # Bridge -> Hook3
    snare_roll(HOOK3_S, bars_build=1.0)
    reverse_tail(HOOK3_S, length_beats=3)
    place(fx_L, fx_R, CRASH * 0.72,
          int(bar_to_s(HOOK3_S, BAR_DUR=BAR_DUR) * SR), 0.44, 0.56, nsamp=NSAMP)

    fx_buf = np.stack([fx_L, fx_R], axis=1)
    fx_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=120),
        pb.Reverb(room_size=0.55, damping=0.60, wet_level=0.18,
                  dry_level=0.92, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=3.0, attack_ms=4, release_ms=120),
        pb.Gain(gain_db=1.5),
    ])
    fx_buf = apply_pb(fx_buf, fx_board)
    print('  Transition FX ready')
    return fx_buf


# ============================================================================
# MIX
# ============================================================================

def mix_stems(stems, NSAMP):
    print('\nMixing stems ...')
    mix = np.zeros((NSAMP, 2), dtype=np.float32)
    for buf, coeff, mode, param in stems:
        if mode == 'center':
            processed = pan_stereo(buf, 0.0)
        elif mode == 'widen':
            processed = stereo_widen(buf, delay_ms=param)
        elif mode == 'pan':
            processed = pan_stereo(buf, param)
        else:
            processed = buf.copy()
        mix += processed * coeff
    return mix


# ============================================================================
# MASTER + EXPORT
# ============================================================================

def master_and_export(mix, NSAMP, BEAT, BAR_DUR, beat_name, version_str,
                      target_lufs=-14.0):
    sections = [
        ('Intro',  INTRO_S,  INTRO_E),
        ('Hook 1', HOOK1_S,  HOOK1_E),
        ('Verse',  VERSE_S,  VERSE_E),
        ('Hook 2', HOOK2_S,  HOOK2_E),
        ('Bridge', BRIDGE_S, BRIDGE_E),
        ('Hook 3', HOOK3_S,  HOOK3_E),
        ('Outro',  OUTRO_S,  OUTRO_E),
    ]

    OUT_WAV = os.path.join(OUTPUT_DIR, f'{beat_name}_{version_str}.wav')
    OUT_MP3 = os.path.join(OUTPUT_DIR, f'{beat_name}_{version_str}.mp3')

    print('\nMaster chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=19000),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=15, release_ms=200),
        pb.Distortion(drive_db=1.0),
        pb.Gain(gain_db=-1.0),
    ])
    mix = apply_pb(mix, master_board)
    SONG_DUR = NBARS * BAR_DUR
    trim = int((SONG_DUR + 2.0) * SR)
    mix = mix[:trim]

    # Fade out last 4 bars
    FADE_BARS = 4
    fade_start = int(bar_to_s(NBARS - FADE_BARS, BAR_DUR=BAR_DUR) * SR)
    fade_end = trim
    fade_len = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: last {FADE_BARS} bars ({fade_len/SR:.1f}s)')

    # LUFS normalization
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(INTRO_E, BAR_DUR=BAR_DUR) * SR)
    measure_end = fade_start

    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    lufs_post_limit = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Post-limiter LUFS (body): {lufs_post_limit:.1f}')

    if np.isfinite(lufs_post_limit):
        gain_db = target_lufs - lufs_post_limit
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    lufs_final = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Final LUFS (body): {lufs_final:.1f}  (target: {target_lufs})')

    # Export
    print('Exporting ...')
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title': f'{beat_name} {version_str}',
        'artist': 'Claude Code',
        'album': 'Melodic Trap Beats',
        'genre': 'Trap',
        'comment': f'Lil Mosey Type Beat | Melodic Trap',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # Mix analysis
    print('\n== Mix Analysis ==')
    y_mono = mix.mean(axis=1).astype(np.float32)
    rms_val = np.sqrt(np.mean(y_mono ** 2))
    final_lufs = ln_meter.integrated_loudness(mix)
    peak_db = 20 * np.log10(np.abs(mix).max() + 1e-9)
    print(f'  RMS: {rms_val:.4f}  ({20*np.log10(rms_val+1e-9):.1f} dB)')
    print(f'  Integrated LUFS: {final_lufs:.1f}')
    print(f'  True peak: {peak_db:.1f} dB')

    print('\n-- Per-Section Loudness --')
    all_warnings = []
    for name, s_bar, e_bar in sections:
        s_samp = int(bar_to_s(s_bar, BAR_DUR=BAR_DUR) * SR)
        e_samp = min(int(bar_to_s(e_bar, BAR_DUR=BAR_DUR) * SR), len(mix))
        section = mix[s_samp:e_samp]
        if len(section) < SR:
            continue
        sec_lufs = ln_meter.integrated_loudness(section)
        sec_peak = 20 * np.log10(np.abs(section).max() + 1e-9)
        status = ''
        if np.isfinite(sec_lufs) and sec_lufs > target_lufs + 3:
            status = ' << TOO LOUD'
            all_warnings.append(f'{name} is {sec_lufs - target_lufs:+.1f} dB over target')
        print(f'  {name:<10} LUFS: {sec_lufs:>6.1f}  peak: {sec_peak:>6.1f} dB{status}')

    clip_count = np.sum(np.abs(mix) > 0.99)
    if clip_count > 0:
        all_warnings.append(f'Clipping: {clip_count} samples')
        print(f'  CLIP: {clip_count} samples')
    else:
        print(f'  Clipping: none')

    if all_warnings:
        print(f'\n  !! {len(all_warnings)} WARNING(S):')
        for w in all_warnings:
            print(f'     - {w}')
    else:
        print('\n  All checks passed.')

    print(f'\nDone!  ->  {OUT_MP3}')
    return mix


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lil Mosey type beat renderer')
    parser.add_argument('--midi', required=True, help='MIDI filename (in LilMosey_Beat/)')
    parser.add_argument('--name', required=True, help='Beat name')
    parser.add_argument('--bpm', required=True, type=float, help='BPM')
    args = parser.parse_args()

    BPM = min(args.bpm, 150)  # cap at 150
    BEAT = 60.0 / BPM
    BAR_DUR = BEAT * 4
    SONG_DUR = NBARS * BAR_DUR
    NSAMP = int((SONG_DUR + 4.0) * SR)
    beat_name = args.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve MIDI path
    midi_path = args.midi
    if not os.path.isabs(midi_path):
        midi_path = os.path.join(OUTPUT_DIR, midi_path)
    FIXED_MID = os.path.join(OUTPUT_DIR, f'{beat_name}_FIXED.mid')

    # Version
    existing = _glob.glob(os.path.join(OUTPUT_DIR, f'{beat_name}_v*.mp3'))
    version = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                   for p in existing], default=0) + 1
    vstr = f'v{version}'

    print(f'\n=== render_mosey.py — {beat_name} ===')
    print(f'  BPM: {BPM}  |  {NBARS} bars  |  Output: {vstr}')

    rng = np.random.RandomState(42)

    # Step 0: Fix MIDI
    fix_midi(midi_path, FIXED_MID)

    # Step 1: Load drum kit
    kit = load_drum_kit(beat_name)

    # Step 2: Drums (track 5 in full MIDI)
    drum_events = parse_track(FIXED_MID, 5)
    drum_events = humanize_notes(drum_events, timing_ms=6, vel_range=4, rng=rng)
    drum_stereo, kick_env = build_drums(drum_events, kit, NSAMP, BAR_DUR, rng)

    # Step 3: Sidechain
    sc_gain = build_sidechain(kick_env, NSAMP)

    # Step 4: 808 Bass (track 6)
    bass_notes = parse_track(FIXED_MID, 6)
    bass_notes = humanize_notes(bass_notes, timing_ms=4, vel_range=3, rng=rng)
    bass_buf = build_808(bass_notes, kit, sc_gain, NSAMP, BEAT, BAR_DUR, rng)

    # Step 5: Melodic instruments
    print('\nStep 5: Building melodic instruments ...')

    # Flute Solo (track 1) — main melody, reverb, widened
    flute_notes = parse_track(FIXED_MID, 1)
    flute_notes = humanize_notes(flute_notes, timing_ms=8, vel_range=4, rng=rng)
    flute_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.65, damping=0.30, wet_level=0.35,
                  dry_level=0.80, width=0.90),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6, release_ms=150),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    flute_buf = build_sampler_track(
        'Flute Solo',
        os.path.join(GB_SAMPLER, 'Flute Solo', 'Flute_LV_na_sus_mf'),
        flute_notes, sc_gain, NSAMP, BEAT, flute_board,
        mix_gain=0.50, widen_ms=14, rng=rng)

    # Harp (track 2) — 16th arp, bright, widened
    harp_notes = parse_track(FIXED_MID, 2)
    harp_notes = humanize_notes(harp_notes, timing_ms=6, vel_range=3, rng=rng)
    harp_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=180),
        pb.LowpassFilter(cutoff_frequency_hz=14000),
        pb.Reverb(room_size=0.55, damping=0.35, wet_level=0.30,
                  dry_level=0.82, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=4, release_ms=120),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    harp_buf = build_sampler_track(
        'Harp',
        os.path.join(GB_SAMPLER, 'Harp', 'Harp_ES_mf'),
        harp_notes, sc_gain, NSAMP, BEAT, harp_board,
        mix_gain=0.40, widen_ms=18, rng=rng)

    # Grand Piano (track 3) — sustained chords, quiet, centered
    piano_notes = parse_track(FIXED_MID, 3)
    piano_notes = humanize_notes(piano_notes, timing_ms=6, vel_range=3, rng=rng)
    piano_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.50, damping=0.40, wet_level=0.25,
                  dry_level=0.85, width=0.75),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=6, release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    piano_buf = build_sampler_track(
        'Grand Piano',
        os.path.join(GB_SAMPLER, 'Grand Piano'),
        piano_notes, sc_gain, NSAMP, BEAT, piano_board,
        mix_gain=0.35, widen_ms=10, rng=rng)

    # Glockenspiel (track 4) — bell accents, bright, widened
    glock_notes = parse_track(FIXED_MID, 4)
    glock_notes = humanize_notes(glock_notes, timing_ms=4, vel_range=2, rng=rng)
    glock_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=500),
        pb.LowpassFilter(cutoff_frequency_hz=16000),
        pb.Reverb(room_size=0.60, damping=0.25, wet_level=0.40,
                  dry_level=0.75, width=0.95),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=2, release_ms=100),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-3.0),
    ])
    glock_buf = build_sampler_track(
        'Glockenspiel',
        os.path.join(GB_SAMPLER, 'Glockenspiel', 'Glockenspiel_Pla_mf1'),
        glock_notes, sc_gain, NSAMP, BEAT, glock_board,
        mix_gain=0.30, widen_ms=16, rng=rng)

    # Transition FX
    fx_buf = build_transition_fx(kit, NSAMP, BAR_DUR, rng)

    # Mix — (buf, coeff, stereo_mode, stereo_param)
    # Centered: kick, 808. Widened: melodies. Stereo: hats (handled in drum build)
    mix = mix_stems([
        (drum_stereo, 0.65, 'center', 0),      # drums
        (bass_buf,    0.55, 'center', 0),       # 808 centered
        (flute_buf,   0.42, 'widen',  14),      # flute — main melody, prominent
        (harp_buf,    0.30, 'widen',  18),      # harp — arp, behind flute
        (piano_buf,   0.25, 'widen',  10),      # piano — quiet bed
        (glock_buf,   0.20, 'widen',  16),      # glock — accents
        (fx_buf,      0.30, 'center', 0),       # FX
    ], NSAMP)

    # Master + export
    master_and_export(mix, NSAMP, BEAT, BAR_DUR, beat_name, vstr, target_lufs=-14.0)
