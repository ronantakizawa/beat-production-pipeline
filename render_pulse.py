"""
Pulse — Render Script (House)
G minor | 124 BPM | 128 bars (~4:08)

MIDI track indices in Pulse_FULL.mid:
  0: tempo/metadata  1: drums        2: shadow perc  3: sub bass
  4: mid bass        5: chord stab   6: lead         7: lead2
  8: pluck           9: pad
"""

import os
import sys
import subprocess
import json
import time
import numpy as np
from collections import defaultdict
from math import gcd
from scipy import signal
from scipy.signal import fftconvolve
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import mido
import pedalboard as pb
import dawdreamer as daw
import pyroomacoustics as pra
import glob as _glob
import pyloudnorm as pyln

from instruments_query import SampleSelector


# ============================================================================
# CONFIG
# ============================================================================

BEAT_NAME = 'Pulse'
OUTPUT    = '/Users/ronantakizawa/Documents/Pulse_Beat'
FULL_MID  = os.path.join(OUTPUT, f'{BEAT_NAME}_FULL.mid')
FIXED_MID = os.path.join(OUTPUT, f'{BEAT_NAME}_FIXED.mid')

_existing = _glob.glob(os.path.join(OUTPUT, f'{BEAT_NAME}_v*.mp3'))
_version  = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                 for p in _existing], default=0) + 1
_vstr     = f'v{_version}'
OUT_WAV   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.wav')
OUT_MP3   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.mp3')
print(f'Output: {OUT_MP3}')

SR    = 44100
BPM   = 124
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 128
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

# Section boundaries (must match compose_pulse.py)
INTRO_S,  INTRO_E  =   0,  16
VERSE_S,  VERSE_E  =  16,  32
BUILD1_S, BUILD1_E =  32,  36
DROP1_S,  DROP1_E  =  36,  68
BREAK_S,  BREAK_E  =  68,  76
BUILD2_S, BUILD2_E =  76,  80
DROP2_S,  DROP2_E  =  80, 112
OUTRO_S,  OUTRO_E  = 112, 128

rng = np.random.RandomState(42)


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


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0):
    e = min(start_s + len(snd), NSAMP)
    if e <= start_s or start_s < 0:
        return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R


def apply_pb(arr2ch, board):
    out = board(arr2ch.T.astype(np.float32), SR)
    return out.T.astype(np.float32)


def midi_to_hz(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))


def parse_track(mid_path, track_idx):
    mid       = mido.MidiFile(mid_path)
    tpb       = mid.ticks_per_beat
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


def make_automation(notes, release_gap=0.015):
    gap      = int(release_gap * SR)
    freq_arr = np.zeros(NSAMP, dtype=np.float32)
    gate_arr = np.zeros(NSAMP, dtype=np.float32)
    gain_arr = np.ones(NSAMP,  dtype=np.float32)
    for start_sec, note_num, vel, dur_sec in notes:
        s = int(start_sec * SR)
        e = min(int((start_sec + dur_sec) * SR), NSAMP)
        hz = midi_to_hz(note_num)
        freq_arr[max(0, s - gap):e] = hz
        gate_arr[s:e] = 1.0
        gain_arr[s:e] = vel / 127.0
    last = midi_to_hz(60)
    for i in range(NSAMP):
        if freq_arr[i] > 0:
            last = freq_arr[i]
        else:
            freq_arr[i] = last
    return freq_arr, gate_arr, gain_arr


def humanize_notes(notes, timing_ms=8, vel_range=5):
    result = []
    jitter_samp = timing_ms / 1000.0
    for start, note_num, vel, dur in notes:
        t_jitter = rng.uniform(-jitter_samp, jitter_samp)
        v_jitter = rng.randint(-vel_range, vel_range + 1)
        result.append((max(0, start + t_jitter), note_num,
                        int(np.clip(vel + v_jitter, 1, 127)), dur))
    return result


def faust_render(dsp_string, freq_arr, gate_arr, gain_arr, vol=1.0):
    engine = daw.RenderEngine(SR, 512)
    synth  = engine.make_faust_processor('s')
    synth.set_dsp_string(dsp_string)
    if not synth.compile():
        raise RuntimeError('FAUST compile failed')
    synth.set_automation('/dawdreamer/freq', freq_arr)
    synth.set_automation('/dawdreamer/gate', gate_arr)
    synth.set_automation('/dawdreamer/gain', gain_arr)
    engine.load_graph([(synth, [])])
    engine.render(NSAMP / SR)
    audio = synth.get_audio()
    return (audio.T * vol).astype(np.float32)


def separate_voices(notes):
    groups = defaultdict(list)
    for n in notes:
        key = round(n[0] * 20) / 20
        groups[key].append(n)
    voices = [[], [], [], []]
    for key in sorted(groups):
        ch = sorted(groups[key], key=lambda x: x[1])
        for i, n in enumerate(ch[:4]):
            voices[i].append(n)
    return voices


def bar_to_s(bar, beat=0.0):
    return (bar + beat / 4.0) * BAR


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


def pitch_shift_sample(snd, semitones):
    """Pitch shift by resampling. Positive = higher, negative = lower."""
    if semitones == 0:
        return snd.copy()
    ratio = 2.0 ** (semitones / 12.0)
    new_len = int(len(snd) / ratio)
    if new_len < 2:
        return snd[:2].copy()
    return signal.resample(snd, new_len).astype(np.float32)


def detect_root_midi(snd, info, freq_lo=20, freq_hi=8000):
    """Get sample root MIDI note via FFT. Index root_midi is unreliable."""
    _fft  = np.abs(np.fft.rfft(snd * np.hanning(len(snd))))
    _freq = np.fft.rfftfreq(len(snd), 1 / SR)
    _mask = (_freq > freq_lo) & (_freq < freq_hi)
    if _mask.sum() == 0 or _fft[_mask].max() < 1e-9:
        root = info.get('root_midi', 60)
        print(f'  Root from index (FFT failed): MIDI {root} ({midi_to_hz(root):.1f} Hz)')
        return root
    _peak = _freq[_mask][np.argmax(_fft[_mask])]
    root = int(round(12 * np.log2(_peak / 440) + 69))
    idx_root = info.get('root_midi')
    if idx_root is not None and idx_root != root:
        print(f'  Root from FFT: MIDI {root} ({_peak:.1f} Hz)  [index said {idx_root}, off by {root - idx_root}]')
    else:
        print(f'  Root from FFT: MIDI {root} ({_peak:.1f} Hz)')
    return root


# ============================================================================
# FAUST DSP — sub bass only (pure sine, workflow.md allows for sub <200Hz)
# ============================================================================

SUB_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
env = en.adsr(0.01, 0.10, 0.90, 0.20, gate);
process = os.osc(freq) * env * gain * 0.70 : fi.lowpass(2, 120) <: _, _;
"""


# ============================================================================
# STEP 0 — Fix MIDI
# ============================================================================

def fix_midi():
    print('\nStep 0: Fixing MIDI ...')
    mid = mido.MidiFile(FULL_MID)
    new_t0     = mido.MidiTrack()
    seen_tempo = False
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            if not seen_tempo:
                new_t0.append(mido.MetaMessage('set_tempo', tempo=msg.tempo, time=0))
                seen_tempo = True
        else:
            new_t0.append(msg)
    mid.tracks[0] = new_t0
    mid.save(FIXED_MID)
    print(f'  FIXED_MID  ({len(mid.tracks)} tracks)')


# ============================================================================
# STEP 1 — Sample selection
# ============================================================================

def _pick_safe(sel, role, extra_filters=None):
    """Pick a sample, verify file exists."""
    from instruments_query import query, recently_used_paths, recently_used_packs, ROOT
    from instruments_query import pitch_offset_st, TARGET_RMS_DB
    for key_filter in [sel.key, None]:
        candidates = query(role, genre=sel.genre, key=key_filter, scale=sel.scale,
                           n=30, exclude_paths=recently_used_paths(sel.usage_log),
                           exclude_packs=recently_used_packs(sel.usage_log),
                           extra_filters=extra_filters)
        for path, entry in candidates:
            abs_path = str(ROOT / path)
            if not os.path.isfile(abs_path):
                continue
            sel.chosen[role] = path
            p_st = 0
            sample_key = entry.get('key')
            if sel.key and sample_key and entry.get('is_tonal'):
                p_st = pitch_offset_st(sample_key, sel.key)
            g_db = 0.0
            sample_rms = entry.get('rms_db')
            target_rms = TARGET_RMS_DB.get(role)
            if sample_rms is not None and target_rms is not None:
                g_db = round(target_rms - sample_rms, 1)
                g_db = max(-18.0, min(18.0, g_db))
            sel.info[role] = {
                'path': abs_path, 'pitch_st': p_st, 'gain_db': g_db,
                'rms_db': sample_rms, 'root_midi': entry.get('root_midi'),
                'bpm': entry.get('bpm'), 'key': sample_key,
                'scale': entry.get('scale'),
            }
            key_info = f"  key={sample_key}/{entry.get('scale','—')}" if sample_key else ''
            adj_info = ''
            if p_st != 0: adj_info += f'  pitch={p_st:+d}st'
            if abs(g_db) > 0.5: adj_info += f'  gain={g_db:+.1f}dB'
            print(f"  [{role:<12}] {path}{key_info}{adj_info}")
            return abs_path
    raise FileNotFoundError(f'No valid sample found on disk for role={role}')


def select_samples():
    print('\nStep 1: Selecting samples ...')
    sel = SampleSelector(genre='house', beat='Pulse', seed=42, key='G')

    samples = {}
    samples['kick']       = _pick_safe(sel, 'kick')
    samples['clap']       = _pick_safe(sel, 'snare')  # house preset: snare role for clap
    samples['hat_closed'] = _pick_safe(sel, 'hat_closed')
    samples['hat_open']   = _pick_safe(sel, 'hat_open')
    # FX not in house preset — use progressive_house selector for FX only
    fx_sel = SampleSelector(genre='progressive_house', beat='Pulse_fx', seed=99, key='G')
    samples['fx'] = _pick_safe(fx_sel, 'fx')

    # Melodic one-shots: hand-picked, ALL DIFFERENT from Horizon + Elevate
    # Root MIDI values will be FFT-verified at build time
    INST = '/Users/ronantakizawa/Documents/instruments'

    # Hand-pick bass — SampleSelector roots are unreliable for 808 samples
    samples['mid_bass']   = os.path.join(INST, 'house/One Shots/Bass/Bass_F (1).wav')  # FFT root=29 (F1)

    samples['chord_stab'] = os.path.join(INST,
        'FL_Studio/Downloads/FL Studio Mobile Factory Data/DirectWave Samples/'
        'Keyboard/Electric/Piano Electric (5).wav')                    # FFT root=60 (C4), warm short piano
    samples['lead']  = os.path.join(INST,
        'FL_Studio/Downloads/FL Studio Mobile Factory Data/DirectWave Samples/'
        'Keyboard/Sytrus Rhodes Piano/Rhodes Piano (6).wav')           # verified root=66 (F#4), warm Rhodes EP
    samples['lead2'] = os.path.join(INST,
        'FL_Studio/Downloads/FL Studio Mobile Factory Data/DirectWave Samples/'
        'Guitar/Nylon Guitar/Nylon Guitar d4.wav')                     # verified root=62 (D4), organic guitar
    samples['pluck'] = os.path.join(INST,
        'FL_Studio/Downloads/FL Studio Mobile Factory Data/DirectWave Samples/'
        'Keyboard/Electric/Piano Electric (4).wav')                    # FFT root=54 (F#3), piano pluck
    samples['pad']   = os.path.join(INST,
        'FL_Studio/Packs/Legacy/Instruments/Pads/PAD_FsM_C4.wav')    # FFT root=72 (C5)

    # Shadow percussion (hand-picked)
    samples['shaker'] = os.path.join(INST,
        'FL_Studio/Packs/Drums (ModeAudio)/Shakers/Attack Shaker 01.wav')
    samples['tamb']   = os.path.join(INST,
        'FL_Studio/Packs/Drums/Percussion/707 Tamb.wav')
    samples['conga']  = os.path.join(INST,
        'FL_Studio/Packs/Drums/Percussion/808 Conga.wav')

    # Register info for hand-picked samples (roots from index, FFT-verified at build)
    for role, est_root, key in [('chord_stab', 60, 'C'),
                                 ('lead', 66, 'F#'),
                                 ('lead2', 62, 'D'),
                                 ('pluck', 54, 'F#'),
                                 ('pad', 72, 'C'),
                                 ('mid_bass', 29, 'F')]:
        from instruments_query import pitch_offset_st
        p_st = pitch_offset_st(key, sel.key)
        sel.info[role] = {
            'path': samples[role], 'pitch_st': p_st, 'gain_db': 0.0,
            'rms_db': None, 'root_midi': est_root, 'bpm': None,
            'key': key, 'scale': None,
        }
        print(f"  [{role:<12}] {os.path.basename(samples[role])}  est_root={est_root}  pitch_st={p_st}")

    sel.save()
    return sel, samples


# ============================================================================
# STEP 2 — Drums
# ============================================================================

def build_drums(drum_events, sel, samples):
    print('\nStep 2: Building drum track ...')

    KICK   = load_sample(samples['kick'])
    CLAP   = load_sample(samples['clap'])
    HH_CL  = load_sample(samples['hat_closed'])
    HH_OP  = load_sample(samples['hat_open'])

    # Try to pick crash
    try:
        crash_sel = SampleSelector(genre='house', beat='Pulse_crash', seed=99, key='G')
        crash_path = crash_sel.pick('fx',
            extra_filters={'name_contains': ['crash', 'cymbal']})
        CRASH = load_sample(crash_path)
    except (ValueError, Exception):
        CRASH = HH_OP * 1.5

    kick_gain  = 10 ** (sel.info['kick']['gain_db'] / 20.0) if sel.info.get('kick') else 1.0
    clap_gain  = 10 ** (sel.info['snare']['gain_db'] / 20.0) if sel.info.get('snare') else 1.0
    hh_cl_gain = 10 ** (sel.info['hat_closed']['gain_db'] / 20.0) if sel.info.get('hat_closed') else 1.0

    drum_map = {
        36: (KICK,   kick_gain * 1.05, 'kick'),
        39: (CLAP,   clap_gain * 0.85, 'clap'),
        42: (HH_CL,  hh_cl_gain * 0.45, 'hat_closed'),
        46: (HH_OP,  0.55, 'hat_open'),
        49: (CRASH,  0.65, 'crash'),
    }

    # Room IR
    room = pra.ShoeBox([3.0, 2.5, 2.4], fs=SR,
                       materials=pra.Material(0.50), max_order=2)
    room.add_source([1.5, 1.2, 1.2])
    room.add_microphone(np.array([[1.8, 1.6, 1.4]]).T)
    room.compute_rir()
    room_ir = np.array(room.rir[0][0], dtype=np.float32)
    room_ir = room_ir[:int(SR * 0.15)]
    room_ir /= (np.abs(room_ir).max() + 1e-9)

    kick_L   = np.zeros(NSAMP, dtype=np.float32)
    kick_R   = np.zeros(NSAMP, dtype=np.float32)
    nk_L     = np.zeros(NSAMP, dtype=np.float32)
    nk_R     = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)

    MAX_JITTER = int(0.005 * SR)

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        g = vel / 127.0

        if note_num not in drum_map:
            continue

        snd_raw, gain, label = drum_map[note_num]

        if note_num == 36:  # Kick
            snd   = snd_raw * g * gain
            chunk = snd[:min(len(snd), NSAMP - bs)]
            e     = bs + len(chunk)
            kick_env[bs:e] += np.abs(chunk)
            kick_L[bs:e]   += chunk * 0.96
            kick_R[bs:e]   += chunk * 0.96
        elif note_num == 42:  # Closed hi-hat
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s      = int(np.clip(bs + jitter, 0, NSAMP - 1))
            v      = g * rng.uniform(0.85, 1.10)
            snd    = snd_raw * v * gain
            pr     = rng.uniform(0.42, 0.58)
            e      = min(s + len(snd), NSAMP)
            ch     = snd[:e - s]
            nk_L[s:e] += ch * (1 - pr) * 2
            nk_R[s:e] += ch * pr * 2
        elif note_num in (46, 49):  # Open HH / Crash
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56)
        else:  # Clap
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s      = int(np.clip(bs + jitter, 0, NSAMP - 1))
            snd    = snd_raw * g * rng.uniform(0.90, 1.10) * gain
            e      = min(s + len(snd), NSAMP)
            nk_L[s:e] += snd[:e - s]
            nk_R[s:e] += snd[:e - s]

    # Combine + room IR
    drum_L = kick_L + nk_L
    drum_R = kick_R + nk_R
    dl_room = fftconvolve(drum_L, room_ir, mode='full')[:NSAMP]
    dr_room = fftconvolve(drum_R, room_ir, mode='full')[:NSAMP]
    drum_L  = drum_L * 0.92 + dl_room * 0.06
    drum_R  = drum_R * 0.92 + dr_room * 0.06

    drum_stereo = np.stack([drum_L, drum_R], axis=1)
    drum_board  = pb.Pedalboard([
        pb.Compressor(threshold_db=-10, ratio=4.0, attack_ms=2, release_ms=80),
        pb.Gain(gain_db=2.5),
        pb.Limiter(threshold_db=-0.8),
    ])
    drum_stereo = apply_pb(drum_stereo, drum_board)

    # Extract clap sample for transition FX
    CLAP_SND = CLAP
    print(f'  {len(drum_events)} drum events')
    return drum_stereo, kick_env, CLAP_SND, CRASH


# ============================================================================
# STEP 2b — Shadow percussion (shaker/tamb/conga from separate MIDI track)
# ============================================================================

def build_shadow_perc(sel, samples, sc_gain):
    print('\nStep 2b: Building shadow percussion ...')

    SHAKER = load_sample(samples['shaker'])
    TAMB   = load_sample(samples['tamb'])
    CONGA  = load_sample(samples['conga'])

    notes = parse_track(FIXED_MID, 2)
    print(f'  Shadow perc events: {len(notes)}')

    # GM mapping from compose: 70=shaker 54=tamb 63=conga
    perc_map = {
        70: (SHAKER, 0.55, 'shaker'),
        54: (TAMB,   0.50, 'tamb'),
        63: (CONGA,  0.60, 'conga'),
    }

    sp_L = np.zeros(NSAMP, dtype=np.float32)
    sp_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, _ in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        if note_num not in perc_map:
            continue
        snd_raw, gain, label = perc_map[note_num]
        g = (vel / 127.0) * gain
        snd = snd_raw * g
        # Slight stereo spread
        pr = rng.uniform(0.35, 0.65)
        e = min(bs + len(snd), NSAMP)
        ch = snd[:e - bs]
        sp_L[bs:e] += ch * (1 - pr) * 2
        sp_R[bs:e] += ch * pr * 2

    sp_buf = np.stack([sp_L, sp_R], axis=1)

    # Sidechain (light)
    sp_buf[:, 0] *= (sc_gain * 0.10 + 0.90)
    sp_buf[:, 1] *= (sc_gain * 0.10 + 0.90)

    sp_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=300),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Compressor(threshold_db=-12, ratio=2.0, attack_ms=3, release_ms=80),
        pb.Limiter(threshold_db=-3.0),
    ])
    sp_buf = apply_pb(sp_buf, sp_board)
    print(f'  Shadow perc  max={np.abs(sp_buf).max():.3f}')
    return sp_buf


# ============================================================================
# STEP 3 — Sidechain
# ============================================================================

def build_sidechain(kick_env):
    print('\nStep 3: Sidechain ...')
    smooth  = int(SR * 0.010)
    sc_env  = np.convolve(kick_env, np.ones(smooth) / smooth, mode='same')
    sc_env /= sc_env.max() + 1e-9
    sc_gain = np.clip(1.0 - sc_env * 0.55, 0.45, 1.0)
    print('  Sidechain ready')
    return sc_gain


# ============================================================================
# STEP 4 — Sub bass (FAUST sine)
# ============================================================================

def build_sub_bass(sc_gain):
    print('\nStep 4: Synthesizing sub bass ...')
    notes = parse_track(FIXED_MID, 3)
    print(f'  Sub bass events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(SUB_DSP, freq_a, gate_a, gain_a, vol=0.70)
    buf = buf[:NSAMP]

    buf[:, 0] *= (sc_gain * 0.50 + 0.50)
    buf[:, 1] *= (sc_gain * 0.50 + 0.50)

    sub_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=120),
        pb.Limiter(threshold_db=-2.0),
    ])
    buf = apply_pb(buf, sub_board)
    print(f'  Sub bass  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 5 — Mid bass (sample-based, follows stab rhythm)
# ============================================================================

def build_mid_bass(sel, samples, sc_gain):
    print('\nStep 5: Building mid bass ...')

    bass_snd = load_sample(samples['mid_bass'])
    bass_info = sel.info.get('mid_bass', sel.info.get('bass', {}))

    notes = parse_track(FIXED_MID, 4)
    notes = humanize_notes(notes, timing_ms=6, vel_range=3)
    print(f'  Mid bass events: {len(notes)}')

    # Use index root_midi
    root_midi = bass_info.get('root_midi', 29)
    print(f'  Using index root: MIDI {root_midi}')

    bass_L = np.zeros(NSAMP, dtype=np.float32)
    bass_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur_sec in notes:
        s = int(sec * SR)
        if s >= NSAMP:
            continue
        st = note_num - root_midi
        snd = pitch_shift_sample(bass_snd, st)
        g = (vel / 127.0) * rng.uniform(0.92, 1.08)
        trim = min(int(dur_sec * SR), len(snd))
        chunk = snd[:trim].copy() * g
        fade_n = max(1, int(SR * 0.015))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        e = min(s + len(chunk), NSAMP)
        bass_L[s:e] += chunk[:e - s] * 0.75
        bass_R[s:e] += chunk[:e - s] * 0.75

    bass_buf = np.stack([bass_L, bass_R], axis=1)

    # Normalize to -6 dB peak before processing
    peak = np.abs(bass_buf).max()
    if peak > 1e-6:
        bass_buf *= (0.5 / peak)

    bass_buf[:, 0] *= (sc_gain * 0.60 + 0.40)
    bass_buf[:, 1] *= (sc_gain * 0.60 + 0.40)

    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=1200),
        pb.Compressor(threshold_db=-10, ratio=3.5, attack_ms=4, release_ms=130),
    ])
    bass_buf = apply_pb(bass_buf, bass_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(bass_buf).max()
    if peak > ceil:
        bass_buf *= (ceil / peak)
    print(f'  Mid bass  max={np.abs(bass_buf).max():.3f}')
    return bass_buf


# ============================================================================
# STEP 6 — Chord stab (sample-based, rhythmic chords)
# ============================================================================

def build_chord_stab(sel, samples, sc_gain):
    print('\nStep 6: Building chord stab ...')

    stab_snd = load_sample(samples['chord_stab'])
    stab_info = sel.info['chord_stab']

    notes = parse_track(FIXED_MID, 5)
    notes = humanize_notes(notes, timing_ms=5, vel_range=3)
    print(f'  Chord stab events: {len(notes)}')

    # Use index root_midi
    ref_midi = stab_info.get('root_midi', 60)
    print(f'  Using index root: MIDI {ref_midi}')

    stb_L = np.zeros(NSAMP, dtype=np.float32)
    stb_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        st = note_num - ref_midi
        snd = pitch_shift_sample(stab_snd, st)
        trim = min(int(dur * SR), len(snd))
        chunk = snd[:trim].copy() * (vel / 127.0)
        fade_n = max(1, int(SR * 0.015))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        pan = rng.uniform(-0.12, 0.12)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(stb_L, stb_R, chunk, bs, g_L, g_R)

    stb_buf = np.stack([stb_L, stb_R], axis=1)

    # Normalize to -6 dB peak before processing to avoid limiter saturation
    peak = np.abs(stb_buf).max()
    if peak > 1e-6:
        stb_buf *= (0.5 / peak)

    # Sidechain
    stb_buf[:, 0] *= (sc_gain * 0.20 + 0.80)
    stb_buf[:, 1] *= (sc_gain * 0.20 + 0.80)

    stb_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=10000),
        pb.Reverb(room_size=0.50, damping=0.45, wet_level=0.22,
                  dry_level=0.88, width=0.80),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=5,
                      release_ms=120),
    ])
    stb_buf = apply_pb(stb_buf, stb_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(stb_buf).max()
    if peak > ceil:
        stb_buf *= (ceil / peak)
    print(f'  Chord stab  max={np.abs(stb_buf).max():.3f}')
    return stb_buf


# ============================================================================
# STEP 7 — Lead melody (sample-based)
# ============================================================================

def build_lead(sel, samples, sc_gain):
    print('\nStep 7: Building lead melody ...')

    lead_snd = load_sample(samples['lead'])
    lead_info = sel.info['lead']

    notes = parse_track(FIXED_MID, 6)
    notes = humanize_notes(notes, timing_ms=6, vel_range=4)
    print(f'  Lead events: {len(notes)}')

    # Use index root_midi
    ref_midi = lead_info.get('root_midi', 66)
    print(f'  Using index root: MIDI {ref_midi}')

    mel_L = np.zeros(NSAMP, dtype=np.float32)
    mel_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        st = note_num - ref_midi
        snd = pitch_shift_sample(lead_snd, st)
        trim = min(int(dur * SR), len(snd))
        chunk = snd[:trim].copy() * (vel / 127.0)
        fade_n = max(1, int(SR * 0.02))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        pan = rng.uniform(-0.12, 0.12)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(mel_L, mel_R, chunk, bs, g_L, g_R)

    mel_buf = np.stack([mel_L, mel_R], axis=1)

    # Normalize to -6 dB peak before processing
    peak = np.abs(mel_buf).max()
    if peak > 1e-6:
        mel_buf *= (0.5 / peak)

    mel_buf[:, 0] *= (sc_gain * 0.15 + 0.85)
    mel_buf[:, 1] *= (sc_gain * 0.15 + 0.85)

    mel_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=180),
        pb.LowpassFilter(cutoff_frequency_hz=14000),
        pb.Reverb(room_size=0.60, damping=0.35, wet_level=0.28,
                  dry_level=0.85, width=0.90),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=5,
                      release_ms=150),
    ])
    mel_buf = apply_pb(mel_buf, mel_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(mel_buf).max()
    if peak > ceil:
        mel_buf *= (ceil / peak)
    print(f'  Lead  max={np.abs(mel_buf).max():.3f}')
    return mel_buf


# ============================================================================
# STEP 7b — Lead2 (softer layer, Drop 2 only)
# ============================================================================

def build_lead2(sel, samples, sc_gain):
    print('\nStep 7b: Building lead2 ...')

    lead2_snd = load_sample(samples['lead2'])
    lead2_info = sel.info['lead2']

    notes = parse_track(FIXED_MID, 7)
    notes = humanize_notes(notes, timing_ms=6, vel_range=3)
    print(f'  Lead2 events: {len(notes)}')

    # Use index root_midi
    ref_midi = lead2_info.get('root_midi', 62)
    print(f'  Using index root: MIDI {ref_midi}')

    mel_L = np.zeros(NSAMP, dtype=np.float32)
    mel_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        st = note_num - ref_midi
        snd = pitch_shift_sample(lead2_snd, st)
        trim = min(int(dur * SR), len(snd))
        chunk = snd[:trim].copy() * (vel / 127.0)
        fade_n = max(1, int(SR * 0.02))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        pan = rng.uniform(-0.16, 0.16)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(mel_L, mel_R, chunk, bs, g_L, g_R)

    mel_buf = np.stack([mel_L, mel_R], axis=1)

    # Normalize to -6 dB peak before processing
    peak = np.abs(mel_buf).max()
    if peak > 1e-6:
        mel_buf *= (0.5 / peak)

    mel_buf[:, 0] *= (sc_gain * 0.12 + 0.88)
    mel_buf[:, 1] *= (sc_gain * 0.12 + 0.88)

    mel_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.70, damping=0.40, wet_level=0.35,
                  dry_level=0.80, width=0.92),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=6,
                      release_ms=160),
    ])
    mel_buf = apply_pb(mel_buf, mel_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(mel_buf).max()
    if peak > ceil:
        mel_buf *= (ceil / peak)
    print(f'  Lead2  max={np.abs(mel_buf).max():.3f}')
    return mel_buf


# ============================================================================
# STEP 8 — Pluck (same melody, octave lower)
# ============================================================================

def build_pluck(sel, samples, sc_gain):
    print('\nStep 8: Building pluck ...')

    pluck_snd = load_sample(samples['pluck'])
    pluck_info = sel.info['pluck']

    notes = parse_track(FIXED_MID, 8)
    notes = humanize_notes(notes, timing_ms=5, vel_range=3)
    print(f'  Pluck events: {len(notes)}')

    # Use index root_midi
    ref_midi = pluck_info.get('root_midi', 54)
    print(f'  Using index root: MIDI {ref_midi}')

    plk_L = np.zeros(NSAMP, dtype=np.float32)
    plk_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        st = note_num - ref_midi
        snd = pitch_shift_sample(pluck_snd, st)
        trim = min(int(dur * SR), len(snd))
        chunk = snd[:trim].copy() * (vel / 127.0)
        fade_n = max(1, int(SR * 0.015))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        pan = rng.uniform(-0.18, 0.18)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(plk_L, plk_R, chunk, bs, g_L, g_R)

    plk_buf = np.stack([plk_L, plk_R], axis=1)

    # Normalize to -6 dB peak before processing
    peak = np.abs(plk_buf).max()
    if peak > 1e-6:
        plk_buf *= (0.5 / peak)

    plk_buf[:, 0] *= (sc_gain * 0.10 + 0.90)
    plk_buf[:, 1] *= (sc_gain * 0.10 + 0.90)

    plk_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=220),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.55, damping=0.40, wet_level=0.28,
                  dry_level=0.88, width=0.90),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=4,
                      release_ms=130),
    ])
    plk_buf = apply_pb(plk_buf, plk_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(plk_buf).max()
    if peak > ceil:
        plk_buf *= (ceil / peak)
    print(f'  Pluck  max={np.abs(plk_buf).max():.3f}')
    return plk_buf


# ============================================================================
# STEP 9 — Pad / Dub chord (sample-based with bandpass filter sweep)
# ============================================================================

def build_pad(sel, samples, sc_gain):
    print('\nStep 9: Building pad (dub chord) ...')

    pad_snd = load_sample(samples['pad'])
    pad_info = sel.info['pad']

    notes = parse_track(FIXED_MID, 9)
    notes = humanize_notes(notes, timing_ms=4, vel_range=2)
    print(f'  Pad events: {len(notes)}')

    # Use index root_midi
    ref_midi = pad_info.get('root_midi', 72)
    print(f'  Using index root: MIDI {ref_midi}')

    pad_L = np.zeros(NSAMP, dtype=np.float32)
    pad_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        st = note_num - ref_midi
        snd = pitch_shift_sample(pad_snd, st)
        trim = min(int(dur * SR), len(snd))
        # Loop if sample shorter than note
        if len(snd) < trim:
            repeats = (trim // len(snd)) + 1
            snd = np.tile(snd, repeats)
        chunk = snd[:trim].copy() * (vel / 127.0)
        fade_in = min(int(SR * 0.05), trim // 4)
        fade_out = min(int(SR * 0.05), trim // 4)
        if fade_in > 1:
            chunk[:fade_in] *= np.linspace(0, 1, fade_in)
        if fade_out > 1:
            chunk[-fade_out:] *= np.linspace(1, 0, fade_out)
        pan = rng.uniform(-0.20, 0.20)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(pad_L, pad_R, chunk, bs, g_L, g_R)

    pad_buf = np.stack([pad_L, pad_R], axis=1)

    # Normalize to -6 dB peak before processing
    peak = np.abs(pad_buf).max()
    if peak > 1e-6:
        pad_buf *= (0.5 / peak)

    # Light sidechain
    pad_buf[:, 0] *= (sc_gain * 0.10 + 0.90)
    pad_buf[:, 1] *= (sc_gain * 0.10 + 0.90)

    # Dub chord effect: bandpass filter with movement
    # Apply per-bar varying cutoff for organic sweep feel
    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=6000),
        pb.Reverb(room_size=0.85, damping=0.35, wet_level=0.50,
                  dry_level=0.65, width=0.95),
        pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=20,
                      release_ms=300),
    ])
    pad_buf = apply_pb(pad_buf, pad_board)
    # Soft-clip to -3 dB ceiling
    ceil = 0.708
    peak = np.abs(pad_buf).max()
    if peak > ceil:
        pad_buf *= (ceil / peak)
    print(f'  Pad  max={np.abs(pad_buf).max():.3f}')
    return pad_buf


# ============================================================================
# STEP 10 — Atmosphere (risers, noise — no impacts)
# ============================================================================

def build_atmosphere(sel, samples):
    print('\nStep 10: Building atmosphere ...')

    atm_L = np.zeros(NSAMP, dtype=np.float32)
    atm_R = np.zeros(NSAMP, dtype=np.float32)

    fx_snd = load_sample(samples['fx'])

    # White noise (continuous, very low)
    noise = rng.randn(NSAMP).astype(np.float32) * 0.015
    noise_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=1000),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
    ])
    noise_stereo = np.stack([noise, noise], axis=1)
    noise_stereo = apply_pb(noise_stereo, noise_board)
    atm_L += noise_stereo[:, 0]
    atm_R += noise_stereo[:, 1]
    print('  Noise layer placed')

    # Risers before drops
    for target_bar in [DROP1_S, DROP2_S]:
        riser_start = int(bar_to_s(target_bar - 4) * SR)
        riser_end   = int(bar_to_s(target_bar) * SR)
        riser_len   = riser_end - riser_start
        if riser_len > 0:
            riser = np.zeros(riser_len, dtype=np.float32)
            chunk_len = min(len(fx_snd), riser_len)
            riser[:chunk_len] = fx_snd[:chunk_len] * 0.35
            env = np.linspace(0.1, 1.0, riser_len) ** 2
            riser *= env
            fade = min(int(0.1 * SR), riser_len // 4)
            if fade > 1:
                riser[:fade] *= np.linspace(0, 1, fade)
            place(atm_L, atm_R, riser, riser_start, 0.48, 0.52)
    print('  Risers placed')

    print('  Impacts skipped (user preference)')

    # Downlifter at breakdown (bar 68)
    dl_start = int(bar_to_s(BREAK_S) * SR)
    dl_len = min(len(fx_snd), int(bar_to_s(BREAK_S + 4) * SR) - dl_start)
    if dl_len > 0:
        dl = fx_snd[:dl_len].copy()[::-1] * 0.35
        env = np.linspace(1.0, 0.0, dl_len)
        dl *= env
        place(atm_L, atm_R, dl, dl_start, 0.50, 0.50)
    print('  Downlifter placed')

    atm_buf = np.stack([atm_L, atm_R], axis=1)
    atm_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=60),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.80, damping=0.45, wet_level=0.35,
                  dry_level=0.80, width=1.0),
        pb.Limiter(threshold_db=-3.0),
    ])
    atm_buf = apply_pb(atm_buf, atm_board)
    print(f'  Atmosphere  max={np.abs(atm_buf).max():.3f}')
    return atm_buf


# ============================================================================
# STEP 11 — Transition FX (reverse tails + crash)
# ============================================================================

def build_transition_fx(clap_sample, crash_sample):
    print('\nStep 11: Building transition FX ...')
    fx_L = np.zeros(NSAMP, dtype=np.float32)
    fx_R = np.zeros(NSAMP, dtype=np.float32)

    def reverse_tail(target_bar, length_beats=3):
        tail_len = int(length_beats * BEAT * SR)
        padded   = np.zeros(tail_len, dtype=np.float32)
        sn_len   = min(len(clap_sample), tail_len)
        padded[:sn_len] = clap_sample[:sn_len] * 0.5
        tail_verb = pb.Pedalboard([
            pb.Reverb(room_size=0.92, damping=0.25, wet_level=0.95,
                      dry_level=0.0, width=1.0),
        ])
        wet  = tail_verb(padded[np.newaxis, :], SR)[0]
        rev  = wet[::-1].copy()
        fade = int(0.04 * SR)
        rev[:fade] *= np.linspace(0, 1, fade)
        end_s   = int(bar_to_s(target_bar) * SR)
        start_s = max(0, end_s - len(rev))
        chunk   = rev[:end_s - start_s]
        place(fx_L, fx_R, chunk, start_s, 0.48, 0.52)

    # Before Drop 1
    reverse_tail(DROP1_S, length_beats=3)
    place(fx_L, fx_R, crash_sample * 0.72,
          int(bar_to_s(DROP1_S) * SR), 0.44, 0.56)

    # Before Drop 2
    reverse_tail(DROP2_S, length_beats=3)
    place(fx_L, fx_R, crash_sample * 0.72,
          int(bar_to_s(DROP2_S) * SR), 0.44, 0.56)

    fx_buf = np.stack([fx_L, fx_R], axis=1)
    fx_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=120),
        pb.Reverb(room_size=0.55, damping=0.60, wet_level=0.18,
                  dry_level=0.92, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=3.0, attack_ms=4,
                      release_ms=120),
        pb.Gain(gain_db=1.5),
    ])
    fx_buf = apply_pb(fx_buf, fx_board)
    print('  Transition FX ready')
    return fx_buf


# ============================================================================
# MIX
# ============================================================================

def mix_stems(stems):
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

def master_and_export(mix, target_lufs=-12.0):
    sections = [
        ('Intro',     INTRO_S,  INTRO_E),
        ('Verse',     VERSE_S,  VERSE_E),
        ('Buildup1',  BUILD1_S, BUILD1_E),
        ('Drop1',     DROP1_S,  DROP1_E),
        ('Breakdown', BREAK_S,  BREAK_E),
        ('Buildup2',  BUILD2_S, BUILD2_E),
        ('Drop2',     DROP2_S,  DROP2_E),
        ('Outro',     OUTRO_S,  OUTRO_E),
    ]

    # Master chain
    print('Master chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=15,
                      release_ms=200),
        pb.Gain(gain_db=0.5),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix  = mix[:trim]

    # Fade out bars 120-127
    FADE_START_BAR = 120
    fade_start = int(bar_to_s(FADE_START_BAR) * SR)
    fade_end   = trim
    fade_len   = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: bars {FADE_START_BAR}-{NBARS} ({fade_len/SR:.1f}s)')

    # LUFS normalization on body only (exclude intro + fade)
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(VERSE_S) * SR)
    measure_end   = fade_start
    measure_region = mix[measure_start:measure_end]

    lufs_before = ln_meter.integrated_loudness(measure_region)
    print(f'  Pre-norm LUFS (body): {lufs_before:.1f}')
    if np.isfinite(lufs_before):
        gain_db = target_lufs - lufs_before
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    # Final brickwall limiter to catch peaks without squashing average
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    # Ensure true-peak ceiling at -0.3 dB
    ceiling = 0.966
    peak = np.abs(mix).max()
    if peak > ceiling:
        mix *= (ceiling / peak)
        print(f'  True-peak limited: {peak:.3f} -> {ceiling:.3f}')

    lufs_after = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Post-norm LUFS (body): {lufs_after:.1f}  (target: {target_lufs})')

    # Export
    print('Exporting ...')
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title':  f'{BEAT_NAME} {_vstr}',
        'artist': 'Claude Code',
        'album':  BEAT_NAME,
        'genre':  'House',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # === Mix analysis ===
    print('\n== Mix Analysis ==')

    y_mono = mix.mean(axis=1).astype(np.float32)
    rms_val = np.sqrt(np.mean(y_mono ** 2))
    _spec = np.abs(np.fft.rfft(y_mono))
    _freqs = np.fft.rfftfreq(len(y_mono), 1.0 / SR)
    _spec_sum = _spec.sum() + 1e-9
    centroid_hz = np.sum(_freqs * _spec) / _spec_sum
    bw_hz = np.sqrt(np.sum(((_freqs - centroid_hz) ** 2) * _spec) / _spec_sum)
    final_lufs = ln_meter.integrated_loudness(mix)
    peak_db = 20 * np.log10(np.abs(mix).max() + 1e-9)
    print(f'  Spectral centroid: {centroid_hz:.0f} Hz')
    print(f'  Spectral bandwidth: {bw_hz:.0f} Hz')
    print(f'  RMS: {rms_val:.4f}  ({20*np.log10(rms_val+1e-9):.1f} dB)')
    print(f'  Integrated LUFS: {final_lufs:.1f}')
    print(f'  True peak: {peak_db:.1f} dB')

    # Per-section loudness
    print('\n-- Per-Section Loudness --')
    all_warnings = []
    for name, s_bar, e_bar in sections:
        s_samp = int(bar_to_s(s_bar) * SR)
        e_samp = min(int(bar_to_s(e_bar) * SR), len(mix))
        section = mix[s_samp:e_samp]
        if len(section) < SR:
            continue
        sec_lufs = ln_meter.integrated_loudness(section)
        sec_peak = 20 * np.log10(np.abs(section).max() + 1e-9)
        sec_rms  = np.sqrt(np.mean(section.mean(axis=1) ** 2))
        sec_rms_db = 20 * np.log10(sec_rms + 1e-9)
        status = ''
        if np.isfinite(sec_lufs) and sec_lufs > target_lufs + 3:
            status = ' << TOO LOUD'
            all_warnings.append(f'{name} is {sec_lufs - target_lufs:+.1f} dB over target')
        if sec_peak > -0.5:
            status += ' << CLIPPING'
            all_warnings.append(f'{name} peak at {sec_peak:.1f} dB')
        print(f'  {name:<12} LUFS: {sec_lufs:>6.1f}  peak: {sec_peak:>6.1f} dB  '
              f'RMS: {sec_rms_db:>6.1f} dB{status}')

    # Artifact detection
    print('\n-- Artifact Check --')
    clip_threshold = 0.99
    clip_count = np.sum(np.abs(mix) > clip_threshold)
    if clip_count > 0:
        clip_ms = clip_count / SR / 2 * 1000
        all_warnings.append(f'Clipping: {clip_count} samples ({clip_ms:.0f} ms)')
        print(f'  CLIP: {clip_count} samples above {clip_threshold} ({clip_ms:.0f} ms)')
    else:
        print(f'  Clipping: none')

    dc_L = np.abs(np.mean(mix[:, 0]))
    dc_R = np.abs(np.mean(mix[:, 1]))
    if dc_L > 0.005 or dc_R > 0.005:
        all_warnings.append(f'DC offset: L={dc_L:.4f} R={dc_R:.4f}')
        print(f'  DC offset: L={dc_L:.4f} R={dc_R:.4f} << NEEDS HPF')
    else:
        print(f'  DC offset: clean')

    mono_rms = np.sqrt(np.mean(y_mono ** 2))
    stereo_rms = np.sqrt(np.mean(mix ** 2))
    mono_loss_db = 20 * np.log10(mono_rms / (stereo_rms + 1e-9) + 1e-9)
    if mono_loss_db < -3.0:
        all_warnings.append(f'Phase cancellation: {mono_loss_db:.1f} dB mono loss')
        print(f'  Phase: {mono_loss_db:.1f} dB mono loss << CHECK STEREO WIDENING')
    else:
        print(f'  Phase: {mono_loss_db:.1f} dB mono loss (ok)')

    for name, s_bar, e_bar in sections:
        s_samp = int(bar_to_s(s_bar) * SR)
        e_samp = min(int(bar_to_s(e_bar) * SR), len(mix))
        if np.abs(mix[s_samp:e_samp]).max() < 0.01:
            all_warnings.append(f'{name} is nearly silent')
            print(f'  SILENT: {name}')

    if all_warnings:
        print(f'\n  !! {len(all_warnings)} WARNING(S):')
        for w in all_warnings:
            print(f'     - {w}')
    else:
        print('\n  All checks passed.')

    # librosa analysis
    _analysis = subprocess.run(
        ['python', '-c', f"""
import numpy as np, json, librosa
y, _ = librosa.load('{OUT_WAV}', sr={SR}, mono=True)
c = float(librosa.feature.spectral_centroid(y=y, sr={SR}).mean())
r = float(librosa.feature.rms(y=y).mean())
b = float(librosa.feature.spectral_bandwidth(y=y, sr={SR}).mean())
print(json.dumps({{"centroid": c, "rms": r, "bandwidth": b}}))
"""], capture_output=True, text=True)
    if _analysis.returncode == 0:
        _a = json.loads(_analysis.stdout.strip())
        print(f'\n  librosa centroid: {_a["centroid"]:.0f} Hz')
        print(f'  librosa bandwidth: {_a["bandwidth"]:.0f} Hz')
        print(f'  librosa RMS: {_a["rms"]:.4f}  ({20*np.log10(_a["rms"]+1e-9):.1f} dB)')
    print('====================')

    print(f'\nDone!  ->  {OUT_MP3}')
    return mix


# ============================================================================
# INTERMEDIATES — save/load for parallel stem rendering
# ============================================================================

INTER_DIR = os.path.join(OUTPUT, 'intermediates')


def save_intermediates(drum_stereo, kick_env, clap_snd, crash_snd, sc_gain,
                       sel, samples):
    os.makedirs(INTER_DIR, exist_ok=True)
    np.save(os.path.join(INTER_DIR, 'drums.npy'), drum_stereo)
    np.save(os.path.join(INTER_DIR, 'kick_env.npy'), kick_env)
    np.save(os.path.join(INTER_DIR, 'clap.npy'), clap_snd)
    np.save(os.path.join(INTER_DIR, 'crash.npy'), crash_snd)
    np.save(os.path.join(INTER_DIR, 'sidechain.npy'), sc_gain)
    with open(os.path.join(INTER_DIR, 'samples.json'), 'w') as f:
        json.dump(samples, f)
    with open(os.path.join(INTER_DIR, 'selector_info.json'), 'w') as f:
        json.dump(sel.info, f)
    print(f'  Intermediates saved to {INTER_DIR}')


def load_intermediates():
    sc_gain = np.load(os.path.join(INTER_DIR, 'sidechain.npy'))
    clap_snd = np.load(os.path.join(INTER_DIR, 'clap.npy'))
    crash_snd = np.load(os.path.join(INTER_DIR, 'crash.npy'))
    with open(os.path.join(INTER_DIR, 'samples.json')) as f:
        samples = json.load(f)
    with open(os.path.join(INTER_DIR, 'selector_info.json')) as f:
        info = json.load(f)
    sel = SampleSelector(genre='house', beat='Pulse', seed=42, key='G')
    sel.info = info
    return sc_gain, clap_snd, crash_snd, sel, samples


def render_single_stem(stem_name):
    """Render one stem, loading intermediates from disk."""
    sc_gain, clap_snd, crash_snd, sel, samples = load_intermediates()

    builders = {
        'sub_bass':     lambda: build_sub_bass(sc_gain),
        'mid_bass':     lambda: build_mid_bass(sel, samples, sc_gain),
        'shadow_perc':  lambda: build_shadow_perc(sel, samples, sc_gain),
        'chord_stab':   lambda: build_chord_stab(sel, samples, sc_gain),
        'lead':         lambda: build_lead(sel, samples, sc_gain),
        'lead2':        lambda: build_lead2(sel, samples, sc_gain),
        'pluck':        lambda: build_pluck(sel, samples, sc_gain),
        'pad':          lambda: build_pad(sel, samples, sc_gain),
        'atmosphere':   lambda: build_atmosphere(sel, samples),
        'transition':   lambda: build_transition_fx(clap_snd, crash_snd),
    }
    buf = builders[stem_name]()
    np.save(os.path.join(INTER_DIR, f'{stem_name}.npy'), buf)
    print(f'  [{stem_name}] saved')
    return buf


# ============================================================================
# MAIN
# ============================================================================

STEM_NAMES = ['sub_bass', 'mid_bass', 'shadow_perc', 'chord_stab',
              'lead', 'lead2', 'pluck', 'pad', 'atmosphere', 'transition']

if __name__ == '__main__':

    # --stem <name>: render a single stem (used by parallel subprocesses)
    if len(sys.argv) >= 3 and sys.argv[1] == '--stem':
        render_single_stem(sys.argv[2])
        sys.exit(0)

    t0 = time.time()

    # -- Phase 1: Setup (sequential) --
    fix_midi()
    sel, samples = select_samples()
    drum_events = parse_track(FIXED_MID, 1)
    drum_stereo, kick_env, CLAP_SND, CRASH_SND = build_drums(
        drum_events, sel, samples)
    sc_gain = build_sidechain(kick_env)
    save_intermediates(drum_stereo, kick_env, CLAP_SND, CRASH_SND,
                       sc_gain, sel, samples)

    t_setup = time.time()
    print(f'\n  Setup took {t_setup - t0:.1f}s')

    # -- Phase 2: Stems (parallel subprocesses) --
    print('\n=== Rendering stems in parallel ===')
    procs = {}
    for stem in STEM_NAMES:
        p = subprocess.Popen(
            [sys.executable, __file__, '--stem', stem],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs[stem] = p
        print(f'  Spawned {stem} (pid {p.pid})')

    for stem, p in procs.items():
        out, _ = p.communicate()
        if p.returncode != 0:
            print(f'  !! {stem} FAILED (rc={p.returncode})')
            print(out)
        else:
            lines = [l for l in out.strip().split('\n') if l.strip()]
            for l in lines[-3:]:
                print(f'  [{stem}] {l.strip()}')

    t_stems = time.time()
    print(f'\n  All stems rendered in {t_stems - t_setup:.1f}s '
          f'(vs ~{len(STEM_NAMES) * 8}s sequential est.)')

    # -- Phase 3: Load stems + Mix + Master --
    print('\n=== Loading stems for mix ===')
    stem_bufs = {}
    for stem in STEM_NAMES:
        path = os.path.join(INTER_DIR, f'{stem}.npy')
        stem_bufs[stem] = np.load(path)
        print(f'  {stem:<12} max={np.abs(stem_bufs[stem]).max():.3f}')

    mix = mix_stems([
        (drum_stereo,                0.70, 'center', 0),
        (stem_bufs['shadow_perc'],   0.18, 'widen',  20),
        (stem_bufs['sub_bass'],      0.18, 'center', 0),
        (stem_bufs['mid_bass'],      0.16, 'center', 0),
        (stem_bufs['chord_stab'],    0.45, 'widen',  14),
        (stem_bufs['lead'],          0.50, 'widen',  14),
        (stem_bufs['lead2'],         0.30, 'widen',  18),
        (stem_bufs['pluck'],         0.35, 'widen',  18),
        (stem_bufs['pad'],           0.28, 'widen',  24),
        (stem_bufs['atmosphere'],    0.14, 'widen',  24),
        (stem_bufs['transition'],    0.35, 'center', 0),
    ])

    master_and_export(mix, target_lufs=-12.0)

    t_total = time.time()
    print(f'\n  Total render time: {t_total - t0:.1f}s')
