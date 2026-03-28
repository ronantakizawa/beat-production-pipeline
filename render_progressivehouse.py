"""
Progressive House -- Render Script
Renders MIDI from compose_progressivehouse.py using GarageBand sampler instruments.

MIDI track indices in *_FULL.mid:
  0: tempo/metadata  1: piano (Grand Piano)  2: lead (Flute Solo)
  3: pluck (Harp)    4: pad (String Ensemble)
  5: drums           6: bass

Usage:
  python render_progressivehouse.py --midi ProgHouse_Anthem_FULL.mid --name "ProgHouse_Anthem" --bpm 128
"""

import argparse
import os
import sys
import re
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
OUTPUT_DIR = '/Users/ronantakizawa/Documents/ProgressiveHouse_Beat'
GB_SAMPLER = os.path.join(INST, 'Apple GarageBand Sampler')

BROOTLE = os.path.join(INST, 'Studio Brootle Free House Kick Sample Pack')
OSHH = os.path.join(INST, 'oldschoolhiphop')
CYM_HOUSE = os.path.join(INST, 'Cymatics - House - Starter Pack', 'Drums', 'Drum One Shots')

INTRO_S, INTRO_E = 0, 8
BD1_S, BD1_E = 8, 24
BU1_S, BU1_E = 24, 28
DROP1_S, DROP1_E = 28, 44
BD2_S, BD2_E = 44, 56
BU2_S, BU2_E = 56, 60
DROP2_S, DROP2_E = 60, 76
OUTRO_S, OUTRO_E = 76, 96
NBARS = 96

sys.path.insert(0, INST)
from lofi_fx import generate_sub_bass


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


def bar_to_s(bar, beat=0.0, BAR_DUR=0):
    return (bar + beat / 4.0) * BAR_DUR


def stereo_widen(buf, delay_ms=12):
    d = int(delay_ms / 1000.0 * SR)
    if d <= 0 or d >= len(buf):
        return buf.copy()
    result = buf.copy()
    result[d:, 1] = buf[:-d, 1]
    result[:d, 1] *= 0.3
    return result


def pan_stereo(buf, position):
    angle = (position + 1) * np.pi / 4
    result = buf.copy()
    result[:, 0] *= np.cos(angle)
    result[:, 1] *= np.sin(angle)
    return result


# ============================================================================
# GARAGEBAND SAMPLER
# ============================================================================

def _parse_note_from_filename(f):
    if 'ped' in f.lower() or '50A-2G' in f:
        return None
    m = re.search(r'_([A-G]#?\d+)\.(wav|aif)$', f, re.IGNORECASE)
    if m:
        n = m.group(1)
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        rm = re.match(r'^([A-G]#?)(\d+)$', n)
        if rm:
            pc = note_map.get(rm.group(1))
            if pc is not None:
                return (int(rm.group(2)) + 1) * 12 + pc
    m = re.match(r'^(\d{3})_', f)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d)\s*([a-g])\.(aif|wav)$', f, re.IGNORECASE)
    if m:
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        nl = m.group(2).upper()
        pc = note_map.get(nl)
        if pc is not None:
            return (int(m.group(1)) + 1) * 12 + pc
    return None


def scan_sampler_dir(instrument_dir, prefer_velocity=None):
    samples = {}
    for root, dirs, files in os.walk(instrument_dir):
        for f in files:
            fl = f.lower()
            if not (fl.endswith('.wav') or fl.endswith('.aif') or '.' not in f):
                continue
            full_path = os.path.join(root, f)
            midi = _parse_note_from_filename(f)
            if midi is None:
                continue
            if midi in samples and prefer_velocity and prefer_velocity in f:
                samples[midi] = full_path
            elif midi not in samples:
                samples[midi] = full_path
    return samples


def build_sampler_pitched(sample_map, target_midis):
    pitched = {}
    available = sorted(sample_map.keys())
    if not available:
        return pitched
    for target in target_midis:
        nearest = min(available, key=lambda x: abs(x - target))
        snd = load_sample(sample_map[nearest])
        st = target - nearest
        if st != 0:
            board_p = pb.Pedalboard([pb.PitchShift(semitones=st)])
            snd = board_p(snd[np.newaxis, :], SR)[0].astype(np.float32)
        pitched[target] = snd
    return pitched


# ============================================================================
# MIDI PARSING
# ============================================================================

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


def humanize_notes(notes, timing_ms=5, vel_range=3, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    result = []
    jitter = timing_ms / 1000.0
    for start, note_num, vel, dur in notes:
        tj = rng.uniform(-jitter, jitter)
        vj = rng.randint(-vel_range, vel_range + 1)
        result.append((max(0, start + tj), note_num,
                        int(np.clip(vel + vj, 1, 127)), dur))
    return result


def fix_midi(full_mid, fixed_mid):
    print('\nStep 0: Fixing MIDI ...')
    mid = mido.MidiFile(full_mid)
    new_t0 = mido.MidiTrack()
    seen = False
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            if not seen:
                new_t0.append(mido.MetaMessage('set_tempo', tempo=msg.tempo, time=0))
                seen = True
        else:
            new_t0.append(msg)
    mid.tracks[0] = new_t0
    mid.save(fixed_mid)
    print(f'  FIXED ({len(mid.tracks)} tracks)')


# ============================================================================
# DRUM KIT
# ============================================================================

def load_drum_kit(beat_name):
    print('\nStep 1: Loading drum kit ...')
    rng = np.random.RandomState(hash(beat_name) % (2**31))

    def glob_wavs(*patterns):
        result = []
        for p in patterns:
            result.extend(sorted(_glob.glob(p)))
        return result

    kicks = glob_wavs(f'{BROOTLE}/*.wav')
    snares = glob_wavs(f'{OSHH}/One Shots/Snares_Rims_Claps/*.wav')
    hats = glob_wavs(f'{OSHH}/One Shots/Hats/*.wav')
    crashes = glob_wavs(f'{CYM_HOUSE}/Cymbals/Crashes/*.wav')

    kit = {}
    for name, pool in [('kick', kicks), ('clap', snares), ('hat', hats), ('crash', crashes)]:
        if pool:
            kit[name] = load_sample(pool[rng.randint(len(pool))])
            print(f'  {name}: {os.path.basename(pool[0])}')
        else:
            kit[name] = np.zeros(int(SR * 0.1), dtype=np.float32)
    return kit


# ============================================================================
# BUILD DRUMS
# ============================================================================

def build_drums(drum_events, kit, NSAMP, BAR_DUR, rng):
    print('\nStep 2: Building drums ...')

    room = pra.ShoeBox([8.0, 6.0, 3.5], fs=SR, materials=pra.Material(0.20), max_order=3)
    room.add_source([4.0, 3.0, 1.5])
    room.add_microphone(np.array([[4.2, 3.2, 1.6]]).T)
    room.compute_rir()
    room_ir = np.array(room.rir[0][0], dtype=np.float32)[:int(SR * 0.15)]
    room_ir /= (np.abs(room_ir).max() + 1e-9)

    kick_L = np.zeros(NSAMP, dtype=np.float32)
    kick_R = np.zeros(NSAMP, dtype=np.float32)
    nk_L = np.zeros(NSAMP, dtype=np.float32)
    nk_R = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)

    KICK, CLAP, HH, CRASH = kit['kick'], kit['clap'], kit['hat'], kit['crash']
    samples = {
        36: (KICK,  0.90, 'kick'),
        39: (CLAP,  0.55, 'clap'),
        42: (HH,    0.35, 'hat'),
        49: (CRASH, 0.45, 'crash'),
    }

    MAX_JITTER = int(0.005 * SR)
    pan_toggle = False

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        g = vel / 127.0
        if note_num not in samples:
            continue
        snd_raw, gain, label = samples[note_num]

        if note_num == 36:
            snd = snd_raw * g * gain
            chunk = snd[:min(len(snd), NSAMP - bs)]
            e = bs + len(chunk)
            kick_env[bs:e] += np.abs(chunk)
            kick_L[bs:e] += chunk * 0.96
            kick_R[bs:e] += chunk * 0.96
        elif note_num == 42:
            pan_toggle = not pan_toggle
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s = int(np.clip(bs + jitter, 0, NSAMP - 1))
            v = g * rng.uniform(0.70, 1.00)
            snd = snd_raw * v * gain
            pr = 0.60 if pan_toggle else 0.40
            e = min(s + len(snd), NSAMP)
            ch = snd[:e - s]
            nk_L[s:e] += ch * (1 - pr) * 2
            nk_R[s:e] += ch * pr * 2
        elif note_num == 49:
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56, nsamp=NSAMP)
        else:
            snd = snd_raw * g * rng.uniform(0.90, 1.10) * gain
            place(nk_L, nk_R, snd, bs, 0.50, 0.50, nsamp=NSAMP)

    drum_L = kick_L + nk_L
    drum_R = kick_R + nk_R
    nk_rev_L = fftconvolve(nk_L, room_ir, mode='full')[:NSAMP]
    nk_rev_R = fftconvolve(nk_R, room_ir, mode='full')[:NSAMP]
    drum_L += nk_rev_L * 0.08
    drum_R += nk_rev_R * 0.08

    drum_stereo = np.stack([drum_L, drum_R], axis=1)
    drum_board = pb.Pedalboard([
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=5, release_ms=120),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-1.0),
    ])
    drum_stereo = apply_pb(drum_stereo, drum_board)
    print(f'  {len(drum_events)} drum events')
    return drum_stereo, kick_env


# ============================================================================
# SIDECHAIN
# ============================================================================

def build_sidechain(kick_env, NSAMP):
    print('\nStep 3: Sidechain ...')
    smooth = int(SR * 0.010)
    sc_env = np.convolve(kick_env, np.ones(smooth) / smooth, mode='same')
    sc_env /= sc_env.max() + 1e-9
    sc_gain = np.clip(1.0 - sc_env * 0.70, 0.30, 1.0)
    return sc_gain


# ============================================================================
# BASS (sine sub)
# ============================================================================

def build_bass(bass_notes, sc_gain, NSAMP, BEAT, BAR_DUR, rng):
    print('\nStep 4: Building bass ...')
    bass_L = np.zeros(NSAMP, dtype=np.float32)
    bass_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, midi_note, vel, dur_sec in bass_notes:
        s = int(sec * SR)
        if s >= NSAMP:
            continue
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        dur = min(dur_sec, 2 * BEAT)
        snd = generate_sub_bass(freq, dur, SR)
        g = (vel / 127.0) * rng.uniform(0.92, 1.08) * 0.40
        chunk = snd * g
        e = min(s + len(chunk), NSAMP)
        bass_L[s:e] += chunk[:e - s]
        bass_R[s:e] += chunk[:e - s]

    bass_buf = np.stack([bass_L, bass_R], axis=1)
    bass_buf[:, 0] *= (sc_gain * 0.75 + 0.25)
    bass_buf[:, 1] *= (sc_gain * 0.75 + 0.25)

    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=180),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=8, release_ms=120),
        pb.Limiter(threshold_db=-2.0),
    ])
    bass_buf = apply_pb(bass_buf, bass_board)
    print(f'  Bass events={len(bass_notes)}')
    return bass_buf


# ============================================================================
# SAMPLER TRACK BUILDER
# ============================================================================

def build_sampler_track(name, instrument_dir, track_notes, sc_gain, NSAMP, BEAT,
                        effects_board, mix_gain=0.40, widen_ms=0, rng=None,
                        prefer_velocity=None):
    print(f'\n  Building {name} ...')
    if rng is None:
        rng = np.random.RandomState(42)

    sample_map = scan_sampler_dir(instrument_dir, prefer_velocity=prefer_velocity)
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
        max_samp = min(int((dur_sec + 0.1) * SR), len(snd))
        chunk = snd[:max_samp].copy() * g
        fade_n = max(1, min(int(SR * 0.02), len(chunk) // 4))
        if len(chunk) > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        e = min(s + len(chunk), NSAMP)
        buf_L[s:e] += chunk[:e - s]
        buf_R[s:e] += chunk[:e - s]

    buf = np.stack([buf_L, buf_R], axis=1)
    buf[:, 0] *= (sc_gain * 0.25 + 0.75)
    buf[:, 1] *= (sc_gain * 0.25 + 0.75)
    buf = apply_pb(buf, effects_board)
    if widen_ms > 0:
        buf = stereo_widen(buf, delay_ms=widen_ms)
    print(f'    {name}: {len(track_notes)} notes')
    return buf


# ============================================================================
# TRANSITION FX
# ============================================================================

def build_transition_fx(kit, NSAMP, BAR_DUR, rng):
    print('\nBuilding transition FX ...')
    fx_L = np.zeros(NSAMP, dtype=np.float32)
    fx_R = np.zeros(NSAMP, dtype=np.float32)
    CRASH = kit['crash']
    CLAP = kit['clap']

    def crash_at(bar):
        pos = int(bar_to_s(bar, BAR_DUR=BAR_DUR) * SR)
        place(fx_L, fx_R, CRASH * 0.60, pos, 0.44, 0.56, nsamp=NSAMP)

    crash_at(DROP1_S)
    crash_at(DROP2_S)

    fx_buf = np.stack([fx_L, fx_R], axis=1)
    fx_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.Reverb(room_size=0.55, damping=0.50, wet_level=0.20, dry_level=0.90, width=0.85),
        pb.Gain(gain_db=1.0),
    ])
    fx_buf = apply_pb(fx_buf, fx_board)
    return fx_buf


# ============================================================================
# MIX + MASTER
# ============================================================================

def mix_stems(stems, NSAMP):
    print('\nMixing stems ...')
    mix = np.zeros((NSAMP, 2), dtype=np.float32)
    for buf, coeff, mode, param in stems:
        if mode == 'center':
            processed = pan_stereo(buf, 0.0)
        elif mode == 'widen':
            processed = stereo_widen(buf, delay_ms=param)
        else:
            processed = buf.copy()
        mix += processed * coeff
    return mix


def master_and_export(mix, NSAMP, BEAT, BAR_DUR, beat_name, version_str,
                      target_lufs=-14.0):
    sections = [
        ('Intro',      INTRO_S, INTRO_E),
        ('Breakdown1', BD1_S,   BD1_E),
        ('Buildup1',   BU1_S,   BU1_E),
        ('Drop1',      DROP1_S, DROP1_E),
        ('Breakdown2', BD2_S,   BD2_E),
        ('Buildup2',   BU2_S,   BU2_E),
        ('Drop2',      DROP2_S, DROP2_E),
        ('Outro',      OUTRO_S, OUTRO_E),
    ]

    OUT_WAV = os.path.join(OUTPUT_DIR, f'{beat_name}_{version_str}.wav')
    OUT_MP3 = os.path.join(OUTPUT_DIR, f'{beat_name}_{version_str}.mp3')

    print('\nMaster chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=19000),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=15, release_ms=200),
        pb.Gain(gain_db=-1.0),
    ])
    mix = apply_pb(mix, master_board)
    SONG_DUR = NBARS * BAR_DUR
    trim = int((SONG_DUR + 2.0) * SR)
    mix = mix[:trim]

    # Fade out last 8 bars
    fade_start = int(bar_to_s(NBARS - 8, BAR_DUR=BAR_DUR) * SR)
    fade_end = trim
    fade_len = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve

    # LUFS
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(DROP1_S, BAR_DUR=BAR_DUR) * SR)
    measure_end = int(bar_to_s(DROP1_E, BAR_DUR=BAR_DUR) * SR)

    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    lufs = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    if np.isfinite(lufs):
        gain_db = target_lufs - lufs
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  LUFS: {lufs:.1f} → {target_lufs} (applied {gain_db:+.1f} dB)')

    peak = np.abs(mix).max()
    if peak > 0.98:
        mix = np.tanh(mix * (1.0 / peak) * 1.3) * 0.95

    # Export
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title': f'{beat_name} {version_str}', 'artist': 'Claude Code',
        'album': 'Progressive House', 'genre': 'Progressive House',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')
    print(f'\nDone!  ->  {OUT_MP3}')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive House renderer')
    parser.add_argument('--midi', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--bpm', required=True, type=float)
    args = parser.parse_args()

    BPM = max(118, min(args.bpm, 135))
    BEAT = 60.0 / BPM
    BAR_DUR = BEAT * 4
    NSAMP = int((NBARS * BAR_DUR + 4.0) * SR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    midi_path = args.midi
    if not os.path.isabs(midi_path):
        midi_path = os.path.join(OUTPUT_DIR, midi_path)
    FIXED_MID = os.path.join(OUTPUT_DIR, f'{args.name}_FIXED.mid')

    existing = _glob.glob(os.path.join(OUTPUT_DIR, f'{args.name}_v*.mp3'))
    version = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                   for p in existing], default=0) + 1
    vstr = f'v{version}'

    print(f'\n=== render_progressivehouse.py — {args.name} ===')
    print(f'  BPM: {BPM}  |  {NBARS} bars  |  Output: {vstr}')

    rng = np.random.RandomState(42)

    fix_midi(midi_path, FIXED_MID)
    kit = load_drum_kit(args.name)

    # Drums (track 5)
    drum_events = parse_track(FIXED_MID, 5)
    drum_events = humanize_notes(drum_events, timing_ms=5, vel_range=3, rng=rng)
    drum_stereo, kick_env = build_drums(drum_events, kit, NSAMP, BAR_DUR, rng)

    sc_gain = build_sidechain(kick_env, NSAMP)

    # Bass (track 6)
    bass_notes = parse_track(FIXED_MID, 6)
    bass_notes = humanize_notes(bass_notes, timing_ms=3, vel_range=2, rng=rng)
    bass_buf = build_bass(bass_notes, sc_gain, NSAMP, BEAT, BAR_DUR, rng)

    # Melodic instruments
    print('\nStep 5: Building melodic instruments ...')

    # Piano (track 1)
    piano_notes = parse_track(FIXED_MID, 1)
    piano_notes = humanize_notes(piano_notes, timing_ms=6, vel_range=4, rng=rng)
    piano_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.50, damping=0.30, wet_level=0.25,
                  dry_level=0.85, width=0.80),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=5, release_ms=150),
        pb.Gain(gain_db=0.5),
    ])
    piano_buf = build_sampler_track(
        'Grand Piano', os.path.join(GB_SAMPLER, 'Grand Piano'),
        piano_notes, sc_gain, NSAMP, BEAT, piano_board,
        mix_gain=0.45, widen_ms=12, rng=rng)

    # Lead (track 2)
    lead_notes = parse_track(FIXED_MID, 2)
    lead_notes = humanize_notes(lead_notes, timing_ms=4, vel_range=3, rng=rng)
    lead_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=300),
        pb.Distortion(drive_db=2.0),
        pb.Compressor(threshold_db=-10, ratio=4.0, attack_ms=2, release_ms=80),
        pb.LowpassFilter(cutoff_frequency_hz=14000),
        pb.Gain(gain_db=2.0),
    ])
    lead_buf = build_sampler_track(
        'Flute Lead', os.path.join(GB_SAMPLER, 'Flute Solo'),
        lead_notes, sc_gain, NSAMP, BEAT, lead_board,
        mix_gain=0.50, widen_ms=14, rng=rng)

    # Pluck (track 3)
    pluck_notes = parse_track(FIXED_MID, 3)
    pluck_notes = humanize_notes(pluck_notes, timing_ms=4, vel_range=2, rng=rng)
    pluck_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=250),
        pb.Reverb(room_size=0.55, damping=0.30, wet_level=0.30,
                  dry_level=0.80, width=0.90),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=4, release_ms=120),
        pb.Gain(gain_db=0.5),
    ])
    pluck_buf = build_sampler_track(
        'Harp Pluck', os.path.join(GB_SAMPLER, 'Harp', 'Harp_ES_mf'),
        pluck_notes, sc_gain, NSAMP, BEAT, pluck_board,
        mix_gain=0.30, widen_ms=18, rng=rng)

    # Pad (track 4)
    pad_notes = parse_track(FIXED_MID, 4)
    pad_notes = humanize_notes(pad_notes, timing_ms=8, vel_range=3, rng=rng)
    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=10000),
        pb.Reverb(room_size=0.75, damping=0.25, wet_level=0.50,
                  dry_level=0.60, width=1.0),
        pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=10, release_ms=250),
    ])
    pad_buf = build_sampler_track(
        'String Pad', os.path.join(GB_SAMPLER, 'String Ensemble'),
        pad_notes, sc_gain, NSAMP, BEAT, pad_board,
        mix_gain=0.25, widen_ms=20, rng=rng)

    # FX
    fx_buf = build_transition_fx(kit, NSAMP, BAR_DUR, rng)

    # Mix
    mix = mix_stems([
        (drum_stereo, 0.70, 'center', 0),
        (bass_buf,    0.15, 'center', 0),
        (piano_buf,   0.45, 'widen',  12),
        (lead_buf,    0.50, 'widen',  14),
        (pluck_buf,   0.30, 'widen',  18),
        (pad_buf,     0.25, 'widen',  20),
        (fx_buf,      0.20, 'center', 0),
    ], NSAMP)

    master_and_export(mix, NSAMP, BEAT, BAR_DUR, args.name, vstr)
