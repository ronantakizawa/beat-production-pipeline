"""
Techno -- Render Script
Renders MIDI from compose_techno.py with heavy distortion/saturation.

MIDI track indices in *_FULL.mid:
  0: tempo  1: acid bass (Fingerstyle Electric Bass)
  2: pad (String Ensemble)  3: drums

Usage:
  python render_techno.py --midi Techno_Phrygian_FULL.mid --name "Techno_Phrygian" --bpm 132
"""

import argparse, os, sys, re, numpy as np
from math import gcd
from scipy import signal
from scipy.signal import fftconvolve
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import mido, pedalboard as pb, pyroomacoustics as pra
import glob as _glob, pyloudnorm as pyln

SR = 44100
INST = '/Users/ronantakizawa/Documents/instruments'
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Techno_Beat'
GB_SAMPLER = os.path.join(INST, 'Apple GarageBand Sampler')

VIRION = os.path.join(INST, 'VIRION - BLESSDEEKIT [JERK DRUMKIT]')
MODTRAP = os.path.join(INST, 'Obie - ALL GENRE KIT PT 2 ', '1. TRAP_NEWAGE_ETC', 'MODERN TRAP')
OSHH = os.path.join(INST, 'oldschoolhiphop')

NBARS = 96
BLK2_S, BLK2_E = 16, 32
BLK3_S, BLK3_E = 32, 48
BRK_S, BRK_E = 48, 56
BLK5_S, BLK5_E = 56, 80


def load_sample(path):
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0, nsamp=0):
    e = min(start_s + len(snd), nsamp)
    if e <= start_s: return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R


def apply_pb(arr2ch, board):
    return board(arr2ch.T.astype(np.float32), SR).T.astype(np.float32)


def bar_to_s(bar, beat=0.0, BAR_DUR=0):
    return (bar + beat / 4.0) * BAR_DUR


def stereo_widen(buf, delay_ms=12):
    d = int(delay_ms / 1000.0 * SR)
    if d <= 0 or d >= len(buf): return buf.copy()
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


# GarageBand sampler loader (same as render_mosey.py)
def _parse_note(f):
    if 'ped' in f.lower(): return None
    m = re.search(r'_([A-G]#?\d+)\.(wav|aif)$', f, re.IGNORECASE)
    if m:
        nm = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
        rm = re.match(r'^([A-G]#?)(\d+)$', m.group(1))
        if rm:
            pc = nm.get(rm.group(1))
            if pc is not None: return (int(rm.group(2))+1)*12+pc
    m = re.match(r'^(\d{3})_', f)
    if m: return int(m.group(1))
    return None


def scan_sampler_dir(d):
    samples = {}
    for root, _, files in os.walk(d):
        for f in files:
            fl = f.lower()
            if not (fl.endswith('.wav') or fl.endswith('.aif') or '.' not in f): continue
            midi = _parse_note(f)
            if midi is not None and midi not in samples:
                samples[midi] = os.path.join(root, f)
    return samples


def build_sampler_pitched(smap, targets):
    pitched = {}
    avail = sorted(smap.keys())
    if not avail: return pitched
    for t in targets:
        nearest = min(avail, key=lambda x: abs(x-t))
        snd = load_sample(smap[nearest])
        st = t - nearest
        if st != 0:
            snd = pb.Pedalboard([pb.PitchShift(semitones=st)])(snd[np.newaxis,:], SR)[0].astype(np.float32)
        pitched[t] = snd
    return pitched


def parse_track(mid_path, idx):
    mid = mido.MidiFile(mid_path)
    tpb = mid.ticks_per_beat
    tv = 500000
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo': tv = msg.tempo; break
    active, result, ticks = {}, [], 0
    for msg in mid.tracks[idx]:
        ticks += msg.time
        t = mido.tick2second(ticks, tpb, tv)
        if msg.type == 'note_on' and msg.velocity > 0:
            active[msg.note] = (t, msg.velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active:
                s, v = active.pop(msg.note)
                if t-s > 0: result.append((s, msg.note, v, t-s))
    return result


def humanize(notes, ms=3, vr=3, rng=None):
    if rng is None: rng = np.random.RandomState(42)
    return [(max(0, s+rng.uniform(-ms/1000, ms/1000)), n, int(np.clip(v+rng.randint(-vr,vr+1),1,127)), d)
            for s,n,v,d in notes]


def fix_midi(full, fixed):
    mid = mido.MidiFile(full)
    t0 = mido.MidiTrack()
    seen = False
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            if not seen: t0.append(mido.MetaMessage('set_tempo', tempo=msg.tempo, time=0)); seen = True
        else: t0.append(msg)
    mid.tracks[0] = t0
    mid.save(fixed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--midi', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--bpm', required=True, type=float)
    args = parser.parse_args()

    BPM = max(124, min(args.bpm, 145))
    BEAT = 60.0 / BPM
    BAR_DUR = BEAT * 4
    NSAMP = int((NBARS * BAR_DUR + 4.0) * SR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    midi_path = args.midi if os.path.isabs(args.midi) else os.path.join(OUTPUT_DIR, args.midi)
    FIXED = os.path.join(OUTPUT_DIR, f'{args.name}_FIXED.mid')

    existing = _glob.glob(os.path.join(OUTPUT_DIR, f'{args.name}_v*.mp3'))
    ver = max([int(os.path.basename(p).split('_v')[1].split('.')[0]) for p in existing], default=0) + 1
    vstr = f'v{ver}'

    print(f'\n=== render_techno.py — {args.name} ===')
    print(f'  BPM: {BPM}  |  {NBARS} bars  |  {vstr}')

    rng = np.random.RandomState(42)
    fix_midi(midi_path, FIXED)

    # Drum kit
    print('\nLoading drum kit ...')
    kicks = sorted(_glob.glob(f'{VIRION}/Kick/*.wav'))
    snares = sorted(_glob.glob(f'{OSHH}/One Shots/Snares_Rims_Claps/*.wav'))
    hats = sorted(_glob.glob(f'{OSHH}/One Shots/Hats/*.wav'))
    kit = {
        'kick': load_sample(kicks[rng.randint(len(kicks))]) if kicks else np.zeros(int(SR*0.1), dtype=np.float32),
        'clap': load_sample(snares[rng.randint(len(snares))]) if snares else np.zeros(int(SR*0.1), dtype=np.float32),
        'hat': load_sample(hats[rng.randint(len(hats))]) if hats else np.zeros(int(SR*0.1), dtype=np.float32),
    }

    # Drums (track 3)
    print('\nBuilding drums ...')
    drum_events = humanize(parse_track(FIXED, 3), ms=2, vr=3, rng=rng)
    room = pra.ShoeBox([4.0, 3.0, 3.0], fs=SR, materials=pra.Material(0.40), max_order=2)
    room.add_source([2.0, 1.5, 1.5]); room.add_microphone(np.array([[2.2, 1.7, 1.6]]).T)
    room.compute_rir()
    room_ir = np.array(room.rir[0][0], dtype=np.float32)[:int(SR*0.08)]
    room_ir /= np.abs(room_ir).max() + 1e-9

    kick_L, kick_R = np.zeros(NSAMP, np.float32), np.zeros(NSAMP, np.float32)
    nk_L, nk_R = np.zeros(NSAMP, np.float32), np.zeros(NSAMP, np.float32)
    kick_env = np.zeros(NSAMP, np.float32)
    KICK, CLAP, HH = kit['kick'], kit['clap'], kit['hat']
    pan_toggle = False

    for sec, nn, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP: continue
        g = vel / 127.0
        if nn == 36:
            snd = KICK * g * 0.75
            ch = snd[:min(len(snd), NSAMP-bs)]
            e = bs + len(ch)
            kick_env[bs:e] += np.abs(ch)
            kick_L[bs:e] += ch * 0.96; kick_R[bs:e] += ch * 0.96
        elif nn == 42:
            pan_toggle = not pan_toggle
            snd = HH * g * 0.35 * rng.uniform(0.7, 1.0)
            pr = 0.60 if pan_toggle else 0.40
            e = min(bs + len(snd), NSAMP)
            nk_L[bs:e] += snd[:e-bs] * (1-pr) * 2
            nk_R[bs:e] += snd[:e-bs] * pr * 2
        elif nn == 39:
            snd = CLAP * g * 0.50 * rng.uniform(0.9, 1.1)
            place(nk_L, nk_R, snd, bs, 0.50, 0.50, nsamp=NSAMP)

    dL = kick_L + nk_L; dR = kick_R + nk_R
    dL += fftconvolve(nk_L, room_ir, 'full')[:NSAMP] * 0.03
    dR += fftconvolve(nk_R, room_ir, 'full')[:NSAMP] * 0.03
    drum_stereo = apply_pb(np.stack([dL, dR], axis=1), pb.Pedalboard([
        pb.Compressor(threshold_db=-12, ratio=3.0, attack_ms=3, release_ms=80),
        pb.Distortion(drive_db=2.0),
        pb.Gain(gain_db=1.5), pb.Limiter(threshold_db=-0.8)]))
    print(f'  {len(drum_events)} events')

    # Sidechain
    sm = int(SR * 0.008)
    sc_env = np.convolve(kick_env, np.ones(sm)/sm, 'same')
    sc_env /= sc_env.max() + 1e-9
    sc_gain = np.clip(1.0 - sc_env * 0.80, 0.20, 1.0)

    # Acid bass (track 1) — Fingerstyle Electric Bass, heavily distorted
    print('\nBuilding acid bass ...')
    acid_notes = humanize(parse_track(FIXED, 1), ms=2, vr=2, rng=rng)
    acid_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=40),
        pb.Distortion(drive_db=10.0),
        pb.LowpassFilter(cutoff_frequency_hz=4000),
        pb.Compressor(threshold_db=-8, ratio=5.0, attack_ms=1, release_ms=40),
        pb.Gain(gain_db=3.0),
    ])
    acid_map = scan_sampler_dir(os.path.join(GB_SAMPLER, 'Tuba Solo'))
    print(f'  {len(acid_map)} bass samples')
    acid_pitched = build_sampler_pitched(acid_map, sorted(set(n[1] for n in acid_notes)))

    aL, aR = np.zeros(NSAMP, np.float32), np.zeros(NSAMP, np.float32)
    for sec, mn, vel, dur in acid_notes:
        s = int(sec * SR)
        if s >= NSAMP or mn not in acid_pitched: continue
        snd = acid_pitched[mn]
        g = (vel/127.0) * rng.uniform(0.92, 1.08) * 0.50
        ms_ = min(int((dur+0.05)*SR), len(snd))
        ch = snd[:ms_].copy() * g
        fn = max(1, min(int(SR*0.008), len(ch)//4))
        if len(ch) > fn: ch[-fn:] *= np.linspace(1, 0, fn)
        e = min(s+len(ch), NSAMP)
        aL[s:e] += ch[:e-s]; aR[s:e] += ch[:e-s]
    acid_buf = np.stack([aL, aR], axis=1)
    acid_buf[:, 0] *= (sc_gain * 0.40 + 0.60)
    acid_buf[:, 1] *= (sc_gain * 0.40 + 0.60)
    acid_buf = apply_pb(acid_buf, acid_board)
    acid_buf = stereo_widen(acid_buf, 10)
    print(f'  {len(acid_notes)} notes')

    # Pad (track 2) — String Ensemble, reverb + distortion
    print('\nBuilding pad ...')
    pad_notes = humanize(parse_track(FIXED, 2), ms=8, vr=3, rng=rng)
    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.Distortion(drive_db=4.0),
        pb.Reverb(room_size=0.80, damping=0.25, wet_level=0.60, dry_level=0.50, width=1.0),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Compressor(threshold_db=-16, ratio=2.0),
    ])
    pad_map = scan_sampler_dir(os.path.join(GB_SAMPLER, 'String Ensemble'))
    print(f'  {len(pad_map)} string samples')
    pad_pitched = build_sampler_pitched(pad_map, sorted(set(n[1] for n in pad_notes)))

    pL, pR = np.zeros(NSAMP, np.float32), np.zeros(NSAMP, np.float32)
    for sec, mn, vel, dur in pad_notes:
        s = int(sec * SR)
        if s >= NSAMP or mn not in pad_pitched: continue
        snd = pad_pitched[mn]
        g = (vel/127.0) * rng.uniform(0.92, 1.08) * 0.25
        ms_ = min(int((dur+0.2)*SR), len(snd))
        ch = snd[:ms_].copy() * g
        fn = max(1, min(int(SR*0.03), len(ch)//4))
        if len(ch) > fn: ch[-fn:] *= np.linspace(1, 0, fn)
        e = min(s+len(ch), NSAMP)
        pL[s:e] += ch[:e-s]; pR[s:e] += ch[:e-s]
    pad_buf = np.stack([pL, pR], axis=1)
    pad_buf[:, 0] *= (sc_gain * 0.30 + 0.70)
    pad_buf[:, 1] *= (sc_gain * 0.30 + 0.70)
    pad_buf = apply_pb(pad_buf, pad_board)
    pad_buf = stereo_widen(pad_buf, 22)
    print(f'  {len(pad_notes)} notes')

    # Mix
    print('\nMixing ...')
    mix = np.zeros((NSAMP, 2), np.float32)
    for buf, coeff, mode, param in [
        (drum_stereo, 0.75, 'center', 0),
        (acid_buf, 0.40, 'widen', 10),
        (pad_buf, 0.25, 'widen', 22),
    ]:
        if mode == 'center': p = pan_stereo(buf, 0.0)
        else: p = stereo_widen(buf, delay_ms=param)
        mix += p * coeff

    # Master
    print('\nMaster chain ...')
    mix = apply_pb(mix, pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=10, release_ms=150),
        pb.Distortion(drive_db=1.5),
        pb.Gain(gain_db=-1.0),
    ]))
    trim = int((NBARS * BAR_DUR + 2.0) * SR)
    mix = mix[:trim]

    # Fade out last 8 bars
    fade_s = int(bar_to_s(NBARS - 8, BAR_DUR=BAR_DUR) * SR)
    fl = trim - fade_s
    if fl > 0:
        fc = np.linspace(1.0, 0.0, fl) ** 2
        mix[fade_s:trim, 0] *= fc; mix[fade_s:trim, 1] *= fc

    # LUFS
    meter = pyln.Meter(SR, block_size=0.400)
    ms_ = int(bar_to_s(BLK5_S, BAR_DUR=BAR_DUR) * SR)
    me_ = int(bar_to_s(BLK5_E, BAR_DUR=BAR_DUR) * SR)
    mix = apply_pb(mix, pb.Pedalboard([pb.Limiter(threshold_db=-1.0)]))
    lufs = meter.integrated_loudness(mix[ms_:me_])
    if np.isfinite(lufs):
        gdb = -14.0 - lufs
        mix = mix * (10 ** (gdb / 20.0))
        print(f'  LUFS: {lufs:.1f} → -14.0 ({gdb:+.1f} dB)')
    pk = np.abs(mix).max()
    if pk > 0.98: mix = np.tanh(mix * (1.0/pk) * 1.3) * 0.95

    # Export
    OUT_WAV = os.path.join(OUTPUT_DIR, f'{args.name}_{vstr}.wav')
    OUT_MP3 = os.path.join(OUTPUT_DIR, f'{args.name}_{vstr}.mp3')
    wavfile.write(OUT_WAV, SR, (mix*32767).clip(-32767,32767).astype(np.int16))
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title': f'{args.name} {vstr}', 'artist': 'Claude Code',
        'album': 'Techno', 'genre': 'Techno'})
    m, s = divmod(int(len(seg)/1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')
    print(f'\nDone!  ->  {OUT_MP3}')
