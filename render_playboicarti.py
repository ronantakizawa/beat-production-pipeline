"""
Playboi Carti / Opium Type Beat — Render Script
Ab minor | 150 BPM | 64 bars

Kit: rap5 ProdScope + rap6 cowbell/FX

MIDI track indices in PlayboiCarti_FULL.mid:
  0: tempo/metadata  1: drums      2: 808 bass
  3: pad (supersaw)  4: lead synth 5: reversed FX
"""

import os
import sys
import subprocess
import json
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


# ============================================================================
# CONFIG
# ============================================================================

BEAT_NAME = 'PlayboiCarti'
OUTPUT    = '/Users/ronantakizawa/Documents/PlayboiCarti_Beat'
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
BPM   = 150
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 64
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

INTRO_S,  INTRO_E  =  0,  8
HOOKA_S,  HOOKA_E  =  8, 24
VERSE_S,  VERSE_E  = 24, 40
BRIDGE_S, BRIDGE_E = 40, 48
HOOKB_S,  HOOKB_E  = 48, 64

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


def humanize_notes(notes, timing_ms=10, vel_range=5):
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


# ============================================================================
# FAUST DSP DEFINITIONS
# ============================================================================

INST = '/Users/ronantakizawa/Documents/instruments'

# SUPERSAW PAD: 7x detuned saw, slow attack 0.4s, release 2.5s, LPF 2000Hz, overdrive
# SUPERSAW RULE: 7 voices detuned saw, NOT triangle/sine
PAD_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = (os.sawtooth(freq)
      + os.sawtooth(freq * 1.005)
      + os.sawtooth(freq * 0.995)
      + os.sawtooth(freq * 1.012)
      + os.sawtooth(freq * 0.988)
      + os.sawtooth(freq * 1.018)
      + os.sawtooth(freq * 0.982)) * 0.143;
env  = en.adsr(0.40, 0.50, 0.65, 2.5, gate);
lfo  = os.osc(0.03) * 0.5 + 0.5;
cutoff = 300.0 + lfo * 1700.0;
process = osc * env * gain * 0.40 : fi.lowpass(2, cutoff) <: _, _;
"""

# LEAD: saw+square, dark pluck, LPF 2500Hz
# LEAD RULE: saw+square dark pluck, NOT bright bell
LEAD_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.sawtooth(freq) * 0.45 + os.square(freq * 0.999) * 0.35 + os.osc(freq * 0.5) * 0.20;
env  = en.adsr(0.020, 0.20, 0.20, 0.8, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 2500) <: _, _;
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
# STEP 1 — Load samples (rap5 ProdScope + rap6 cowbell/FX)
# ============================================================================

RAP5 = os.path.join(INST, 'rap5')
RAP6 = os.path.join(INST, 'rap6')

print('\nStep 1: Loading samples ...')
KICK     = load_sample(os.path.join(RAP5, 'Kicks/Kick  (1).wav'))
CLAP     = load_sample(os.path.join(RAP5, 'Claps/Clap  (1).wav'))
SNARE    = load_sample(os.path.join(RAP5, 'Snares/Snare  (1).wav'))
HH_CL    = load_sample(os.path.join(RAP5, 'Hats/Closed/Hat  (1).wav'))
HH_OP    = load_sample(os.path.join(RAP5, 'Hats/Open/OH  (1).wav'))
COWBELL  = load_sample(os.path.join(RAP6, 'Percs/cardo cowbell 2k17.wav'))
CRASH    = load_sample(os.path.join(RAP5, 'FX/Crashes/Crash (1).wav'))
BASS_808 = load_sample(os.path.join(RAP5, '808s/ProdScope - Carti - 808.wav'))
FX_SND   = load_sample(os.path.join(RAP6, 'FX/pbs - fki [ Transition].wav'))

# Pitch-shift hats down -3 semitones for darker feel (workflow + tutorial 2)
hh_board = pb.Pedalboard([pb.PitchShift(semitones=-3)])
HH_CL    = hh_board(HH_CL[np.newaxis, :], SR)[0].astype(np.float32)
HH_OP    = hh_board(HH_OP[np.newaxis, :], SR)[0].astype(np.float32)
print('  Samples loaded (hats pitched -3st)')


# ============================================================================
# STEP 2 — Drums
# ============================================================================

def build_drums(drum_events):
    print('\nStep 2: Building drum track ...')

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

    samples = {
        36: (KICK,    1.10, 'kick'),
        39: (CLAP,    0.85, 'clap'),
        38: (SNARE,   0.70, 'snare'),
        42: (HH_CL,   0.50, 'hh'),
        46: (HH_OP,   0.75, 'oh'),
        56: (COWBELL,  0.55, 'cowbell'),
        49: (CRASH,   0.60, 'crash'),
    }

    MAX_JITTER = int(0.005 * SR)
    pan_toggle = False

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        # Skip drums in intro (bars 0-7)
        if sec < bar_to_s(INTRO_E) and note_num != 49:
            continue
        # Skip drums in bridge (bars 40-47)
        if bar_to_s(BRIDGE_S) <= sec < bar_to_s(BRIDGE_E):
            continue
        g = vel / 127.0

        if note_num not in samples:
            continue
        snd_raw, gain, label = samples[note_num]

        if note_num == 36:
            snd   = snd_raw * g * gain
            chunk = snd[:min(len(snd), NSAMP - bs)]
            e     = bs + len(chunk)
            kick_env[bs:e] += np.abs(chunk)
            kick_L[bs:e]   += chunk * 0.96
            kick_R[bs:e]   += chunk * 0.96
        elif note_num == 42:
            pan_toggle = not pan_toggle
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s      = int(np.clip(bs + jitter, 0, NSAMP - 1))
            v      = g * rng.uniform(0.68, 1.00)
            snd    = snd_raw * v * gain
            pr     = 0.62 if pan_toggle else 0.38
            e      = min(s + len(snd), NSAMP)
            ch     = snd[:e - s]
            nk_L[s:e] += ch * (1 - pr) * 2
            nk_R[s:e] += ch * pr * 2
        elif note_num in (46, 49):
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56)
        elif note_num == 56:
            # Cowbell — centered, slight pan variation
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.52, 0.48)
        else:
            jitter = rng.randint(-MAX_JITTER, MAX_JITTER + 1)
            s      = int(np.clip(bs + jitter, 0, NSAMP - 1))
            snd    = snd_raw * g * rng.uniform(0.90, 1.10) * gain
            e      = min(s + len(snd), NSAMP)
            nk_L[s:e] += snd[:e - s]
            nk_R[s:e] += snd[:e - s]

    drum_L = kick_L + nk_L
    drum_R = kick_R + nk_R
    dl_room = fftconvolve(drum_L, room_ir, mode='full')[:NSAMP]
    dr_room = fftconvolve(drum_R, room_ir, mode='full')[:NSAMP]
    drum_L  = drum_L * 0.92 + dl_room * 0.06
    drum_R  = drum_R * 0.92 + dr_room * 0.06

    drum_stereo = np.stack([drum_L, drum_R], axis=1)
    drum_board  = pb.Pedalboard([
        pb.Compressor(threshold_db=-14, ratio=3.0, attack_ms=2, release_ms=80),
        pb.Gain(gain_db=2.5),
        pb.Limiter(threshold_db=-0.8),
    ])
    drum_stereo = apply_pb(drum_stereo, drum_board)
    print(f'  {len(drum_events)} drum events')
    return drum_stereo, kick_env


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
# STEP 4 — 808 Bass (pitch-shifted)
# ============================================================================

def build_808(bass_notes, sc_gain):
    print('\nStep 4: Building 808 bass ...')

    _fft  = np.abs(np.fft.rfft(BASS_808 * np.hanning(len(BASS_808))))
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
        # Skip 808 in bridge section
        if bar_to_s(BRIDGE_S) <= sec < bar_to_s(BRIDGE_E):
            continue
        target = min(pitched.keys(), key=lambda t: abs(t - midi_note))
        snd    = pitched[target]
        g      = (vel / 127.0) * rng.uniform(0.92, 1.08)
        # Short-medium: max 3 beats duration
        max_dur = min(dur_sec, 3 * BEAT)
        trim   = min(int(max_dur * SR), len(snd))
        chunk  = snd[:trim].copy() * g
        fade_n = max(1, int(SR * 0.015))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)
        e = min(s + len(chunk), NSAMP)
        bass_L[s:e] += chunk[:e - s] * 0.75
        bass_R[s:e] += chunk[:e - s] * 0.75

    bass_buf = np.stack([bass_L, bass_R], axis=1)
    bass_buf[:, 0] *= (sc_gain * 0.70 + 0.30)
    bass_buf[:, 1] *= (sc_gain * 0.70 + 0.30)

    # 808 effects: distortion 4dB (heavier per tutorial), LPF 1200Hz
    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=1200),
        pb.Distortion(drive_db=4.0),
        pb.Compressor(threshold_db=-10, ratio=3.5, attack_ms=4, release_ms=130),
        pb.Gain(gain_db=1.5),
        pb.Limiter(threshold_db=-2.0),
    ])
    bass_buf = apply_pb(bass_buf, bass_board)
    print(f'  808 events={len(bass_notes)}  max={np.abs(bass_buf).max():.3f}')
    return bass_buf


# ============================================================================
# STEP 5 — Supersaw Pad (FAUST — dissonant chords, overdrive)
# ============================================================================

def build_pad(sc_gain):
    print('\nStep 5: Synthesizing Supersaw Pad (FAUST) ...')

    notes = parse_track(FIXED_MID, 3)
    notes = humanize_notes(notes, timing_ms=12, vel_range=4)
    print(f'  Pad events: {len(notes)}')

    voices = separate_voices(notes)
    buf = np.zeros((NSAMP, 2), dtype=np.float32)
    for vi, voice in enumerate(voices):
        if not voice:
            continue
        freq_a, gate_a, gain_a = make_automation(voice)
        audio = faust_render(PAD_DSP, freq_a, gate_a, gain_a, vol=0.40)
        buf += audio[:NSAMP]
        print(f'  Voice {vi+1}: {len(voice)} notes')

    # Sidechain
    buf[:, 0] *= (sc_gain * 0.20 + 0.80)
    buf[:, 1] *= (sc_gain * 0.20 + 0.80)

    # Pad effects: HPF 150Hz, LPF 2000Hz, overdrive 2dB, reverb
    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=2000),
        pb.Distortion(drive_db=2.0),
        pb.Reverb(room_size=0.60, damping=0.35, wet_level=0.35,
                  dry_level=0.75, width=0.95),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=8,
                      release_ms=200),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, pad_board)
    print(f'  Pad  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 6 — Lead Synth (FAUST — dark pluck, comically repetitive)
# ============================================================================

def build_lead(sc_gain):
    print('\nStep 6: Synthesizing Lead (FAUST) ...')

    notes = parse_track(FIXED_MID, 4)
    notes = humanize_notes(notes, timing_ms=8, vel_range=4)
    print(f'  Lead events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(LEAD_DSP, freq_a, gate_a, gain_a, vol=0.45)
    buf = buf[:NSAMP]

    # Sidechain
    buf[:, 0] *= (sc_gain * 0.15 + 0.85)
    buf[:, 1] *= (sc_gain * 0.15 + 0.85)

    # Lead effects: HPF 200Hz, LPF 2500Hz, EQ cut lows+highs (tutorial 2)
    lead_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=2500),
        pb.Reverb(room_size=0.45, damping=0.40, wet_level=0.25,
                  dry_level=0.82, width=0.80),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=4,
                      release_ms=150),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, lead_board)
    print(f'  Lead  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# TRANSITION FX (reversed pad swells + crashes)
# ============================================================================

def build_transition_fx():
    print('\nBuilding transition FX ...')
    fx_L = np.zeros(NSAMP, dtype=np.float32)
    fx_R = np.zeros(NSAMP, dtype=np.float32)

    def snare_roll(target_bar, bars_build=1.0):
        n_beats = int(bars_build * 4)
        densities = [2, 3, 4, 6, 8]
        for beat_i in range(n_beats):
            progress = beat_i / n_beats
            d_idx    = min(int(progress * len(densities)), len(densities) - 1)
            n_hits   = densities[d_idx]
            for h in range(n_hits):
                t   = bar_to_s(target_bar - bars_build, beat_i + h / n_hits)
                vel = (0.25 + 0.65 * progress) * rng.uniform(0.90, 1.10)
                s   = int(t * SR)
                if 0 <= s < NSAMP:
                    place(fx_L, fx_R, SNARE * vel * 0.55, s, 0.50, 0.50)

    def reverse_tail(target_bar, length_beats=3):
        tail_len = int(length_beats * BEAT * SR)
        padded   = np.zeros(tail_len, dtype=np.float32)
        sn_len   = min(len(SNARE), tail_len)
        padded[:sn_len] = SNARE[:sn_len] * 0.5
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

    def reverse_fx_swell(target_bar, length_beats=6):
        """Reversed FX sample swell (tutorial: reversed samples for texture)."""
        swell_len = int(length_beats * BEAT * SR)
        padded    = np.zeros(swell_len, dtype=np.float32)
        fx_len    = min(len(FX_SND), swell_len)
        padded[:fx_len] = FX_SND[:fx_len] * 0.4
        fx_verb = pb.Pedalboard([
            pb.Reverb(room_size=0.88, damping=0.30, wet_level=0.80,
                      dry_level=0.10, width=1.0),
        ])
        wet = fx_verb(padded[np.newaxis, :], SR)[0]
        rev = wet[::-1].copy()
        fade = int(0.05 * SR)
        rev[:fade] *= np.linspace(0, 1, fade)
        end_s   = int(bar_to_s(target_bar) * SR)
        start_s = max(0, end_s - len(rev))
        chunk   = rev[:end_s - start_s]
        place(fx_L, fx_R, chunk, start_s, 0.45, 0.55)

    # Intro -> Hook A: reverse FX swell + snare roll + crash
    reverse_fx_swell(HOOKA_S, length_beats=6)
    snare_roll(HOOKA_S, bars_build=1.0)
    place(fx_L, fx_R, CRASH * 0.72,
          int(bar_to_s(HOOKA_S) * SR), 0.44, 0.56)

    # Hook A -> Verse: crash
    place(fx_L, fx_R, CRASH * 0.45,
          int(bar_to_s(VERSE_S) * SR), 0.44, 0.56)

    # Verse -> Bridge: reverse tail
    reverse_tail(BRIDGE_S, length_beats=2)
    reverse_fx_swell(BRIDGE_S, length_beats=4)

    # Bridge -> Hook B: big build — snare roll + reverse FX + crash
    snare_roll(HOOKB_S, bars_build=2.0)
    reverse_fx_swell(HOOKB_S, length_beats=6)
    reverse_tail(HOOKB_S, length_beats=3)
    place(fx_L, fx_R, CRASH * 0.72,
          int(bar_to_s(HOOKB_S) * SR), 0.44, 0.56)

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
# ATMOSPHERE — noise texture layer (synthesized, no MIDI)
# ============================================================================

def build_atmosphere():
    print('\nBuilding atmosphere layer ...')
    atmo = rng.randn(NSAMP).astype(np.float32) * 0.015
    # Shape: fade in during intro, present during hooks, quiet during verse
    env = np.zeros(NSAMP, dtype=np.float32)
    # Intro: fade in
    intro_s = int(bar_to_s(INTRO_S) * SR)
    intro_e = int(bar_to_s(INTRO_E) * SR)
    env[intro_s:intro_e] = np.linspace(0.0, 0.8, intro_e - intro_s)
    # Hook A
    ha_s = int(bar_to_s(HOOKA_S) * SR)
    ha_e = int(bar_to_s(HOOKA_E) * SR)
    env[ha_s:ha_e] = 1.0
    # Verse: quiet
    vs_s = int(bar_to_s(VERSE_S) * SR)
    vs_e = int(bar_to_s(VERSE_E) * SR)
    env[vs_s:vs_e] = 0.3
    # Bridge
    br_s = int(bar_to_s(BRIDGE_S) * SR)
    br_e = int(bar_to_s(BRIDGE_E) * SR)
    env[br_s:br_e] = 0.7
    # Hook B
    hb_s = int(bar_to_s(HOOKB_S) * SR)
    hb_e = int(bar_to_s(HOOKB_E) * SR)
    env[hb_s:hb_e] = 1.0

    atmo *= env
    atmo_stereo = np.stack([atmo, atmo], axis=1)
    # Heavy reverb, HPF 300Hz
    atmo_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=300),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.80, damping=0.50, wet_level=0.70,
                  dry_level=0.30, width=1.0),
    ])
    atmo_stereo = apply_pb(atmo_stereo, atmo_board)
    print('  Atmosphere ready')
    return atmo_stereo


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
# MASTER + EXPORT (workflow Step 5 LUFS pipeline)
# ============================================================================

def master_and_export(mix, target_lufs=-14.0):
    sections = [
        ('Intro',   INTRO_S,  INTRO_E),
        ('Hook A',  HOOKA_S,  HOOKA_E),
        ('Verse',   VERSE_S,  VERSE_E),
        ('Bridge',  BRIDGE_S, BRIDGE_E),
        ('Hook B',  HOOKB_S,  HOOKB_E),
    ]

    # Step 1: Master chain — HPF + LPF + comp + saturation + gain, NO limiter
    print('Master chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=15,
                      release_ms=200),
        pb.Distortion(drive_db=1.5),
        pb.Gain(gain_db=-2.0),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix  = mix[:trim]

    # Step 2: Fade out last 4 bars (quadratic curve)
    FADE_BARS  = 4
    fade_start = int(bar_to_s(NBARS - FADE_BARS) * SR)
    fade_end   = trim
    fade_len   = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: last {FADE_BARS} bars ({fade_len/SR:.1f}s)')

    # Step 3: Measure LUFS on body only (exclude intro + fade-out)
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(INTRO_E) * SR)
    measure_end   = fade_start

    lufs_body = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Pre-norm LUFS (body): {lufs_body:.1f}')

    # Step 4: Apply gain to hit -14 LUFS target
    if np.isfinite(lufs_body):
        gain_db = target_lufs - lufs_body
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    # Step 5: Final true-peak limiter AFTER normalization at -1.0 dB ceiling
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    # Post-limiter correction: limiter raises average loudness, so re-measure
    # and apply trim gain to hit target accurately
    lufs_post_limit = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    if np.isfinite(lufs_post_limit) and abs(lufs_post_limit - target_lufs) > 0.5:
        trim_db = target_lufs - lufs_post_limit
        mix = mix * (10 ** (trim_db / 20.0))
        print(f'  Post-limiter trim: {trim_db:+.1f} dB')

    lufs_final = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Final LUFS (body): {lufs_final:.1f}  (target: {target_lufs})')

    # Export
    print('Exporting ...')
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title':   f'{BEAT_NAME} {_vstr}',
        'artist':  'Claude Code',
        'album':   BEAT_NAME,
        'genre':   'Hip-Hop',
        'comment': 'Ab minor | 150 BPM | Opium/Rage Trap',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # Mix analysis
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
        print(f'  {name:<10} LUFS: {sec_lufs:>6.1f}  peak: {sec_peak:>6.1f} dB  '
              f'RMS: {sec_rms_db:>6.1f} dB{status}')

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
# MAIN
# ============================================================================

if __name__ == '__main__':
    fix_midi()

    # Step 2: Drums
    drum_events = parse_track(FIXED_MID, 1)
    drum_stereo, kick_env = build_drums(drum_events)

    # Step 3: Sidechain
    sc_gain = build_sidechain(kick_env)

    # Step 4: 808 Bass
    bass_notes = parse_track(FIXED_MID, 2)
    bass_buf = build_808(bass_notes, sc_gain)

    # Step 5: Supersaw Pad
    pad_buf = build_pad(sc_gain)

    # Step 6: Lead
    lead_buf = build_lead(sc_gain)

    # Transition FX
    fx_buf = build_transition_fx()

    # Atmosphere
    atmo_buf = build_atmosphere()

    # Mix — (buf, coeff, stereo_mode, stereo_param)
    # Tutorials: kick+808 loudest, melody behind drums/bass
    mix = mix_stems([
        (drum_stereo, 0.60, 'center', 0),      # drums center
        (bass_buf,    0.46, 'center', 0),       # 808 center
        (pad_buf,     0.22, 'widen',  20),      # supersaw pad widened
        (lead_buf,    0.18, 'widen',  14),      # lead widened
        (fx_buf,      0.25, 'widen',  16),      # reversed FX widened
        (atmo_buf,    0.12, 'widen',  22),      # atmosphere widened
    ])

    # Master + export (LUFS -14 for rap)
    master_and_export(mix, target_lufs=-14.0)
