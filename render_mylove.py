"""
MyLove — Render Script (Floyy Menor style modern reggaeton)
A major (11B) | 100 BPM | 72 bars

Kit: REGGAETON 4 + reggaeton5 XXL + reggaeton3 URBANITO

MIDI track indices in MyLove_FULL.mid:
  0: tempo/metadata  1: drums  2: sub bass
  3: pad             4: lead
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

BEAT_NAME = 'MyLove'
OUTPUT    = '/Users/ronantakizawa/Documents/MyLove_Beat'
FULL_MID  = os.path.join(OUTPUT, f'{BEAT_NAME}_FULL.mid')
FIXED_MID = os.path.join(OUTPUT, f'{BEAT_NAME}_FIXED.mid')

# Auto-versioning
_existing = _glob.glob(os.path.join(OUTPUT, f'{BEAT_NAME}_v*.mp3'))
_version  = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                 for p in _existing], default=0) + 1
_vstr     = f'v{_version}'
OUT_WAV   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.wav')
OUT_MP3   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.mp3')
print(f'Output: {OUT_MP3}')

SR    = 44100
BPM   = 100
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 72
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

# Section boundaries (matched to compose_mylove.py)
INTRO_S,   INTRO_E   =  0,  8
VERSE1_S,  VERSE1_E  =  8, 12
HOOK1_S,   HOOK1_E   = 12, 28
BRIDGE1_S, BRIDGE1_E = 28, 32
BRIDGE2_S, BRIDGE2_E = 32, 36
HOOK2_S,   HOOK2_E   = 36, 60
OUTRO_S,   OUTRO_E   = 60, 72

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


def make_automation(notes, retrigger_ms=8):
    """Build freq/gate/gain automation arrays.
    retrigger_ms: brief gate-off gap before each note to force ADSR retrigger.
    """
    retrig   = int(retrigger_ms / 1000.0 * SR)
    freq_arr = np.zeros(NSAMP, dtype=np.float32)
    gate_arr = np.zeros(NSAMP, dtype=np.float32)
    gain_arr = np.ones(NSAMP,  dtype=np.float32)
    for start_sec, note_num, vel, dur_sec in notes:
        s = int(start_sec * SR)
        e = min(int((start_sec + dur_sec) * SR), NSAMP)
        hz = midi_to_hz(note_num)
        # Set frequency slightly ahead so it's ready at gate-on
        freq_arr[max(0, s - retrig):e] = hz
        # Force gate off briefly before this note (retrigger)
        gate_arr[max(0, s - retrig):s] = 0.0
        # Gate on for note duration
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


# ============================================================================
# FAUST DSP LIBRARY
# ============================================================================

# SUB BASS: sine osc, LPF 200Hz
SUB_BASS_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq) * 0.85 + os.osc(freq * 0.5) * 0.15;
env  = en.adsr(0.008, 0.15, 0.80, 0.20, gate);
process = osc * env * gain * 0.70 : fi.lowpass(2, 200) <: _, _;
"""

# PAD: 5x detuned saw strings
PAD_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = (os.sawtooth(freq)
      + os.sawtooth(freq * 1.009)
      + os.sawtooth(freq * 0.991)
      + os.sawtooth(freq * 1.018)
      + os.sawtooth(freq * 0.982)) * 0.2;
env  = en.adsr(0.50, 0.60, 0.68, 3.0, gate);
lfo  = os.osc(0.03) * 0.5 + 0.5;
cutoff = 200.0 + lfo * 600.0;
process = osc * env * gain * 0.40 : fi.lowpass(2, cutoff) <: _, _;
"""

# LEAD: sine+FM pluck
LEAD_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
mod  = os.osc(freq * 2.01) * freq * 0.3;
osc  = os.osc(freq + mod) + os.osc(freq * 0.5) * 0.3;
env  = en.adsr(0.01, 0.15, 0.65, 1.2, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 2500) <: _, _;
"""

# Guitar one-shot reference pitch (detected via autocorrelation)
GUITAR_REF_MIDI = 61  # C#4 (~277 Hz)


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
# STEP 1 — Load samples
# ============================================================================

INST    = '/Users/ronantakizawa/Documents/instruments'
REG4    = os.path.join(INST, 'REGGAETON 4')
REG5    = os.path.join(INST, 'reggaeton5')
REG3    = os.path.join(INST, 'reggaeton3/URBANITO Producer Bundle/URBANO Drum Kit')
RAP5    = os.path.join(INST, 'rap5')
REGDRMS = os.path.join(INST, 'reggaetondrums')

print('\nStep 1: Loading samples ...')
KICK    = load_sample(os.path.join(REG4, 'Kicks/KICK  @el.obie 5.wav'))
RIM     = load_sample(os.path.join(REG3, 'Rims/Rim 4 - URBANO Producer Bundle - @seventhbeats.wav'))
HH_CL   = load_sample(os.path.join(REG5, 'XXL Drum Samples/XXL Hi Hats/07_Closed_Hat.wav'))
HH_OP   = load_sample(os.path.join(REG5, 'XXL Drum Samples/XXL Hi Hats/06_Open_Hat.wav'))
GUITAR  = load_sample(os.path.join(REGDRMS, 'Percu/Clasicos/NO ME CONOCE_REGAE GUITAR.wav'))
print('  Samples loaded')


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

    MAX_JITTER = int(0.006 * SR)
    pan_toggle = False

    # GM mapping: 36=kick 37=rim 42=hh 46=oh
    samples = {
        36: (KICK,   1.05, 'kick'),
        37: (RIM,    0.30, 'rim'),
        42: (HH_CL,  0.48, 'hh'),
        46: (HH_OP,  0.80, 'oh'),
    }

    # Apply swing to hi-hats (30.4% from mix_profile)
    SWING_AMOUNT = 0.304
    swing_offset = BEAT * SWING_AMOUNT * 0.5  # shift offbeat 16ths

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        g = vel / 127.0

        if note_num not in samples:
            continue
        snd_raw, gain, label = samples[note_num]

        # Apply swing to hi-hats on offbeat positions
        if note_num == 42:
            beat_pos = (sec / BEAT) % 1.0
            if 0.2 < beat_pos < 0.8:  # offbeat
                bs = int((sec + swing_offset) * SR)
                if bs >= NSAMP:
                    continue

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
        elif note_num in (46,):
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56)
        else:
            # Rim and others
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
        pb.Compressor(threshold_db=-10, ratio=4.0, attack_ms=2, release_ms=80),
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
# STEP 4 — Sub Bass (FAUST synth)
# ============================================================================

def build_sub_bass(sc_gain):
    """Sub bass via FAUST synthesis — instruments.bass.type=sub_bass."""
    print('\nStep 4: Building sub bass (FAUST) ...')

    notes = parse_track(FIXED_MID, 2)
    notes = humanize_notes(notes, timing_ms=6, vel_range=3)
    print(f'  Sub bass events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(SUB_BASS_DSP, freq_a, gate_a, gain_a, vol=0.75)
    buf = buf[:NSAMP]

    # Sidechain — depth 0.55
    buf[:, 0] *= (sc_gain * 0.55 + 0.45)
    buf[:, 1] *= (sc_gain * 0.55 + 0.45)

    # Sub bass effects — LPF 200Hz, compression, mono
    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=200),
        pb.Compressor(threshold_db=-10, ratio=3.5, attack_ms=4, release_ms=130),
        pb.Gain(gain_db=3.0),
        pb.Limiter(threshold_db=-2.0),
    ])
    buf = apply_pb(buf, bass_board)

    # Force mono (stereo_width.sub = 0.04 from mix_profile)
    mono = (buf[:, 0] + buf[:, 1]) * 0.5
    buf[:, 0] = mono
    buf[:, 1] = mono

    print(f'  Sub bass  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 5 — Pad (FAUST strings)
# ============================================================================

def build_pad(sc_gain):
    """Pad layer via FAUST — instruments.melody.pad.type=strings."""
    print('\nStep 5: Synthesizing Pad (strings) ...')

    notes = parse_track(FIXED_MID, 3)
    notes = humanize_notes(notes, timing_ms=10, vel_range=4)
    print(f'  Pad events: {len(notes)}')

    voices = separate_voices(notes)
    buf = np.zeros((NSAMP, 2), dtype=np.float32)
    for vi, voice in enumerate(voices):
        if not voice:
            continue
        freq_a, gate_a, gain_a = make_automation(voice)
        audio = faust_render(PAD_DSP, freq_a, gate_a, gain_a, vol=0.60)
        buf += audio[:NSAMP]
        print(f'  Voice {vi+1}: {len(voice)} notes')

    # Sidechain — depth 0.20
    buf[:, 0] *= (sc_gain * 0.20 + 0.80)
    buf[:, 1] *= (sc_gain * 0.20 + 0.80)

    # Intro filter sweep: LPF 200Hz->8000Hz over bars 0-8
    intro_end_samp = int(bar_to_s(INTRO_E) * SR)
    if intro_end_samp > 0 and intro_end_samp < NSAMP:
        sweep_seg = buf[:intro_end_samp].copy()
        n_chunks = INTRO_E  # one chunk per bar
        chunk_size = intro_end_samp // n_chunks
        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, intro_end_samp)
            progress = ci / n_chunks
            cutoff = 200 + progress * 7800  # 200 -> 8000 Hz
            lpf = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=cutoff)])
            chunk = sweep_seg[s:e]
            if len(chunk) > 0:
                sweep_seg[s:e] = apply_pb(chunk, lpf)
        buf[:intro_end_samp] = sweep_seg

    # Pad effects — HPF 150Hz, reverb (2.9s decay -> room_size ~0.65)
    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.65, damping=0.50, wet_level=0.30,
                  dry_level=0.85, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                      release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, pad_board)
    print(f'  Pad  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 6 — Lead (FAUST lead_synth)
# ============================================================================

def build_lead(sc_gain):
    """Lead melody via FAUST — instruments.melody.main.type=lead_synth."""
    print('\nStep 6: Synthesizing Lead ...')

    notes = parse_track(FIXED_MID, 4)
    notes = humanize_notes(notes, timing_ms=8, vel_range=4)
    print(f'  Lead events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(LEAD_DSP, freq_a, gate_a, gain_a, vol=0.55)
    buf = buf[:NSAMP]

    # Sidechain — depth 0.15
    buf[:, 0] *= (sc_gain * 0.15 + 0.85)
    buf[:, 1] *= (sc_gain * 0.15 + 0.85)

    # Intro filter sweep
    intro_end_samp = int(bar_to_s(INTRO_E) * SR)
    if intro_end_samp > 0 and intro_end_samp < NSAMP:
        sweep_seg = buf[:intro_end_samp].copy()
        n_chunks = INTRO_E
        chunk_size = intro_end_samp // n_chunks
        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, intro_end_samp)
            progress = ci / n_chunks
            cutoff = 300 + progress * 7700
            lpf = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=cutoff)])
            chunk = sweep_seg[s:e]
            if len(chunk) > 0:
                sweep_seg[s:e] = apply_pb(chunk, lpf)
        buf[:intro_end_samp] = sweep_seg

    # Lead effects
    lead_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=6000),
        pb.Reverb(room_size=0.65, damping=0.50, wet_level=0.25,
                  dry_level=0.85, width=0.80),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                      release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, lead_board)
    print(f'  Lead  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 6b — Guitar (pitch-shifted one-shot, alternate)
# ============================================================================

def pitch_shift_sample(snd, semitones):
    """Pitch shift by resampling. Positive = higher, negative = lower."""
    if semitones == 0:
        return snd.copy()
    ratio = 2.0 ** (semitones / 12.0)
    new_len = int(len(snd) / ratio)
    if new_len < 2:
        return snd[:2].copy()
    return signal.resample(snd, new_len).astype(np.float32)


def build_guitar(sc_gain):
    """Guitar melody — pitch-shifted one-shot from NO ME CONOCE_REGAE GUITAR."""
    print('\nStep 6: Building guitar ...')

    notes = parse_track(FIXED_MID, 4)
    notes = humanize_notes(notes, timing_ms=8, vel_range=4)
    print(f'  Guitar events: {len(notes)}')

    gtr_L = np.zeros(NSAMP, dtype=np.float32)
    gtr_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        semitones = note_num - GUITAR_REF_MIDI
        snd = pitch_shift_sample(GUITAR, semitones)
        g = vel / 127.0 * 0.65
        # Slight random pan for stereo interest
        pr = rng.uniform(0.40, 0.60)
        e = min(bs + len(snd), NSAMP)
        chunk = snd[:e - bs] * g
        gtr_L[bs:e] += chunk * (1 - pr) * 2
        gtr_R[bs:e] += chunk * pr * 2

    buf = np.stack([gtr_L, gtr_R], axis=1)

    # Sidechain — depth 0.15
    buf[:, 0] *= (sc_gain * 0.15 + 0.85)
    buf[:, 1] *= (sc_gain * 0.15 + 0.85)

    # Intro filter sweep
    intro_end_samp = int(bar_to_s(INTRO_E) * SR)
    if intro_end_samp > 0 and intro_end_samp < NSAMP:
        sweep_seg = buf[:intro_end_samp].copy()
        n_chunks = INTRO_E
        chunk_size = intro_end_samp // n_chunks
        for ci in range(n_chunks):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, intro_end_samp)
            progress = ci / n_chunks
            cutoff = 300 + progress * 7700
            lpf = pb.Pedalboard([pb.LowpassFilter(cutoff_frequency_hz=cutoff)])
            chunk = sweep_seg[s:e]
            if len(chunk) > 0:
                sweep_seg[s:e] = apply_pb(chunk, lpf)
        buf[:intro_end_samp] = sweep_seg

    # Guitar effects — HPF 200Hz, reverb, compression
    gtr_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.45, damping=0.55, wet_level=0.20,
                  dry_level=0.90, width=0.75),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                      release_ms=180),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, gtr_board)
    print(f'  Guitar  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# FINAL MIX
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
# MASTER CHAIN + LUFS NORMALIZATION
# ============================================================================

def master_and_export(mix, target_lufs=-12.5):
    sections = [
        ('Intro',   INTRO_S,   INTRO_E),
        ('Verse1',  VERSE1_S,  VERSE1_E),
        ('Hook1',   HOOK1_S,   HOOK1_E),
        ('Bridge1', BRIDGE1_S, BRIDGE1_E),
        ('Bridge2', BRIDGE2_S, BRIDGE2_E),
        ('Hook2',   HOOK2_S,   HOOK2_E),
        ('Outro',   OUTRO_S,   OUTRO_E),
    ]

    # Master chain
    print('Master chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=15,
                      release_ms=200),
        pb.Distortion(drive_db=0.5),
        pb.Gain(gain_db=-2.0),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix  = mix[:trim]

    # Fade out over outro section (bars 60-72)
    FADE_START_BAR = OUTRO_S
    fade_start = int(bar_to_s(FADE_START_BAR) * SR)
    fade_end   = trim
    fade_len   = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: bars {FADE_START_BAR}-{OUTRO_E} ({fade_len/SR:.1f}s)')

    # LUFS normalization
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(INTRO_E) * SR)
    measure_end   = fade_start

    # Step 1: true-peak limiter
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    # Step 2: measure body LUFS after limiting
    lufs_post_limit = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Post-limiter LUFS (body): {lufs_post_limit:.1f}')

    # Step 3: apply gain to hit target
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
        'title':  f'{BEAT_NAME} {_vstr}',
        'artist': 'Claude Code',
        'album':  BEAT_NAME,
        'genre':  'Reggaeton',
        'comment': 'A major (11B) | 100 BPM',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # === Mix analysis + per-section loudness + artifact detection ===
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
        print(f'  {name:<10} LUFS: {sec_lufs:>6.1f}  peak: {sec_peak:>6.1f} dB  '
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

    # librosa in subprocess (avoids dawdreamer/numba LLVM conflict)
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

    # Step 4: Sub Bass (FAUST)
    bass_buf = build_sub_bass(sc_gain)

    # Step 5: Pad (FAUST strings)
    pad_buf = build_pad(sc_gain)

    # Step 6: Lead (FAUST lead_synth)
    lead_buf = build_lead(sc_gain)

    # Mix — (buf, coeff, stereo_mode, stereo_param)
    # Stereo from mix_profile: sub mono, widen mids/highs
    mix = mix_stems([
        (drum_stereo, 0.62, 'center', 0),       # drums center
        (bass_buf,    0.50, 'center', 0),        # sub bass mono center
        (pad_buf,     0.35, 'widen',  20),       # pad widened
        (lead_buf,    0.26, 'widen',  14),       # lead widened
    ])

    # Master + export
    master_and_export(mix)
