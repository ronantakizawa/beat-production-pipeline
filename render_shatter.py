"""
Shatter — Render Script (Breakcore)
D minor | 170 BPM | 72 bars

MIDI track indices in Shatter_FULL.mid:
  0: tempo/metadata  1: reese bass  2: pad  3: melody  4: arp

Amen breaks are driven by Shatter_slices.json (NOT MIDI).
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

from instruments_query import SampleSelector


# ============================================================================
# CONFIG
# ============================================================================

BEAT_NAME = 'Shatter'
OUTPUT    = '/Users/ronantakizawa/Documents/Shatter_Beat'
FULL_MID  = os.path.join(OUTPUT, f'{BEAT_NAME}_FULL.mid')
FIXED_MID = os.path.join(OUTPUT, f'{BEAT_NAME}_FIXED.mid')
SLICE_JSON = os.path.join(OUTPUT, f'{BEAT_NAME}_slices.json')

_existing = _glob.glob(os.path.join(OUTPUT, f'{BEAT_NAME}_v*.mp3'))
_version  = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                 for p in _existing], default=0) + 1
_vstr     = f'v{_version}'
OUT_WAV   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.wav')
OUT_MP3   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.mp3')
print(f'Output: {OUT_MP3}')

SR    = 44100
BPM   = 170
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 72
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

# Section boundaries
INTRO_S,  INTRO_E  =  0,  8
BUILD_S,  BUILD_E  =  8, 16
DROP1_S,  DROP1_E  = 16, 32
BREAK_S,  BREAK_E  = 32, 40
DROP2_S,  DROP2_E  = 40, 56
OUTRO_S,  OUTRO_E  = 56, 72

rng = np.random.RandomState(42)

# ============================================================================
# SAMPLE PATHS
# ============================================================================

INST = '/Users/ronantakizawa/Documents/instruments'

# 12 pre-chopped amen slices at 170 BPM
AMEN_CHOPS_DIR = os.path.join(INST, 'breakcore/170/Amen 170bpm/Chops')
AMEN_SLICE_FILES = [
    '1.Amen - Kick.wav',
    '2.Amen - Kick.wav',
    '3.Amen - Snare.wav',
    '4.Amen - Hat Shuffle.wav',
    '5.Amen - Hat Shuffle.wav',
    '6.Amen - ShuffleKick.wav',
    '7.Amen - Snare.wav',
    '8.Amen - Hat Shuffle.wav',
    '9.Amen - Snare +1 Pitch.wav',
    '10.Amen - Snare +2 Pitch.wav',
    '11.Amen - Snare +3 Pitch.wav',
    '12.Amen - Snare +4 Pitch.wav',
]

# Secondary amen loop (full, for layering in drops)
AMEN_LOOP_PATH = os.path.join(
    INST, 'breaks/Amen Breaks Compilation/Amen Breaks Volume 1/WAV/cw_amen19_172.wav')

# Atmosphere FX
ENDJINN = os.path.join(INST, 'Endjinn Sample Pack - Salvage Vol. 1/FX')
NOISE_PATH      = os.path.join(ENDJINN, 'Noise/White Noise.wav')
RISER_PATH      = os.path.join(ENDJINN, 'Risers/Primordial Sludge Riser 174bpm.wav')
IMPACT_PATH     = os.path.join(ENDJINN, 'Impacts/War Drum Impact E.wav')
DOWNLIFTER_PATH = os.path.join(ENDJINN, 'Downlifters/Downlifter Emin 134bpm.wav')

# Texture drone (Endjinn Textures & Pads)
ENDJINN_TEX = os.path.join(INST, 'Endjinn Sample Pack - Salvage Vol. 1/Textures & Pads')
TEXTURE_PATH = os.path.join(ENDJINN_TEX, 'Hollow Earth Theory Soundscape.wav')

# Vocals (Endjinn)
ENDJINN_VOX = os.path.join(INST, 'Endjinn Sample Pack - Salvage Vol. 1/Vocals')
VOX_LOOP_PATH = os.path.join(ENDJINN_VOX, 'Vocal Chops & Loops/La La Vox Loop Emin 168bpm.wav')
VOX_CHANT_PATH = os.path.join(ENDJINN_VOX, 'Chants/Yah 150bpm.wav')
VOX_UNCERTAIN_PATH = os.path.join(ENDJINN_VOX, 'Vocal Chops & Loops/Uncertain Vox G#min 172bpm.wav')


# ============================================================================
# HELPERS
# ============================================================================

def load_sample(path):
    """Load a WAV sample, convert to mono float32 at SR."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0):
    """Place a mono sample into stereo buffers at sample offset start_s."""
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


# ============================================================================
# FAUST DSP
# ============================================================================

# Reese bass: 2x detuned saws + LPF + cubic distortion
REESE_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc = os.sawtooth(freq) * 0.5 + os.sawtooth(freq * 1.015) * 0.5;
env = en.adsr(0.01, 0.15, 0.85, 0.30, gate);
lfo = os.osc(0.5) * 0.5 + 0.5;
cutoff = 120.0 + lfo * 400.0;
process = osc * env * gain * 0.70 : fi.lowpass(4, cutoff)
        : ef.cubicnl(0.4, 0.0) <: _, _;
"""

# Pad: drawbar organ — additive sine harmonics, slow rotary LFO
PAD_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq) * 0.35
     + os.osc(freq * 2.0) * 0.25
     + os.osc(freq * 3.0) * 0.12
     + os.osc(freq * 4.0) * 0.06
     + os.osc(freq * 0.5) * 0.20;
env  = en.adsr(0.40, 0.50, 0.75, 2.5, gate);
lfo  = os.osc(0.8) * 0.3 + 0.7;
process = osc * env * gain * lfo * 0.45 : fi.lowpass(2, 3500) <: _, _;
"""

# Sub sine: pure sine an octave below reese, drops only
SUB_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq * 0.5);
env  = en.adsr(0.01, 0.10, 0.90, 0.20, gate);
process = osc * env * gain * 0.60 : fi.lowpass(2, 150) <: _, _;
"""

# Arp: FM bell, short decay
ARP_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
mod  = os.osc(freq * 3.01) * freq * 0.25;
osc  = os.osc(freq + mod) * 0.6 + os.osc(freq * 2.0) * 0.2;
env  = en.adsr(0.003, 0.20, 0.08, 0.50, gate);
process = osc * env * gain * 0.40 : fi.lowpass(2, 5000) <: _, _;
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
# STEP 1 — Load amen slices + samples
# ============================================================================

def load_amen_slices():
    """Load the 12 pre-chopped amen WAVs into a list."""
    print('\nStep 1: Loading amen slices ...')
    slices = []
    for fname in AMEN_SLICE_FILES:
        path = os.path.join(AMEN_CHOPS_DIR, fname)
        slices.append(load_sample(path))
        print(f'  [{len(slices)-1:>2}] {fname}  ({len(slices[-1])/SR:.3f}s)')
    return slices


# ============================================================================
# STEP 2 — Build amen breaks from slice map
# ============================================================================

def stutter_slice(audio, count, total_dur_samp):
    """Repeat a truncated slice `count` times within total_dur_samp."""
    if count <= 0:
        return audio[:total_dur_samp].copy()
    hit_len = max(1, total_dur_samp // count)
    hit = audio[:hit_len].copy()
    # Tiny fade to avoid clicks
    fade = min(32, hit_len // 4)
    if fade > 1:
        hit[-fade:] *= np.linspace(1, 0, fade)
    out = np.zeros(total_dur_samp, dtype=np.float32)
    for i in range(count):
        s = i * hit_len
        e = min(s + hit_len, total_dur_samp)
        out[s:e] = hit[:e - s]
    return out


def build_amen_breaks(amen_slices, slice_map):
    """Assemble full amen break track from JSON slice map."""
    print('\nStep 2: Building amen breaks ...')

    brk_L = np.zeros(NSAMP, dtype=np.float32)
    brk_R = np.zeros(NSAMP, dtype=np.float32)
    kick_env = np.zeros(NSAMP, dtype=np.float32)

    total_events = 0
    for bar_str, bar_data in slice_map.items():
        bar = int(bar_str)
        for ev in bar_data['events']:
            sl_idx   = ev['slice']
            beat     = ev['beat']
            dur      = ev['dur']
            vel      = ev['vel']
            reverse  = ev['reverse']
            pitch_st = ev['pitch_st']
            stutter  = ev['stutter']

            if sl_idx < 0 or sl_idx >= len(amen_slices):
                continue

            snd = amen_slices[sl_idx].copy()

            # Pitch shift
            if pitch_st != 0:
                snd = pitch_shift_sample(snd, pitch_st)

            # Reverse
            if reverse:
                snd = snd[::-1].copy()

            # Stutter
            dur_samp = int(dur * BEAT * SR)
            if stutter > 0:
                snd = stutter_slice(snd, stutter, dur_samp)
            else:
                snd = snd[:dur_samp]

            # Apply velocity
            snd = snd * vel

            # Place in buffer
            start_s = int(bar_to_s(bar, beat) * SR)
            if start_s >= NSAMP or start_s < 0:
                continue

            # Slight random pan for width
            pan = rng.uniform(-0.15, 0.15)
            g_L = 1.0 - max(0, pan)
            g_R = 1.0 + min(0, pan)
            place(brk_L, brk_R, snd, start_s, g_L, g_R)

            # Feed kick envelope for sidechain (slices 0,1,5 are kicks)
            if sl_idx in (0, 1, 5):
                e = min(start_s + len(snd), NSAMP)
                kick_env[start_s:e] += np.abs(snd[:e - start_s]) * 0.5

            total_events += 1

    # Layer secondary amen loop on accent bars only (every 4th bar in drops)
    print('  Layering secondary amen loop on accent bars ...')
    amen_loop = load_sample(AMEN_LOOP_PATH)
    loop_len = len(amen_loop)

    for s_bar, e_bar in [(DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar):
            if (bar - s_bar) % 4 != 0:
                continue
            start_s = int(bar_to_s(bar) * SR)
            bar_samp = int(BAR * SR)
            chunk_len = min(loop_len, bar_samp)
            chunk = amen_loop[:chunk_len] * 0.18
            if start_s < NSAMP:
                place(brk_L, brk_R, chunk, start_s, 0.5, 0.5)

    # Combine into stereo
    brk_stereo = np.stack([brk_L, brk_R], axis=1)

    # Post-processing: multiband distortion, heavy compression, HPF
    brk_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=100),
        pb.Distortion(drive_db=8.0),
        pb.Compressor(threshold_db=-8, ratio=6.0, attack_ms=1, release_ms=60),
        pb.Gain(gain_db=3.0),
        pb.Limiter(threshold_db=-1.5),
    ])
    brk_stereo = apply_pb(brk_stereo, brk_board)

    print(f'  {total_events} slice events placed')
    print(f'  Amen breaks  max={np.abs(brk_stereo).max():.3f}')
    return brk_stereo, kick_env


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
# STEP 4 — Reese bass (FAUST)
# ============================================================================

def build_reese_bass(sc_gain):
    print('\nStep 4: Synthesizing reese bass ...')
    notes = parse_track(FIXED_MID, 1)
    notes = humanize_notes(notes, timing_ms=6, vel_range=3)
    print(f'  Reese bass events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(REESE_DSP, freq_a, gate_a, gain_a, vol=0.75)
    buf = buf[:NSAMP]

    # Sidechain to amen kicks
    buf[:, 0] *= (sc_gain * 0.50 + 0.50)
    buf[:, 1] *= (sc_gain * 0.50 + 0.50)

    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=1200),
        pb.Compressor(threshold_db=-10, ratio=4.0, attack_ms=4, release_ms=130),
        pb.Gain(gain_db=1.5),
        pb.Limiter(threshold_db=-2.0),
    ])
    buf = apply_pb(buf, bass_board)
    print(f'  Reese bass  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 5 — Pad (FAUST)
# ============================================================================

def build_pad(sc_gain):
    print('\nStep 5: Synthesizing pad ...')
    notes = parse_track(FIXED_MID, 2)
    notes = humanize_notes(notes, timing_ms=4, vel_range=2)
    print(f'  Pad events: {len(notes)}')

    voices = separate_voices(notes)
    buf = np.zeros((NSAMP, 2), dtype=np.float32)
    for vi, voice in enumerate(voices):
        if not voice:
            continue
        freq_a, gate_a, gain_a = make_automation(voice)
        audio = faust_render(PAD_DSP, freq_a, gate_a, gain_a, vol=0.55)
        buf += audio[:NSAMP]
        print(f'  Voice {vi+1}: {len(voice)} notes')

    # Intro filter sweep: apply heavier LPF in intro bars
    intro_end_samp = int(bar_to_s(INTRO_E) * SR)
    # Simulate sweep by just filtering the intro portion more aggressively
    intro_buf = buf[:intro_end_samp].copy()
    intro_board = pb.Pedalboard([
        pb.LowpassFilter(cutoff_frequency_hz=800),
    ])
    buf[:intro_end_samp] = apply_pb(
        intro_buf, intro_board) if len(intro_buf) > 0 else intro_buf

    # Light sidechain
    buf[:, 0] *= (sc_gain * 0.10 + 0.90)
    buf[:, 1] *= (sc_gain * 0.10 + 0.90)

    pad_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=6000),
        pb.Reverb(room_size=0.80, damping=0.40, wet_level=0.45,
                  dry_level=0.70, width=0.95),
        pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=20,
                      release_ms=300),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, pad_board)
    print(f'  Pad  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 6 — Melody (sample-based, SampleSelector)
# ============================================================================

def build_melody(sc_gain):
    print('\nStep 6: Building melody ...')

    # Pick a melodic one-shot via SampleSelector
    sel = SampleSelector(genre='breakcore', beat='Shatter', seed=42, key='D')
    melody_path = sel.pick('melody')
    melody_info = sel.info['melody']
    sel.save()

    melody_snd = load_sample(melody_path)
    print(f'  Melody sample: {os.path.basename(melody_path)}')
    print(f'    key={melody_info["key"]}  pitch_st={melody_info["pitch_st"]}')

    # Apply auto pitch correction from selector
    base_pitch_st = melody_info['pitch_st']

    # Parse melody MIDI (track 3)
    notes = parse_track(FIXED_MID, 3)
    notes = humanize_notes(notes, timing_ms=6, vel_range=4)
    print(f'  Melody events: {len(notes)}')

    mel_L = np.zeros(NSAMP, dtype=np.float32)
    mel_R = np.zeros(NSAMP, dtype=np.float32)

    # D5=74 as reference pitch for the sample
    ref_midi = 74

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        # Pitch shift relative to D5 + auto correction
        st = (note_num - ref_midi) + base_pitch_st
        snd = pitch_shift_sample(melody_snd, st)

        # Trim to note duration
        trim = min(int(dur * SR), len(snd))
        chunk = snd[:trim].copy() * (vel / 127.0)
        # Fade out
        fade_n = max(1, int(SR * 0.02))
        if trim > fade_n:
            chunk[-fade_n:] *= np.linspace(1, 0, fade_n)

        # Slight random pan
        pan = rng.uniform(-0.2, 0.2)
        g_L = 1.0 - max(0, pan)
        g_R = 1.0 + min(0, pan)
        place(mel_L, mel_R, chunk, bs, g_L, g_R)

    mel_buf = np.stack([mel_L, mel_R], axis=1)

    # Sidechain
    mel_buf[:, 0] *= (sc_gain * 0.15 + 0.85)
    mel_buf[:, 1] *= (sc_gain * 0.15 + 0.85)

    mel_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=300),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Delay(delay_seconds=BEAT * 0.75, feedback=0.30, mix=0.25),
        pb.Reverb(room_size=0.65, damping=0.50, wet_level=0.35,
                  dry_level=0.80, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                      release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    mel_buf = apply_pb(mel_buf, mel_board)
    print(f'  Melody  max={np.abs(mel_buf).max():.3f}')
    return mel_buf


# ============================================================================
# STEP 7 — Atmosphere (noise, risers, impacts, downlifter)
# ============================================================================

def build_atmosphere():
    print('\nStep 7: Building atmosphere ...')

    atm_L = np.zeros(NSAMP, dtype=np.float32)
    atm_R = np.zeros(NSAMP, dtype=np.float32)

    # --- White noise: continuous, very low, filtered ---
    noise_snd = load_sample(NOISE_PATH)
    noise_len = len(noise_snd)
    # Loop noise across full song at -30dB
    noise_gain = 0.03  # ~-30dB
    pos = 0
    while pos < NSAMP:
        chunk_len = min(noise_len, NSAMP - pos)
        chunk = noise_snd[:chunk_len] * noise_gain
        atm_L[pos:pos + chunk_len] += chunk
        atm_R[pos:pos + chunk_len] += chunk
        pos += noise_len
    print('  Noise layer placed')

    # --- Risers: before drops (bars 14-16, bars 38-40) ---
    riser_snd = load_sample(RISER_PATH)
    riser_dur = len(riser_snd) / SR
    for target_bar in [16, 40]:
        # Place riser ending at drop entry, 2 bars before
        riser_start_s = max(0, int(bar_to_s(target_bar - 2) * SR))
        trim = min(len(riser_snd), int(bar_to_s(target_bar) * SR) - riser_start_s)
        if trim > 0:
            chunk = riser_snd[:trim] * 0.45
            # Fade in
            fade = min(int(0.5 * SR), trim // 2)
            chunk[:fade] *= np.linspace(0, 1, fade)
            place(atm_L, atm_R, chunk, riser_start_s, 0.48, 0.52)
    print(f'  Risers placed (dur={riser_dur:.1f}s)')

    # --- Impacts: at drop entries (bars 16, 40) ---
    impact_snd = load_sample(IMPACT_PATH)
    for bar in [16, 40]:
        s = int(bar_to_s(bar) * SR)
        place(atm_L, atm_R, impact_snd * 0.65, s, 0.50, 0.50)
    print('  Impacts placed')

    # --- Downlifter: at break entry (bar 32) ---
    dl_snd = load_sample(DOWNLIFTER_PATH)
    dl_start = int(bar_to_s(32) * SR)
    dl_trim = min(len(dl_snd), int(bar_to_s(36) * SR) - dl_start)
    if dl_trim > 0:
        place(atm_L, atm_R, dl_snd[:dl_trim] * 0.40, dl_start, 0.50, 0.50)
    print('  Downlifter placed')

    atm_buf = np.stack([atm_L, atm_R], axis=1)

    atm_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=60),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Reverb(room_size=0.75, damping=0.50, wet_level=0.30,
                  dry_level=0.85, width=1.0),
        pb.Limiter(threshold_db=-3.0),
    ])
    atm_buf = apply_pb(atm_buf, atm_board)
    print(f'  Atmosphere  max={np.abs(atm_buf).max():.3f}')
    return atm_buf


# ============================================================================
# STEP 8 — Arp (FAUST FM bell)
# ============================================================================

def build_arp(sc_gain):
    print('\nStep 8: Synthesizing arp ...')
    notes = parse_track(FIXED_MID, 4)
    notes = humanize_notes(notes, timing_ms=5, vel_range=3)
    print(f'  Arp events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(ARP_DSP, freq_a, gate_a, gain_a, vol=0.50)
    buf = buf[:NSAMP]

    buf[:, 0] *= (sc_gain * 0.10 + 0.90)
    buf[:, 1] *= (sc_gain * 0.10 + 0.90)

    arp_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=250),
        pb.LowpassFilter(cutoff_frequency_hz=7000),
        pb.Delay(delay_seconds=BEAT * 0.5, feedback=0.25, mix=0.20),
        pb.Reverb(room_size=0.70, damping=0.45, wet_level=0.40,
                  dry_level=0.75, width=0.90),
        pb.Compressor(threshold_db=-16, ratio=2.0, attack_ms=8,
                      release_ms=200),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, arp_board)
    print(f'  Arp  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 9 — Sub sine (reinforces bass in drops)
# ============================================================================

def build_sub(sc_gain):
    print('\nStep 9: Synthesizing sub sine ...')
    # Reuse bass MIDI (track 1) but render with sub sine DSP
    notes = parse_track(FIXED_MID, 1)
    print(f'  Sub events: {len(notes)}')

    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(SUB_DSP, freq_a, gate_a, gain_a, vol=0.60)
    buf = buf[:NSAMP]

    # Sidechain
    buf[:, 0] *= (sc_gain * 0.50 + 0.50)
    buf[:, 1] *= (sc_gain * 0.50 + 0.50)

    sub_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=120),
        pb.Limiter(threshold_db=-2.0),
    ])
    buf = apply_pb(buf, sub_board)
    print(f'  Sub  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# STEP 10 — Texture drone (Endjinn soundscape loop)
# ============================================================================

def build_texture():
    print('\nStep 10: Building texture drone ...')

    tex_snd = load_sample(TEXTURE_PATH)
    tex_len = len(tex_snd)
    print(f'  Texture sample: {os.path.basename(TEXTURE_PATH)} ({tex_len/SR:.1f}s)')

    tex_L = np.zeros(NSAMP, dtype=np.float32)
    tex_R = np.zeros(NSAMP, dtype=np.float32)

    # Loop across full song at very low gain
    tex_gain = 0.06
    pos = 0
    while pos < NSAMP:
        chunk_len = min(tex_len, NSAMP - pos)
        chunk = tex_snd[:chunk_len] * tex_gain
        # Crossfade at loop boundaries
        fade = min(int(0.5 * SR), chunk_len // 4)
        if pos > 0 and fade > 1:
            chunk[:fade] *= np.linspace(0, 1, fade)
        if chunk_len == tex_len and fade > 1:
            chunk[-fade:] *= np.linspace(1, 0, fade)
        tex_L[pos:pos + chunk_len] += chunk
        tex_R[pos:pos + chunk_len] += chunk
        pos += tex_len

    tex_buf = np.stack([tex_L, tex_R], axis=1)
    tex_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=100),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.85, damping=0.40, wet_level=0.50,
                  dry_level=0.60, width=1.0),
        pb.Limiter(threshold_db=-6.0),
    ])
    tex_buf = apply_pb(tex_buf, tex_board)
    print(f'  Texture  max={np.abs(tex_buf).max():.3f}')
    return tex_buf


# ============================================================================
# STEP 11 — Vocal chops
# ============================================================================

def build_vocals():
    """Vocal layer: loop in drops, chant stabs at drop entries, ambient vox in break."""
    print('\nStep 11: Building vocal chops ...')

    vox_L = np.zeros(NSAMP, dtype=np.float32)
    vox_R = np.zeros(NSAMP, dtype=np.float32)

    # --- Vocal loop: La La Vox in drops, looped per 2 bars ---
    vox_loop = load_sample(VOX_LOOP_PATH)
    loop_len = len(vox_loop)
    print(f'  Vox loop: {os.path.basename(VOX_LOOP_PATH)} ({loop_len/SR:.1f}s)')

    for s_bar, e_bar in [(DROP1_S, DROP1_E), (DROP2_S, DROP2_E)]:
        for bar in range(s_bar, e_bar, 2):
            start_s = int(bar_to_s(bar) * SR)
            two_bars = int(2 * BAR * SR)
            chunk_len = min(loop_len, two_bars)
            chunk = vox_loop[:chunk_len] * 0.18
            # Fade edges
            fade = min(int(0.03 * SR), chunk_len // 4)
            if fade > 1:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            if start_s < NSAMP:
                place(vox_L, vox_R, chunk, start_s, 0.45, 0.55)

    # --- Chant stabs: "Yah" at drop entries + every 8 bars in drops ---
    chant = load_sample(VOX_CHANT_PATH)
    chant_bars = [DROP1_S, DROP1_S + 8, DROP1_S + 12,
                  DROP2_S, DROP2_S + 8, DROP2_S + 12]
    for bar in chant_bars:
        s = int(bar_to_s(bar) * SR)
        if s < NSAMP:
            place(vox_L, vox_R, chant * 0.30, s, 0.50, 0.50)
    print(f'  Chant stabs: {len(chant_bars)} placements')

    # --- Uncertain vox: ambient texture in break section ---
    uncertain = load_sample(VOX_UNCERTAIN_PATH)
    unc_start = int(bar_to_s(BREAK_S) * SR)
    unc_end = int(bar_to_s(BREAK_E) * SR)
    unc_len = min(len(uncertain), unc_end - unc_start)
    if unc_len > 0:
        chunk = uncertain[:unc_len] * 0.15
        fade = min(int(0.2 * SR), unc_len // 4)
        if fade > 1:
            chunk[:fade] *= np.linspace(0, 1, fade)
            chunk[-fade:] *= np.linspace(1, 0, fade)
        place(vox_L, vox_R, chunk, unc_start, 0.48, 0.52)
    print('  Uncertain vox in break section')

    vox_buf = np.stack([vox_L, vox_R], axis=1)
    vox_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=300),
        pb.LowpassFilter(cutoff_frequency_hz=10000),
        pb.Delay(delay_seconds=BEAT * 0.375, feedback=0.20, mix=0.15),
        pb.Reverb(room_size=0.70, damping=0.45, wet_level=0.35,
                  dry_level=0.80, width=0.90),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                      release_ms=180),
        pb.Limiter(threshold_db=-3.0),
    ])
    vox_buf = apply_pb(vox_buf, vox_board)
    print(f'  Vocals  max={np.abs(vox_buf).max():.3f}')
    return vox_buf


# ============================================================================
# STEP 12 — Transition FX (reverse amen tails, accelerating snare rolls)
# ============================================================================

def build_transition_fx(amen_slices):
    print('\nStep 12: Building transition FX ...')
    fx_L = np.zeros(NSAMP, dtype=np.float32)
    fx_R = np.zeros(NSAMP, dtype=np.float32)

    snare = amen_slices[2]  # snare slice

    def snare_roll(target_bar, bars_build=2.0):
        """Accelerating snare roll: 1/8 -> 1/32 over bars_build bars."""
        total_beats = int(bars_build * 4)
        densities = [2, 3, 4, 6, 8]
        for beat_i in range(total_beats):
            progress = beat_i / total_beats
            d_idx    = min(int(progress * len(densities)), len(densities) - 1)
            n_hits   = densities[d_idx]
            for h in range(n_hits):
                t   = bar_to_s(target_bar - bars_build, (beat_i + h / n_hits) * 4)
                vel = (0.25 + 0.65 * progress) * rng.uniform(0.90, 1.10)
                s   = int(t * SR)
                if 0 <= s < NSAMP:
                    place(fx_L, fx_R, snare * vel * 0.50, s, 0.50, 0.50)

    def reverse_tail(target_bar, length_beats=2):
        """Reverse amen tail with heavy reverb ending at target_bar."""
        tail_len = int(length_beats * BEAT * SR)
        padded   = np.zeros(tail_len, dtype=np.float32)
        sn_len   = min(len(snare), tail_len)
        padded[:sn_len] = snare[:sn_len] * 0.5
        tail_verb = pb.Pedalboard([
            pb.Reverb(room_size=0.92, damping=0.25, wet_level=0.95,
                      dry_level=0.0, width=1.0),
        ])
        wet  = tail_verb(padded[np.newaxis, :], SR)[0]
        rev  = wet[::-1].copy()
        fade = int(0.04 * SR)
        if fade > 0 and len(rev) > fade:
            rev[:fade] *= np.linspace(0, 1, fade)
        end_s   = int(bar_to_s(target_bar) * SR)
        start_s = max(0, end_s - len(rev))
        chunk   = rev[:end_s - start_s]
        place(fx_L, fx_R, chunk, start_s, 0.48, 0.52)

    # Before drop1 (bar 16)
    snare_roll(DROP1_S, bars_build=2.0)
    reverse_tail(DROP1_S, length_beats=2)

    # Before drop2 (bar 40)
    snare_roll(DROP2_S, bars_build=2.0)
    reverse_tail(DROP2_S, length_beats=2)

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
    print(f'  Transition FX  max={np.abs(fx_buf).max():.3f}')
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
# MASTER CHAIN + LUFS NORMALIZATION
# ============================================================================

def master_and_export(mix, target_lufs=-12.0):
    sections = [
        ('Intro',  INTRO_S, INTRO_E),
        ('Build',  BUILD_S, BUILD_E),
        ('Drop 1', DROP1_S, DROP1_E),
        ('Break',  BREAK_S, BREAK_E),
        ('Drop 2', DROP2_S, DROP2_E),
        ('Outro',  OUTRO_S, OUTRO_E),
    ]

    print('\nMaster chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-10, ratio=2.5, attack_ms=12,
                      release_ms=180),
        pb.Distortion(drive_db=3.0),
        pb.Gain(gain_db=1.0),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix  = mix[:trim]

    # Fade out over bars 56-71 (entire outro)
    FADE_START_BAR = OUTRO_S
    fade_start = int(bar_to_s(FADE_START_BAR) * SR)
    fade_end   = trim
    fade_len   = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: bars {FADE_START_BAR}-{OUTRO_E} ({fade_len/SR:.1f}s)')

    # LUFS normalization on body (drops)
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(DROP1_S) * SR)
    measure_end   = int(bar_to_s(DROP1_E) * SR)
    measure_region = mix[measure_start:measure_end]

    lufs_before = ln_meter.integrated_loudness(measure_region)
    print(f'  Pre-norm LUFS (drop1): {lufs_before:.1f}')
    if np.isfinite(lufs_before):
        gain_db = target_lufs - lufs_before
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    # Final limiter
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    lufs_after = ln_meter.integrated_loudness(mix[measure_start:measure_end])
    print(f'  Post-norm LUFS (drop1): {lufs_after:.1f}  (target: {target_lufs})')

    # Export
    print('Exporting ...')
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title':  f'{BEAT_NAME} {_vstr}',
        'artist': 'Claude Code',
        'album':  BEAT_NAME,
        'genre':  'Breakcore',
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

    # librosa analysis in subprocess
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

    # Step 1: Load amen slices
    amen_slices = load_amen_slices()

    # Load slice map
    with open(SLICE_JSON) as f:
        slice_map = json.load(f)

    # Step 2: Amen breaks
    amen_buf, kick_env = build_amen_breaks(amen_slices, slice_map)

    # Step 3: Sidechain
    sc_gain = build_sidechain(kick_env)

    # Step 4: Reese bass
    bass_buf = build_reese_bass(sc_gain)

    # Step 5: Pad
    pad_buf = build_pad(sc_gain)

    # Step 6: Melody
    mel_buf = build_melody(sc_gain)

    # Step 7: Atmosphere
    atm_buf = build_atmosphere()

    # Step 8: Arp
    arp_buf = build_arp(sc_gain)

    # Step 9: Sub sine
    sub_buf = build_sub(sc_gain)

    # Step 10: Texture drone
    tex_buf = build_texture()

    # Step 11: Vocals
    vox_buf = build_vocals()

    # Step 12: Transition FX
    fx_buf = build_transition_fx(amen_slices)

    # Mix — (buf, coeff, stereo_mode, stereo_param)
    mix = mix_stems([
        (amen_buf,  0.70, 'widen',  16),    # Centerpiece, distorted
        (bass_buf,  0.50, 'center', 0),     # Mono reese bass
        (sub_buf,   0.30, 'center', 0),     # Sub sine reinforcement
        (pad_buf,   0.25, 'widen',  22),    # Organ pad
        (arp_buf,   0.20, 'widen',  18),    # FM bell arp
        (mel_buf,   0.18, 'widen',  14),    # Quiet melody
        (vox_buf,   0.22, 'widen',  16),    # Vocal chops
        (atm_buf,   0.15, 'widen',  24),    # Noise/risers/impacts
        (tex_buf,   0.12, 'widen',  26),    # Soundscape drone
        (fx_buf,    0.45, 'center', 0),     # Transitions
    ])

    # Master + export (target LUFS -12)
    master_and_export(mix, target_lufs=-12.0)
