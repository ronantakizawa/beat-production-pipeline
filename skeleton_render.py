"""
[Beat Name] — Render Script
[KEY] | [BPM] BPM | [NBARS] bars

Kit: [kit name(s)]

MIDI track indices in [BEAT]_FULL.mid:
  0: tempo/metadata  1: drums  2: 808 bass
  3: [layer]         4: [layer]  ...
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
# CONFIG — edit these for each beat
# ============================================================================

BEAT_NAME = 'MyBeat'
OUTPUT    = '/Users/ronantakizawa/Documents/MyBeat_Beat'
FULL_MID  = os.path.join(OUTPUT, f'{BEAT_NAME}_FULL.mid')
FIXED_MID = os.path.join(OUTPUT, f'{BEAT_NAME}_FIXED.mid')

# Auto-versioning: finds existing vN files, increments
_existing = _glob.glob(os.path.join(OUTPUT, f'{BEAT_NAME}_v*.mp3'))
_version  = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                 for p in _existing], default=0) + 1
_vstr     = f'v{_version}'
OUT_WAV   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.wav')
OUT_MP3   = os.path.join(OUTPUT, f'{BEAT_NAME}_{_vstr}.mp3')
print(f'Output: {OUT_MP3}')

SR    = 44100
BPM   = 140       # must match compose script
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 64
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

# Section boundaries (must match compose script)
INTRO_S,  INTRO_E  =  0,  8
HOOKA_S,  HOOKA_E  =  8, 24
VERSE_S,  VERSE_E  = 24, 40
BRIDGE_S, BRIDGE_E = 40, 48
HOOKB_S,  HOOKB_E  = 48, 64

rng = np.random.RandomState(42)


# ============================================================================
# HELPERS — reuse as-is across all beats
# ============================================================================

def load_sample(path):
    """Load a WAV sample, convert to mono float32 at SR."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g    = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)


def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0):
    """Place a mono sample into stereo buffers at sample offset start_s."""
    e = min(start_s + len(snd), NSAMP)
    if e <= start_s:
        return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R


def apply_pb(arr2ch, board):
    """Apply a pedalboard effects chain to (NSAMP, 2) array."""
    out = board(arr2ch.T.astype(np.float32), SR)
    return out.T.astype(np.float32)


def midi_to_hz(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))


def parse_track(mid_path, track_idx):
    """Extract note events from a MIDI track.
    Returns list of (start_sec, midi_note, velocity, duration_sec)."""
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
    """Create freq/gate/gain automation arrays for FAUST from note list."""
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
    """Add timing jitter and velocity randomization."""
    result = []
    jitter_samp = timing_ms / 1000.0
    for start, note_num, vel, dur in notes:
        t_jitter = rng.uniform(-jitter_samp, jitter_samp)
        v_jitter = rng.randint(-vel_range, vel_range + 1)
        result.append((max(0, start + t_jitter), note_num,
                        int(np.clip(vel + v_jitter, 1, 127)), dur))
    return result


def faust_render(dsp_string, freq_arr, gate_arr, gain_arr, vol=1.0):
    """Render audio from FAUST DSP code + automation arrays.
    Returns (NSAMP, 2) stereo array."""
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
    """Split chords into up to 4 monophonic voices for FAUST rendering."""
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
    """Convert bar number + beat to seconds."""
    return (bar + beat / 4.0) * BAR


def pan_stereo(buf, position):
    """Pan stereo buffer. position: -1 (left) to +1 (right), 0 = center."""
    angle = (position + 1) * np.pi / 4
    result = buf.copy()
    result[:, 0] *= np.cos(angle)
    result[:, 1] *= np.sin(angle)
    return result


def stereo_widen(buf, delay_ms=12):
    """Haas-effect stereo widening by delaying the right channel."""
    d = int(delay_ms / 1000.0 * SR)
    if d <= 0 or d >= len(buf):
        return buf.copy()
    result = buf.copy()
    result[d:, 1] = buf[:-d, 1]
    result[:d, 1] *= 0.3
    return result


# ============================================================================
# FAUST DSP LIBRARY — pick and customize per beat
# ============================================================================
# Each DSP must expose: freq (Hz), gain (0-1), gate (0/1)
# Include a TIMBRE RULE comment for each to prevent drift across iterations.

# --- PAD: 5x detuned saw, slow attack, long release ---
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

# --- PIANO: triangle+sine, short envelope, plucked/bell-like ---
PIANO_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.triangle(freq) * 0.6 + os.osc(freq * 2.0) * 0.15 + os.osc(freq * 3.0) * 0.05;
env  = en.adsr(0.005, 0.35, 0.15, 0.8, gate);
process = osc * env * gain * 0.50 : fi.lowpass(2, 4000) <: _, _;
"""

# --- BELL: FM synthesis, ethereal ---
BELL_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
mod  = os.osc(freq * 2.01) * freq * 0.3;
osc  = os.osc(freq + mod) + os.osc(freq * 0.5) * 0.3;
env  = en.adsr(0.002, 0.30, 0.12, 1.0, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 2500) <: _, _;
"""

# --- PLUCK: triangle+sine, very fast decay ---
PLUCK_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.triangle(freq) * 0.55 + os.osc(freq * 2.0) * 0.25 + os.osc(freq * 3.0) * 0.10;
env  = en.adsr(0.002, 0.08, 0.05, 0.10, gate);
process = osc * env * gain * 0.50 : fi.lowpass(2, 6000) <: _, _;
"""

# --- BRASS: bright saw stab, fast ADSR ---
BRASS_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = (os.sawtooth(freq) + os.sawtooth(freq * 1.005) + os.square(freq * 0.998)) * 0.33;
env  = en.adsr(0.001, 0.06, 0.30, 0.08, gate);
process = osc * env * gain * 0.55 : fi.lowpass(2, 5000) <: _, _;
"""

# --- FLUTE: sine + breath noise ---
FLUTE_DSP = """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq) * 0.7 + no.noise * 0.03;
env  = en.adsr(0.08, 0.20, 0.60, 0.40, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 3500) <: _, _;
"""


# ============================================================================
# STEP 0 — Fix MIDI (deduplicate tempo messages)
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
# Define your sample paths here. Example using rap2 kit:
#
# RAP2 = '/Users/ronantakizawa/Documents/instruments/rap2'
# KICK   = load_sample(f'{RAP2}/Kicks/(Kick) - Top.wav')
# CLAP   = load_sample(f'{RAP2}/Claps/(Clap) - Tab.wav')
# SNARE  = load_sample(f'{RAP2}/Snares/(Snare) - Space.wav')
# HH_CL  = load_sample(f'{RAP2}/HiHats/(HH) - Classic.wav')
# HH_OP  = load_sample(f'{RAP2}/Open Hats/(OH) - Frida.wav')
# BASS_808 = load_sample(f'{RAP2}/808s/(808) - Storm.wav')
# RIM    = load_sample(f'{RAP2}/Percs_Rims/(Rim) - Blow.wav')
# CRASH  = load_sample(f'{LEX}/LEX TRAP Crash(1).wav')


# ============================================================================
# STEP 2 — Drums (sample placement + room IR)
# ============================================================================

def build_drums(drum_events, samples, intro_filter=None):
    """Build drum stereo buffer from MIDI events and sample dict.

    Args:
        drum_events: list from parse_track()
        samples:     dict mapping GM note numbers to (sample_array, gain, label)
                     e.g. {36: (KICK, 1.05, 'kick'), 39: (CLAP, 0.80, 'clap')}
        intro_filter: if set, a function(sec, note_num) -> bool that returns
                      True to SKIP the event (e.g., skip non-crash in intro)

    Returns:
        drum_stereo: (NSAMP, 2) processed drum buffer
        kick_env:    kick envelope for sidechain
    """
    print('\nStep 2: Building drum track ...')

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

    MAX_JITTER = int(0.006 * SR)
    pan_toggle = False

    for sec, note_num, vel, _ in drum_events:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        if intro_filter and intro_filter(sec, note_num):
            continue
        g = vel / 127.0

        if note_num not in samples:
            continue

        snd_raw, gain, label = samples[note_num]

        if note_num == 36:  # Kick — no jitter, feed sidechain
            snd   = snd_raw * g * gain
            chunk = snd[:min(len(snd), NSAMP - bs)]
            e     = bs + len(chunk)
            kick_env[bs:e] += np.abs(chunk)
            kick_L[bs:e]   += chunk * 0.96
            kick_R[bs:e]   += chunk * 0.96
        elif note_num == 42:  # Closed HH — alternating pan
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
        elif note_num in (46, 49):  # Open HH / Crash — slight pan
            snd = snd_raw * g * gain
            place(nk_L, nk_R, snd, bs, 0.44, 0.56)
        else:  # Snare, clap, rim, etc. — jittered, center
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
    print(f'  {len(drum_events)} drum events')
    return drum_stereo, kick_env


# ============================================================================
# STEP 3 — Sidechain
# ============================================================================

def build_sidechain(kick_env):
    """Create sidechain gain curve from kick envelope."""
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

def build_808(bass_808_sample, bass_notes, sc_gain):
    """Pitch-shift 808 to chord roots and place from MIDI.

    Args:
        bass_808_sample: raw 808 sample array
        bass_notes:      list from parse_track()
        sc_gain:         sidechain gain array
    """
    print('\nStep 4: Building 808 bass ...')

    # Detect root pitch
    _fft  = np.abs(np.fft.rfft(bass_808_sample * np.hanning(len(bass_808_sample))))
    _freq = np.fft.rfftfreq(len(bass_808_sample), 1 / SR)
    _mask = (_freq > 20) & (_freq < 500)
    _peak = _freq[_mask][np.argmax(_fft[_mask])]
    root_midi = int(round(12 * np.log2(_peak / 440) + 69))
    print(f'  808 root: MIDI {root_midi} ({_peak:.1f} Hz)')

    # Collect unique target pitches from MIDI
    target_midis = sorted(set(n[1] for n in bass_notes))
    pitched = {}
    for t in target_midis:
        st = t - root_midi
        if st == 0:
            pitched[t] = bass_808_sample.copy()
        else:
            board_p = pb.Pedalboard([pb.PitchShift(semitones=st)])
            pitched[t] = board_p(bass_808_sample[np.newaxis, :], SR)[0].astype(np.float32)

    bass_L = np.zeros(NSAMP, dtype=np.float32)
    bass_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, midi_note, vel, dur_sec in bass_notes:
        s = int(sec * SR)
        if s >= NSAMP:
            continue
        target = min(pitched.keys(), key=lambda t: abs(t - midi_note))
        snd    = pitched[target]
        g      = (vel / 127.0) * rng.uniform(0.92, 1.08)
        trim   = min(int(dur_sec * SR), len(snd))
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

    bass_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=30),
        pb.LowpassFilter(cutoff_frequency_hz=1200),
        pb.Distortion(drive_db=2.5),
        pb.Compressor(threshold_db=-10, ratio=3.5, attack_ms=4, release_ms=130),
        pb.Gain(gain_db=1.5),
        pb.Limiter(threshold_db=-2.0),
    ])
    bass_buf = apply_pb(bass_buf, bass_board)
    print(f'  808 events={len(bass_notes)}  max={np.abs(bass_buf).max():.3f}')
    return bass_buf


# ============================================================================
# STEP 5+ — Synth layers (FAUST)
# ============================================================================

def render_synth_layer(label, track_idx, dsp, sc_gain,
                       vol=0.65, humanize_ms=6, humanize_vel=3,
                       sc_depth=0.15, polyphonic=False,
                       fx_board=None):
    """Render a FAUST synth layer from MIDI track.

    Args:
        label:       display name
        track_idx:   MIDI track index in FIXED_MID
        dsp:         FAUST DSP string
        sc_gain:     sidechain gain array
        vol:         synth volume
        humanize_ms: timing jitter in ms
        humanize_vel: velocity jitter range
        sc_depth:    sidechain depth (0 = no SC, 1 = full SC)
        polyphonic:  True for chords (separate_voices), False for mono
        fx_board:    pedalboard.Pedalboard for post-processing
    """
    step = 5 + track_idx - 3  # auto-number steps
    print(f'\nStep {step}: Synthesizing {label} ...')

    notes = parse_track(FIXED_MID, track_idx)
    notes = humanize_notes(notes, timing_ms=humanize_ms, vel_range=humanize_vel)
    print(f'  {label} events: {len(notes)}')

    if polyphonic:
        voices = separate_voices(notes)
        buf = np.zeros((NSAMP, 2), dtype=np.float32)
        for vi, voice in enumerate(voices):
            if not voice:
                continue
            freq_a, gate_a, gain_a = make_automation(voice)
            audio = faust_render(dsp, freq_a, gate_a, gain_a, vol=vol)
            buf += audio[:NSAMP]
            print(f'  Voice {vi+1}: {len(voice)} notes')
    else:
        freq_a, gate_a, gain_a = make_automation(notes)
        buf = faust_render(dsp, freq_a, gate_a, gain_a, vol=vol)
        buf = buf[:NSAMP]

    # Sidechain
    buf[:, 0] *= (sc_gain * sc_depth + (1 - sc_depth))
    buf[:, 1] *= (sc_gain * sc_depth + (1 - sc_depth))

    # Post-processing
    if fx_board is None:
        fx_board = pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=200),
            pb.LowpassFilter(cutoff_frequency_hz=8000),
            pb.Reverb(room_size=0.60, damping=0.50, wet_level=0.30,
                      dry_level=0.85, width=0.85),
            pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6,
                          release_ms=180),
            pb.Gain(gain_db=0.5),
            pb.Limiter(threshold_db=-3.0),
        ])
    buf = apply_pb(buf, fx_board)
    print(f'  {label}  max={np.abs(buf).max():.3f}')
    return buf


# ============================================================================
# TRANSITION FX — snare rolls, reverse tails, crash hits
# ============================================================================

def build_transition_fx(snare_sample, crash_sample):
    """Build transition effects between sections."""
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
                    place(fx_L, fx_R, snare_sample * vel * 0.55, s, 0.50, 0.50)

    def reverse_tail(target_bar, length_beats=3):
        tail_len = int(length_beats * BEAT * SR)
        padded   = np.zeros(tail_len, dtype=np.float32)
        sn_len   = min(len(snare_sample), tail_len)
        padded[:sn_len] = snare_sample[:sn_len] * 0.5
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

    # Customize transitions per beat:
    # Intro -> Hook A
    snare_roll(HOOKA_S, bars_build=1.0)
    reverse_tail(HOOKA_S, length_beats=3)
    place(fx_L, fx_R, crash_sample * 0.72,
          int(bar_to_s(HOOKA_S) * SR), 0.44, 0.56)

    # Hook A -> Verse
    place(fx_L, fx_R, crash_sample * 0.45,
          int(bar_to_s(HOOKA_E) * SR), 0.44, 0.56)

    # Verse -> Bridge
    reverse_tail(BRIDGE_S, length_beats=2)

    # Bridge -> Hook B
    snare_roll(HOOKB_S, bars_build=2.0)
    reverse_tail(HOOKB_S, length_beats=3)
    place(fx_L, fx_R, crash_sample * 0.72,
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
# FINAL MIX — stereo panning + level balancing
# ============================================================================

def mix_stems(stems):
    """Mix stem buffers with stereo processing.

    Args:
        stems: list of (buf, mix_coeff, stereo_mode, stereo_param)
               stereo_mode: 'center', 'widen', or 'pan'
               stereo_param: delay_ms for widen, position for pan

    Returns:
        (NSAMP, 2) mixed stereo array
    """
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

def master_and_export(mix, target_lufs=-14.0, sections=None):
    """Apply master chain, LUFS normalization, and export WAV+MP3.

    Args:
        mix:         (NSAMP, 2) stereo array
        target_lufs: target integrated loudness
        sections:    list of (name, start_bar, end_bar) for per-section analysis
    """
    if sections is None:
        sections = [
            ('Intro',   INTRO_S,  INTRO_E),
            ('Hook A',  HOOKA_S,  HOOKA_E),
            ('Verse',   VERSE_S,  VERSE_E),
            ('Bridge',  BRIDGE_S, BRIDGE_E),
            ('Hook B',  HOOKB_S,  HOOKB_E),
        ]

    # Master chain (NO limiter — applied after LUFS norm so they don't fight)
    print('Master chain ...')
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=28),
        pb.LowpassFilter(cutoff_frequency_hz=18000),
        pb.Compressor(threshold_db=-12, ratio=2.5, attack_ms=15,
                      release_ms=200),
        pb.Distortion(drive_db=1.5),
        pb.Gain(gain_db=1.0),
    ])
    mix = apply_pb(mix, master_board)
    trim = int((SONG + 2.0) * SR)
    mix  = mix[:trim]

    # Fade out over last 4 bars
    FADE_BARS  = 4
    fade_start = int(bar_to_s(NBARS - FADE_BARS) * SR)
    fade_end   = trim
    fade_len   = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve
        print(f'  Fade out: last {FADE_BARS} bars ({fade_len/SR:.1f}s)')

    # LUFS normalization — measure on non-faded body only
    print('LUFS normalization ...')
    ln_meter = pyln.Meter(SR, block_size=0.400)
    measure_start = int(bar_to_s(INTRO_E) * SR)
    measure_end   = fade_start
    measure_region = mix[measure_start:measure_end]

    lufs_before = ln_meter.integrated_loudness(measure_region)
    print(f'  Pre-norm LUFS (body): {lufs_before:.1f}')
    if np.isfinite(lufs_before):
        gain_db = target_lufs - lufs_before
        mix = mix * (10 ** (gain_db / 20.0))
        print(f'  Applied {gain_db:+.1f} dB gain')

    # Final true-peak limiter AFTER normalization
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

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
        'genre':  'Trap',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB  |  {m}:{s:02d}')

    # === Mix analysis + per-section loudness + artifact detection ===
    print('\n== Mix Analysis ==')

    # Global metrics
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
# MAIN — wire it all together (customize per beat)
# ============================================================================

if __name__ == '__main__':
    fix_midi()

    # Step 1: Load samples
    # KICK = load_sample(...)
    # ...

    # Step 2: Drums
    # drum_events = parse_track(FIXED_MID, 1)
    # drum_stereo, kick_env = build_drums(drum_events, {
    #     36: (KICK,  1.05, 'kick'),
    #     39: (CLAP,  0.80, 'clap'),
    #     38: (SNARE, 0.65, 'snare'),
    #     42: (HH_CL, 0.48, 'hh'),
    #     46: (HH_OP, 0.80, 'oh'),
    #     49: (CRASH, 0.58, 'crash'),
    #     37: (RIM,   0.45, 'rim'),
    # }, intro_filter=lambda sec, n: sec < bar_to_s(INTRO_E) and n != 49)

    # Step 3: Sidechain
    # sc_gain = build_sidechain(kick_env)

    # Step 4: 808
    # bass_buf = build_808(BASS_808, parse_track(FIXED_MID, 2), sc_gain)

    # Steps 5+: Synth layers
    # pad_buf = render_synth_layer('Pad', 3, PAD_DSP, sc_gain,
    #     vol=0.70, polyphonic=True, sc_depth=0.20,
    #     fx_board=pb.Pedalboard([...]))
    #
    # pluck_buf = render_synth_layer('Pluck', 4, PLUCK_DSP, sc_gain,
    #     vol=0.65, sc_depth=0.15)

    # Transition FX
    # fx_buf = build_transition_fx(SNARE, CRASH)

    # Mix — (buf, coeff, stereo_mode, stereo_param)
    # mix = mix_stems([
    #     (drum_stereo, 0.90, 'center', 0),
    #     (bass_buf,    0.72, 'center', 0),
    #     (pad_buf,     0.50, 'widen',  20),
    #     (pluck_buf,   0.38, 'widen',  14),
    #     (piano_buf,   0.35, 'widen',  16),
    #     (bell_buf,    0.28, 'widen',  18),
    #     (fx_buf,      0.50, 'center', 0),
    # ])

    # Master + export
    # master_and_export(mix)

    print('Skeleton render — uncomment and customize the steps above.')
