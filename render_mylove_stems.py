"""
MyLove — Render instrument variations for melody and pad
Outputs separate MP3s for each variant into MyLove_Beat/stems/
"""

import os
import numpy as np
from math import gcd
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import mido
import pedalboard as pb
import dawdreamer as daw
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT = '/Users/ronantakizawa/Documents/MyLove_Beat'
STEMS  = os.path.join(OUTPUT, 'stems')
os.makedirs(STEMS, exist_ok=True)

FIXED_MID = os.path.join(OUTPUT, 'MyLove_FIXED.mid')

SR    = 44100
BPM   = 100
BEAT  = 60.0 / BPM
BAR   = BEAT * 4
NBARS = 72
SONG  = NBARS * BAR
NSAMP = int((SONG + 4.0) * SR)

rng = np.random.RandomState(42)


# ============================================================================
# HELPERS (from render_mylove.py)
# ============================================================================

def load_sample(path):
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)

def midi_to_hz(n):
    return 440.0 * (2 ** ((n - 69) / 12.0))

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

def make_automation(notes, retrigger_ms=8):
    """Build freq/gate/gain automation arrays.
    retrigger_ms: brief gate-off gap before each note to force ADSR retrigger.
    """
    retrig = int(retrigger_ms / 1000.0 * SR)
    freq_arr = np.zeros(NSAMP, dtype=np.float32)
    gate_arr = np.zeros(NSAMP, dtype=np.float32)
    gain_arr = np.ones(NSAMP, dtype=np.float32)
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
    synth = engine.make_faust_processor('s')
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
        ch = sorted(groups[key], key=lambda x: x[0])
        for i, n in enumerate(ch[:4]):
            voices[i].append(n)
    return voices

def apply_pb(arr2ch, board):
    out = board(arr2ch.T.astype(np.float32), SR)
    return out.T.astype(np.float32)

def place(buf_L, buf_R, snd, start_s, gain_L=1.0, gain_R=1.0):
    e = min(start_s + len(snd), NSAMP)
    if e <= start_s:
        return
    chunk = snd[:e - start_s]
    buf_L[start_s:e] += chunk * gain_L
    buf_R[start_s:e] += chunk * gain_R

def export_stem(buf, name):
    """Normalize and export a stem as MP3."""
    trim = int((SONG + 2.0) * SR)
    buf = buf[:trim]
    peak = np.abs(buf).max()
    if peak > 0.01:
        buf = buf * (0.8 / peak)  # normalize to -2 dBFS
    wav_path = os.path.join(STEMS, f'{name}.wav')
    mp3_path = os.path.join(STEMS, f'{name}.mp3')
    out_i16 = (buf * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(wav_path, SR, out_i16)
    seg = AudioSegment.from_wav(wav_path)
    seg.export(mp3_path, format='mp3', bitrate='192k')
    os.remove(wav_path)
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'  -> {name}.mp3  ({m}:{s:02d})')


# ============================================================================
# MELODY VARIANTS (FAUST DSP)
# ============================================================================

MELODY_VARIANTS = {
    'melody_1_sineFM': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
mod  = os.osc(freq * 2.01) * freq * 0.3;
osc  = os.osc(freq + mod) + os.osc(freq * 0.5) * 0.3;
env  = en.adsr(0.01, 0.15, 0.65, 1.2, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 2500) <: _, _;
""",

    'melody_2_bell': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
mod1 = os.osc(freq * 3.0) * freq * 0.8;
mod2 = os.osc(freq * 5.01) * freq * 0.2;
osc  = os.osc(freq + mod1 + mod2) * 0.7 + os.osc(freq * 2.0) * 0.3;
env  = en.adsr(0.001, 0.5, 0.30, 2.5, gate);
process = osc * env * gain * 0.40 : fi.lowpass(2, 5000) <: _, _;
""",

    'melody_3_pluck': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
noise = no.noise * 0.5;
osc   = os.osc(freq) + os.osc(freq * 1.002) * 0.5;
env   = en.adsr(0.001, 0.20, 0.40, 1.0, gate);
mix   = (osc * 0.7 + noise * env * 0.3);
process = mix * env * gain * 0.50 : fi.lowpass(2, freq * 3.0 + 200) <: _, _;
""",

    'melody_4_warmSine': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq) * 0.6 + os.osc(freq * 2.0) * 0.25 + os.osc(freq * 0.5) * 0.15;
env  = en.adsr(0.03, 0.25, 0.65, 1.5, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 1800) <: _, _;
""",

    'melody_5_squareLead': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.square(freq) * 0.5 + os.square(freq * 0.999) * 0.3 + os.osc(freq * 0.5) * 0.2;
env  = en.adsr(0.005, 0.20, 0.60, 1.0, gate);
process = osc * env * gain * 0.35 : fi.lowpass(2, 2000) <: _, _;
""",
}


# ============================================================================
# PAD VARIANTS (FAUST DSP)
# ============================================================================

PAD_VARIANTS = {
    'pad_1_detunedSaw': """
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
""",

    'pad_2_warmSine': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = os.osc(freq) * 0.5 + os.osc(freq * 2.0) * 0.2
     + os.osc(freq * 0.999) * 0.3;
env  = en.adsr(0.80, 0.50, 0.75, 4.0, gate);
process = osc * env * gain * 0.45 : fi.lowpass(2, 1200) <: _, _;
""",

    'pad_3_squarePad': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = (os.square(freq) * 0.4
      + os.square(freq * 1.005) * 0.3
      + os.square(freq * 0.995) * 0.3);
env  = en.adsr(0.60, 0.50, 0.70, 3.5, gate);
lfo  = os.osc(0.05) * 0.5 + 0.5;
cutoff = 300.0 + lfo * 500.0;
process = osc * env * gain * 0.35 : fi.lowpass(2, cutoff) <: _, _;
""",

    'pad_4_brightStrings': """
import("stdfaust.lib");
freq = hslider("freq[unit:Hz]", 440, 0.001, 20000, 0.001);
gain = hslider("gain", 1, 0, 1, 0.01);
gate = button("gate");
osc  = (os.sawtooth(freq)
      + os.sawtooth(freq * 1.006)
      + os.sawtooth(freq * 0.994)
      + os.sawtooth(freq * 1.012)
      + os.sawtooth(freq * 0.988)) * 0.2;
env  = en.adsr(0.30, 0.40, 0.75, 2.5, gate);
process = osc * env * gain * 0.40 : fi.lowpass(2, 3000) <: _, _;
""",
}


# ============================================================================
# GUITAR VARIANT (sample-based)
# ============================================================================

INST = '/Users/ronantakizawa/Documents/instruments'
GUITAR_SAMPLE = load_sample(os.path.join(INST, 'reggaetondrums/Percu/Clasicos/NO ME CONOCE_REGAE GUITAR.wav'))
GUITAR_REF_MIDI = 61  # C#4

def pitch_shift_sample(snd, semitones):
    if semitones == 0:
        return snd.copy()
    ratio = 2.0 ** (semitones / 12.0)
    new_len = int(len(snd) / ratio)
    if new_len < 2:
        return snd[:2].copy()
    return signal.resample(snd, new_len).astype(np.float32)


# ============================================================================
# RENDER
# ============================================================================

def render_melody_faust(name, dsp, notes):
    """Render a melody variant using FAUST DSP."""
    print(f'\n  Rendering {name} ...')
    freq_a, gate_a, gain_a = make_automation(notes)
    buf = faust_render(dsp, freq_a, gate_a, gain_a, vol=0.55)
    buf = buf[:NSAMP]

    board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=6000),
        pb.Reverb(room_size=0.65, damping=0.50, wet_level=0.25,
                  dry_level=0.85, width=0.80),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6, release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, board)
    export_stem(buf, name)


def render_melody_guitar(notes):
    """Render melody using pitch-shifted guitar one-shot."""
    print(f'\n  Rendering melody_6_guitar ...')
    gtr_L = np.zeros(NSAMP, dtype=np.float32)
    gtr_R = np.zeros(NSAMP, dtype=np.float32)

    for sec, note_num, vel, dur in notes:
        bs = int(sec * SR)
        if bs >= NSAMP:
            continue
        semitones = note_num - GUITAR_REF_MIDI
        snd = pitch_shift_sample(GUITAR_SAMPLE, semitones)
        g = vel / 127.0 * 0.65
        pr = rng.uniform(0.40, 0.60)
        e = min(bs + len(snd), NSAMP)
        chunk = snd[:e - bs] * g
        gtr_L[bs:e] += chunk * (1 - pr) * 2
        gtr_R[bs:e] += chunk * pr * 2

    buf = np.stack([gtr_L, gtr_R], axis=1)
    board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=200),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.45, damping=0.55, wet_level=0.20,
                  dry_level=0.90, width=0.75),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6, release_ms=180),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, board)
    export_stem(buf, 'melody_6_guitar')


def render_pad_faust(name, dsp, notes):
    """Render a pad variant using FAUST DSP with voice separation."""
    print(f'\n  Rendering {name} ...')
    voices = separate_voices(notes)
    buf = np.zeros((NSAMP, 2), dtype=np.float32)
    for vi, voice in enumerate(voices):
        if not voice:
            continue
        freq_a, gate_a, gain_a = make_automation(voice)
        audio = faust_render(dsp, freq_a, gate_a, gain_a, vol=0.60)
        buf += audio[:NSAMP]

    board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=150),
        pb.LowpassFilter(cutoff_frequency_hz=8000),
        pb.Reverb(room_size=0.65, damping=0.50, wet_level=0.30,
                  dry_level=0.85, width=0.85),
        pb.Compressor(threshold_db=-14, ratio=2.5, attack_ms=6, release_ms=180),
        pb.Gain(gain_db=0.5),
        pb.Limiter(threshold_db=-3.0),
    ])
    buf = apply_pb(buf, board)
    export_stem(buf, name)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('Loading MIDI data ...')
    melody_notes = parse_track(FIXED_MID, 4)
    melody_notes = humanize_notes(melody_notes, timing_ms=8, vel_range=4)
    pad_notes = parse_track(FIXED_MID, 3)
    pad_notes = humanize_notes(pad_notes, timing_ms=10, vel_range=4)
    print(f'  Melody: {len(melody_notes)} notes')
    print(f'  Pad: {len(pad_notes)} notes')

    print('\n== MELODY VARIANTS ==')
    for name, dsp in MELODY_VARIANTS.items():
        render_melody_faust(name, dsp, melody_notes)

    # Guitar variant (sample-based)
    render_melody_guitar(melody_notes)

    print('\n== PAD VARIANTS ==')
    for name, dsp in PAD_VARIANTS.items():
        render_pad_faust(name, dsp, pad_notes)

    print(f'\nDone! Stems saved to:\n  {STEMS}')
