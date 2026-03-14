"""
analyze_reference.py — Reference Beat Analyzer
================================================
Takes any MP3/WAV, extracts everything needed to clone it via compose/render.
Outputs a JSON profile + prints a human-readable report.

Usage:
    python analyze_reference.py <input.mp3|wav> [output_dir]

Pipeline:
    1.  Demucs stem separation (drums, bass, other, vocals)
    2.  Global properties (BPM, key, loudness, spectral, danceability)
    3.  Chord progression (essentia HPCP + ChordsDetection on bass+other)
    4.  Bass pattern (basic-pitch on bass stem)
    5.  Drum pattern + swing (essentia onset + spectral classification)
    6.  Melody extraction (basic-pitch on other stem, split by register)
    7.  Song structure + energy curve (librosa change-points + per-bar RMS)
    8.  Mix profile (stereo width, EQ balance, reverb, vocal presence)
    9.  Genre / vibe (heuristic from features)
    10. JSON output + printed report

Requirements:
    pip install essentia librosa soundfile numpy demucs basic-pitch music21
"""

import sys
import os
import json
import subprocess
import numpy as np
from collections import Counter

try:
    import essentia.standard as es
except ImportError:
    sys.exit("essentia not installed — pip install essentia")

try:
    import librosa
except ImportError:
    sys.exit("librosa not installed — pip install librosa")


# ============================================================================
# HELPERS
# ============================================================================

def midi_to_name(midi_num):
    """MIDI note number to music21 name (e.g. 60 -> 'C4')."""
    from music21 import pitch
    return pitch.Pitch(midi=int(round(midi_num))).nameWithOctave


def quantize(time_s, bpm, subdiv=16):
    """Snap a time (seconds) to the nearest grid point (beats within a bar)."""
    beat_s = 60.0 / bpm
    bar_s = beat_s * 4
    beat_in_bar = (time_s % bar_s) / beat_s
    grid = 4.0 / subdiv
    return round(round(beat_in_bar / grid) * grid, 2)


def find_cycle_len(seq, candidates=(2, 4, 8)):
    """Shortest repeating unit in a list via majority-match."""
    if len(seq) < 4:
        return max(len(seq), 1)
    for ln in candidates:
        if ln > len(seq) // 2:
            continue
        pat = seq[:ln]
        hits = total = 0
        for i in range(ln, len(seq) - ln + 1, ln):
            total += 1
            if seq[i:i + ln] == pat:
                hits += 1
        if total > 0 and hits / total >= 0.5:
            return ln
    return 4


def find_pattern(onsets_beats, bar_beats=4.0, candidates=(1, 2, 4)):
    """Autocorrelation-based repeating-pattern detector on a quantized grid."""
    if not onsets_beats:
        return 1, []
    grid = 0.25
    max_bars = max(candidates)
    grid_len = int(max_bars * bar_beats / grid)
    arr = np.zeros(grid_len)
    for b in onsets_beats:
        idx = int(b / grid) % grid_len
        arr[idx] = 1

    best, best_sc = candidates[0], -1
    for nb in candidates:
        pl = int(nb * bar_beats / grid)
        if pl == 0 or pl > grid_len:
            continue
        pat = arr[:pl]
        sc, cnt = 0, 0
        for s in range(pl, grid_len - pl + 1, pl):
            chunk = arr[s:s + pl]
            if len(chunk) == pl:
                sc += np.sum(pat * chunk) / max(np.sum(pat), 1)
                cnt += 1
        if cnt > 0 and sc / cnt > best_sc:
            best_sc = sc / cnt
            best = nb

    pat_beats = best * bar_beats
    events = sorted(set(round(e % pat_beats, 2) for e in onsets_beats))
    return best, events


# ============================================================================
# STEP 1 — DEMUCS STEM SEPARATION
# ============================================================================

def step1_separate(input_path, output_dir):
    print("\n[1/10] Separating stems with Demucs ...")
    stems_dir = os.path.join(output_dir, 'ref_stems')

    basename = os.path.splitext(os.path.basename(input_path))[0]
    demucs_out = os.path.join(stems_dir, 'htdemucs', basename)

    expected = [os.path.join(demucs_out, f'{s}.wav')
                for s in ('drums', 'bass', 'other', 'vocals')]
    if all(os.path.exists(p) for p in expected):
        print("  (cached)")
    else:
        subprocess.run(
            [sys.executable, '-m', 'demucs', '-n', 'htdemucs',
             '-o', stems_dir, input_path],
            check=True,
        )

    stems = {}
    for name in ('drums', 'bass', 'other', 'vocals'):
        p = os.path.join(demucs_out, f'{name}.wav')
        if os.path.exists(p):
            stems[name] = p
            print(f"  {name}.wav")
    return stems


# ============================================================================
# STEP 2 — GLOBAL PROPERTIES  (reuses compare_beats.py extraction pattern)
# ============================================================================

def step2_global(input_path):
    print("\n[2/10] Extracting global properties ...")
    extractor = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev'],
    )
    features, _ = extractor(input_path)

    def g(key, default=0.0):
        try:
            v = features[key]
            return float(v) if not hasattr(v, '__len__') else [float(x) for x in v]
        except Exception:
            return default

    def gs(key, default='?'):
        try:
            return str(features[key])
        except Exception:
            return default

    y, sr_lib = librosa.load(input_path, sr=None, mono=True)
    duration_s = len(y) / sr_lib

    props = {
        'bpm':                round(g('rhythm.bpm')),
        'key':                gs('tonal.key_edma.key'),
        'scale':              gs('tonal.key_edma.scale'),
        'key_strength':       round(g('tonal.key_edma.strength'), 3),
        'tuning_freq':        round(g('tonal.tuning_frequency'), 2),
        'loudness_lufs':      round(g('lowlevel.loudness_ebu128.integrated'), 1),
        'loudness_range':     round(g('lowlevel.loudness_ebu128.loudness_range'), 1),
        'dynamic_complexity': round(g('lowlevel.dynamic_complexity'), 1),
        'spectral_centroid':  round(g('lowlevel.spectral_centroid.mean'), 1),
        'spectral_rolloff':   round(g('lowlevel.spectral_rolloff.mean'), 1),
        'spectral_flux':      round(g('lowlevel.spectral_flux.mean'), 4),
        'danceability':       round(g('rhythm.danceability'), 3),
        'onset_rate':         round(g('rhythm.onset_rate'), 2),
        'beats_loudness':     round(g('rhythm.beats_loudness.mean'), 3),
        'mfcc_mean':          [round(x, 2) for x in g('lowlevel.mfcc.mean', [0]*13)],
        'gfcc_mean':          [round(x, 2) for x in g('lowlevel.gfcc.mean', [0]*13)],
        'duration_s':         round(duration_s, 1),
    }
    for k, v in props.items():
        if k not in ('mfcc_mean', 'gfcc_mean'):
            print(f"  {k}: {v}")
    return props


# ============================================================================
# STEP 3 — CHORD PROGRESSION  (essentia on bass+other stems)
# ============================================================================

def step3_chords(bass_path, other_path, bpm, sr=44100):
    print("\n[3/10] Detecting chord progression ...")
    bass  = es.MonoLoader(filename=bass_path,  sampleRate=sr)()
    other = es.MonoLoader(filename=other_path, sampleRate=sr)()
    n = min(len(bass), len(other))
    mix = bass[:n] + other[:n]

    fs, hs = 4096, 2048
    w       = es.Windowing(type='blackmanharris62')
    spec    = es.Spectrum()
    peaks   = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.0001,
                                maxPeaks=60, sampleRate=sr)
    hpcp_fn = es.HPCP(size=36, referenceFrequency=440,
                       nonLinear=True, normalized='unitMax')

    hpcp_frames = []
    for frame in es.FrameGenerator(mix, frameSize=fs, hopSize=hs):
        s = spec(w(frame))
        freqs, mags = peaks(s)
        hpcp_frames.append(hpcp_fn(freqs, mags))

    chords, strengths = es.ChordsDetection(hopSize=hs, sampleRate=sr)(
        np.array(hpcp_frames))

    # quantize to bars
    bar_s   = 60.0 / bpm * 4
    frame_s = hs / sr
    n_bars  = int(len(mix) / sr / bar_s)

    bar_chords = []
    for bar in range(n_bars):
        si = int(bar * bar_s / frame_s)
        ei = min(int((bar + 1) * bar_s / frame_s), len(chords))
        if si >= len(chords):
            break
        cs = [c for c in chords[si:ei] if c != 'N'] or ['N']
        bar_chords.append(Counter(cs).most_common(1)[0][0])

    cycle     = find_cycle_len(bar_chords)
    chord_cyc = bar_chords[:cycle]
    voicings  = _build_voicings(chord_cyc)

    print(f"  Cycle ({cycle} bars): {chord_cyc}")
    return chord_cyc, voicings, bar_chords


def _normalize_chord(name):
    """Convert essentia chord names to unambiguous music21 format.

    'Bb' is ambiguous in music21 (B + 'b' modifier?), so we use 'B-'.
    """
    flat_roots = {'Cb': 'C-', 'Db': 'D-', 'Eb': 'E-', 'Fb': 'F-',
                  'Gb': 'G-', 'Ab': 'A-', 'Bb': 'B-'}
    for old, new in flat_roots.items():
        if name.startswith(old):
            return new + name[len(old):]
    return name


def _build_voicings(chord_names, base_oct=3):
    """Root-position voicings at base_oct via music21."""
    try:
        from music21 import harmony, pitch as m21p
    except ImportError:
        return [[n] for n in chord_names]

    voicings = []
    for name in chord_names:
        if name == 'N':
            voicings.append([])
            continue
        try:
            m21name = _normalize_chord(name)
            cs      = harmony.ChordSymbol(m21name)
            root_pc = cs.root().pitchClass
            voicing = []
            for p in cs.pitches:
                np_ = m21p.Pitch(p.name)
                np_.octave = base_oct
                if np_.pitchClass < root_pc:
                    np_.octave += 1
                voicing.append(np_.nameWithOctave)
            voicings.append(voicing)
        except Exception:
            voicings.append([name])
    return voicings


# ============================================================================
# STEP 4 — BASS PATTERN  (basic-pitch on bass stem)
# ============================================================================

def _basic_pitch_predict(audio_path):
    """Run basic-pitch, preferring the ONNX model if TF is broken."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import pathlib

    onnx_path = pathlib.Path(ICASSP_2022_MODEL_PATH).with_suffix('.onnx')
    model = str(onnx_path) if onnx_path.exists() else ICASSP_2022_MODEL_PATH
    return predict(audio_path, model_or_model_path=model)


def step4_bass(bass_path, bpm, chord_cycle):
    print("\n[4/10] Extracting bass pattern ...")
    try:
        from basic_pitch.inference import predict
    except ImportError:
        print("  (skipped — basic-pitch not installed)")
        return [], []

    _, midi_data, _ = _basic_pitch_predict(bass_path)

    notes = [(n.start, n.end - n.start, n.pitch, n.velocity)
             for inst in midi_data.instruments for n in inst.notes]
    if not notes:
        print("  No bass notes detected")
        return [], []

    beat_s = 60.0 / bpm
    quantized = []
    for start, dur, pitch, vel in notes:
        b = quantize(start, bpm)
        d = max(0.25, round(dur / beat_s * 4) / 4)
        quantized.append({'beat': b, 'dur': round(d, 2), 'pitch': pitch})

    # detect rhythm pattern (pitch-free)
    pat_bars, _ = find_pattern([q['beat'] for q in quantized], candidates=(1, 2, 4))
    pat_beats = pat_bars * 4
    seen = set()
    pattern = []
    for q in quantized:
        b = round(q['beat'] % pat_beats, 2)
        if b not in seen:
            seen.add(b)
            pattern.append({'beat': b, 'dur': q['dur']})
    pattern.sort(key=lambda x: x['beat'])

    # one root per chord-cycle bar
    bass_roots = []
    n_cyc = max(len(chord_cycle), 1)
    for i in range(n_cyc):
        bar_notes = [q for q in quantized if int(q['beat'] / 4) % n_cyc == i]
        if bar_notes:
            bass_roots.append(midi_to_name(min(bar_notes, key=lambda x: x['pitch'])['pitch']))
        else:
            bass_roots.append(midi_to_name(36))

    print(f"  Pattern: {len(pattern)} events / {pat_bars}-bar cycle")
    print(f"  Bass roots: {bass_roots}")
    return pattern, bass_roots


# ============================================================================
# STEP 5 — DRUM PATTERN  (essentia onset + spectral classification)
# ============================================================================

def step5_drums(drums_path, bpm, sr=44100):
    print("\n[5/10] Transcribing drum pattern + swing ...")
    audio = es.MonoLoader(filename=drums_path, sampleRate=sr)()

    # onset detection
    od  = es.OnsetDetection(method='hfc')
    w   = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()

    feat = []
    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512):
        mag, phase = c2p(fft(w(frame)))
        feat.append(od(mag, phase))
    onsets = es.Onsets()(np.array([feat]), [1])

    bar_s = 60.0 / bpm * 4

    # Swing: measure deviation from strict grid before quantizing
    beat_s = 60.0 / bpm
    sixteenth_s = beat_s / 4
    deviations = []
    for t in onsets:
        nearest_grid = round(t / sixteenth_s) * sixteenth_s
        deviations.append(abs(t - nearest_grid))
    if deviations:
        mean_dev = float(np.mean(deviations))
        swing_pct = round(min(mean_dev / sixteenth_s * 100, 50.0), 1)
    else:
        swing_pct = 0.0

    # Pass 1: compute spectral centroid for every onset
    onset_data = []   # (beat_position, centroid_hz, time_s)
    crash_bars = []

    for t in onsets:
        center = int(t * sr)
        win = 2048
        seg = audio[max(0, center - win // 2): min(len(audio), center + win // 2)]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))

        sp    = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        power = sp ** 2
        total_pow = np.sum(power)
        if total_pow == 0:
            continue

        centroid = float(np.sum(freqs * power) / total_pow)
        beat = quantize(t, bpm)
        onset_data.append((beat, centroid, t))

    # Pass 2: adaptive thresholds from this track's centroid distribution
    grid_votes = {}
    if onset_data:
        all_c = np.array([d[1] for d in onset_data])
        kick_cut = np.percentile(all_c, 30)
        hh_cut   = np.percentile(all_c, 70)

        for beat, centroid, t in onset_data:
            if centroid <= kick_cut:
                label = 'kick'
            elif centroid >= hh_cut:
                label = 'hh'
            else:
                label = 'snare'
            grid_votes.setdefault(beat, []).append(label)

            if centroid >= hh_cut and abs(beat) < 0.1:
                crash_bars.append(int(t / bar_s))

    # Majority vote per grid slot → one label per position
    kicks_b, snares_b, hihats_b = [], [], []
    for beat, votes in grid_votes.items():
        winner = Counter(votes).most_common(1)[0][0]
        if winner == 'kick':
            kicks_b.append(beat)
        elif winner == 'hh':
            hihats_b.append(beat)
        else:
            snares_b.append(beat)

    _, kick_pat  = find_pattern(kicks_b)
    _, snare_pat = find_pattern(snares_b)
    _, hh_pat    = find_pattern(hihats_b)

    has_triplets = any(
        abs(b % 1.0 - r) < 0.08
        for b in hihats_b for r in (1 / 3, 2 / 3)
    )

    crash_every = None
    if len(crash_bars) >= 2:
        diffs = [crash_bars[i + 1] - crash_bars[i]
                 for i in range(len(crash_bars) - 1) if crash_bars[i + 1] > crash_bars[i]]
        if diffs:
            crash_every = int(round(np.median(diffs)))

    drum = {
        'kick':               kick_pat,
        'clap':               snare_pat,
        'hh':                 hh_pat,
        'has_triplets':       has_triplets,
        'crash_every_n_bars': crash_every,
    }

    print(f"  Kick: {kick_pat}")
    print(f"  Clap: {snare_pat}")
    print(f"  HH:   {len(hh_pat)} hits, triplets={has_triplets}")
    print(f"  Crash every {crash_every} bars")
    print(f"  Swing: {swing_pct}%")
    return drum, swing_pct


# ============================================================================
# STEP 6 — MELODY  (basic-pitch on "other" stem, split by register)
# ============================================================================

REGISTER_BOUNDS = {       # MIDI note boundaries (approx Hz)
    'pad':  (55, 71),     # ~200–500 Hz
    'main': (71, 90),     # ~500–1500 Hz
    'top':  (90, 128),    # ~1500+ Hz
}


def step6_melody(other_path, bpm):
    print("\n[6/10] Extracting melody layers ...")
    try:
        from basic_pitch.inference import predict
    except ImportError:
        print("  (skipped — basic-pitch not installed)")
        return {}

    _, midi_data, _ = _basic_pitch_predict(other_path)

    notes = [(n.start, n.end - n.start, n.pitch, n.velocity)
             for inst in midi_data.instruments for n in inst.notes]
    if not notes:
        print("  No melody notes detected")
        return {}

    beat_s = 60.0 / bpm
    layers = {k: [] for k in REGISTER_BOUNDS}

    for start, dur, pitch, vel in notes:
        if pitch < 55:
            continue
        b = quantize(start, bpm)
        d = max(0.25, round(dur / beat_s * 4) / 4)
        entry = {'pitch': midi_to_name(pitch), 'beat': b, 'dur': round(d, 2)}

        for name, (lo, hi) in REGISTER_BOUNDS.items():
            if lo <= pitch < hi:
                layers[name].append(entry)
                break

    result = {}
    for name, events in layers.items():
        if not events:
            continue
        pat_bars, _ = find_pattern([e['beat'] for e in events], candidates=(2, 4, 8))
        pat_beats = pat_bars * 4
        seen = set()
        pattern = []
        for e in events:
            b   = round(e['beat'] % pat_beats, 2)
            key = (e['pitch'], b)
            if key not in seen:
                seen.add(key)
                pattern.append({'pitch': e['pitch'], 'beat': b, 'dur': e['dur']})
        pattern.sort(key=lambda x: x['beat'])
        result[name] = pattern
        print(f"  {name}: {len(pattern)} notes / {pat_bars}-bar cycle")

    return result


# ============================================================================
# STEP 7 — SONG STRUCTURE  (librosa energy/spectral change-points)
# ============================================================================

def step7_structure(input_path, bpm):
    print("\n[7/10] Detecting song structure + energy curve ...")
    y, sr_lib = librosa.load(input_path, sr=22050, mono=True)

    bar_s   = 60.0 / bpm * 4
    hop_len = max(1, int(bar_s * sr_lib))

    rms      = librosa.feature.rms(y=y, hop_length=hop_len)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr_lib, hop_length=hop_len)[0]
    onset_e  = librosa.onset.onset_strength(y=y, sr=sr_lib, hop_length=hop_len)

    n_bars = len(rms)
    rms_n  = rms / rms.max() if rms.max() > 0 else rms
    cent_n = centroid / centroid.max() if centroid.max() > 0 else centroid

    # novelty = |diff| of normalized energy + spectral centroid
    rms_d  = np.abs(np.diff(rms_n,  prepend=rms_n[0]))
    cent_d = np.abs(np.diff(cent_n, prepend=cent_n[0]))
    novelty = rms_d + cent_d

    thresh = np.mean(novelty) + np.std(novelty)
    boundaries = [0]
    for i in range(1, len(novelty)):
        if novelty[i] > thresh and i - boundaries[-1] >= 4:
            boundaries.append(i)
    if boundaries[-1] != n_bars:
        boundaries.append(n_bars)

    sections = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        avg_rms = float(np.mean(rms_n[s:e]))
        if avg_rms < 0.3:
            label = 'intro' if s == 0 else 'bridge'
        elif avg_rms > 0.7:
            label = 'hook'
        else:
            label = 'verse'
        sections.append({'name': label, 'start_bar': int(s), 'end_bar': int(e)})
        print(f"  {label:<8} bars {s:>3}-{e:>3}  (energy={avg_rms:.2f})")

    energy_curve = [round(float(v), 4) for v in rms_n]
    print(f"  Energy curve: {len(energy_curve)} bars")
    return sections, energy_curve


# ============================================================================
# STEP 8 — MIX PROFILE  (stereo width, EQ balance, reverb, vocal presence)
# ============================================================================

EQ_BANDS = {'sub': (20, 100), 'low': (100, 300), 'mid': (300, 2000),
            'high': (2000, 8000), 'air': (8000, 20000)}


def step8_mix_profile(input_path, stems, bpm, sections):
    print("\n[8/10] Analyzing mix profile ...")
    sr_mix = 44100

    # --- stereo width per band ---
    y_stereo, _ = librosa.load(input_path, sr=sr_mix, mono=False)
    if y_stereo.ndim == 2 and y_stereo.shape[0] == 2:
        L, R = y_stereo[0], y_stereo[1]
        mid_sig = (L + R) / 2
        side_sig = (L - R) / 2

        n_fft = 4096
        S_mid  = np.abs(librosa.stft(mid_sig,  n_fft=n_fft))
        S_side = np.abs(librosa.stft(side_sig, n_fft=n_fft))
        freqs  = librosa.fft_frequencies(sr=sr_mix, n_fft=n_fft)

        stereo_width = {}
        for band, (lo, hi) in EQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            mid_e  = float(np.mean(S_mid[mask] ** 2))  if mask.any() else 1e-10
            side_e = float(np.mean(S_side[mask] ** 2)) if mask.any() else 0.0
            stereo_width[band] = round(side_e / max(mid_e + side_e, 1e-10), 3)
        print(f"  Stereo width: {stereo_width}")
    else:
        stereo_width = {b: 0.0 for b in EQ_BANDS}
        mid_sig = y_stereo if y_stereo.ndim == 1 else y_stereo[0]
        print("  (mono file — stereo width all 0)")

    # --- EQ balance (dB per band) ---
    y_mono = mid_sig if y_stereo.ndim == 1 else (y_stereo[0] + y_stereo[1]) / 2
    S_full = np.abs(librosa.stft(y_mono, n_fft=4096))
    freqs  = librosa.fft_frequencies(sr=sr_mix, n_fft=4096)

    eq_balance = {}
    for band, (lo, hi) in EQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = float(np.mean(S_full[mask] ** 2)) if mask.any() else 1e-10
        eq_balance[band] = round(10 * np.log10(max(energy, 1e-10)), 1)
    print(f"  EQ balance: {eq_balance}")

    # --- reverb estimate (decay from "other" stem) ---
    reverb_decay_s = 0.0
    if 'other' in stems:
        y_other, sr_o = librosa.load(stems['other'], sr=22050, mono=True)
        env = librosa.feature.rms(y=y_other, hop_length=512)[0]
        if len(env) > 1:
            autocorr = np.correlate(env - np.mean(env), env - np.mean(env), mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                # find where autocorrelation drops below 0.3 (approximate RT60-like)
                decay_idx = np.argmax(autocorr < 0.3)
                if decay_idx > 0:
                    reverb_decay_s = round(decay_idx * 512 / sr_o, 2)
    print(f"  Reverb decay: {reverb_decay_s}s")

    # --- vocal presence per section ---
    vocal_presence = []
    if 'vocals' in stems and sections:
        y_voc, sr_v = librosa.load(stems['vocals'], sr=22050, mono=True)
        y_full, _   = librosa.load(input_path, sr=22050, mono=True)
        bar_s = 60.0 / bpm * 4

        for sec in sections:
            s_samp = int(sec['start_bar'] * bar_s * sr_v)
            e_samp = min(int(sec['end_bar'] * bar_s * sr_v), len(y_voc), len(y_full))
            if s_samp >= e_samp:
                continue
            voc_rms  = float(np.sqrt(np.mean(y_voc[s_samp:e_samp] ** 2)))
            full_rms = float(np.sqrt(np.mean(y_full[s_samp:e_samp] ** 2)))
            ratio = round(voc_rms / max(full_rms, 1e-10), 3)
            vocal_presence.append({'name': sec['name'], 'start_bar': sec['start_bar'],
                                   'ratio': min(ratio, 1.0)})
        print(f"  Vocal presence: {[(v['name'], v['ratio']) for v in vocal_presence]}")
    else:
        print("  Vocal presence: (no vocal stem)")

    return {
        'stereo_width':    stereo_width,
        'eq_balance_db':   eq_balance,
        'reverb_decay_s':  reverb_decay_s,
        'vocal_presence':  vocal_presence,
    }


# ============================================================================
# STEP 9 — GENRE / VIBE  (heuristic from features)
# ============================================================================

def step9_genre(props, drum_pattern, swing_pct):
    print("\n[9/10] Classifying genre / vibe ...")
    bpm = props.get('bpm', 0)
    centroid = props.get('spectral_centroid', 1500)
    loudness = props.get('loudness_lufs', -14)
    danceability = props.get('danceability', 0)
    has_triplets = drum_pattern.get('has_triplets', False)
    kick = drum_pattern.get('kick', [])
    clap = drum_pattern.get('clap', [])
    hh   = drum_pattern.get('hh', [])

    # four-on-the-floor: kicks on 0, 1, 2, 3
    four_floor = all(b in kick for b in [0.0, 1.0, 2.0, 3.0])
    # dembow: clap/snare on offbeats
    dembow = any(abs(b - 0.75) < 0.1 or abs(b - 2.75) < 0.1 for b in clap)

    genre = 'unknown'
    if 60 <= bpm <= 100 and dembow:
        genre = 'reggaeton'
    elif 130 <= bpm <= 170 and has_triplets:
        genre = 'trap'
    elif 130 <= bpm <= 170 and not has_triplets:
        genre = 'drill'
    elif 85 <= bpm <= 115 and swing_pct > 15:
        genre = 'boom bap'
    elif 118 <= bpm <= 135 and four_floor:
        genre = 'house'
    elif 85 <= bpm <= 115 and danceability > 0.8:
        genre = 'r&b'
    elif 60 <= bpm <= 90:
        genre = 'lo-fi'
    elif 90 <= bpm <= 115:
        genre = 'hip-hop'

    # mood from spectral centroid
    if centroid < 1000:
        mood = 'dark'
    elif centroid > 2500:
        mood = 'bright'
    else:
        mood = 'neutral'

    # energy level from loudness
    if loudness > -10:
        energy_level = 'high'
    elif loudness > -16:
        energy_level = 'mid'
    else:
        energy_level = 'low'

    print(f"  Genre: {genre}")
    print(f"  Mood: {mood}  (centroid={centroid} Hz)")
    print(f"  Energy: {energy_level}  (loudness={loudness} LUFS)")
    return genre, mood, energy_level


# ============================================================================
# STEP 10 — OUTPUT  (JSON + human-readable report)
# ============================================================================

def print_report(profile):
    W = 62
    print("\n" + "=" * W)
    print("  REFERENCE ANALYSIS REPORT")
    print("=" * W)
    print(f"  Source:       {profile['source_file']}")
    print(f"  Genre:        {profile.get('genre', '?')}  |  Mood: {profile.get('mood', '?')}  |  Energy: {profile.get('energy_level', '?')}")
    print(f"  BPM:          {profile['bpm']}")
    print(f"  Key:          {profile['key']} {profile['scale']}")
    print(f"  Duration:     {profile['duration_s']:.1f}s ({profile['n_bars']} bars)")
    print(f"  Loudness:     {profile['loudness_lufs']} LUFS")
    print(f"  Centroid:     {profile['spectral_centroid']} Hz")
    print(f"  Danceability: {profile.get('danceability', '?')}")
    print(f"  Onset rate:   {profile.get('onset_rate', '?')}")
    print(f"  Swing:        {profile.get('swing_pct', 0)}%")

    print(f"\n  Chords:    {profile['chords']}")
    for i, v in enumerate(profile.get('chord_voicings', [])):
        name = profile['chords'][i] if i < len(profile['chords']) else '?'
        print(f"    Bar {i}: {name:<6} {v}")

    print(f"\n  Bass roots: {profile.get('bass_roots', [])}")
    bp = profile.get('bass_pattern', [])
    if bp:
        print(f"  Bass pattern ({len(bp)} events):")
        for e in bp:
            print(f"    beat {e['beat']:>5.2f}  dur {e['dur']}")

    dp = profile.get('drum_pattern', {})
    print(f"\n  Drums:")
    print(f"    Kick:     {dp.get('kick', [])}")
    print(f"    Clap:     {dp.get('clap', [])}")
    print(f"    HH:       {len(dp.get('hh', []))} hits, triplets={dp.get('has_triplets')}")
    print(f"    Crash:    every {dp.get('crash_every_n_bars', '?')} bars")

    ml = profile.get('melody_layers', {})
    if ml:
        print(f"\n  Melody layers:")
        for name, events in ml.items():
            print(f"    {name}: {len(events)} notes")
            for e in events[:6]:
                print(f"      {e['pitch']:<6} beat {e['beat']:>5.2f}  dur {e['dur']}")
            if len(events) > 6:
                print(f"      ... ({len(events) - 6} more)")

    secs = profile.get('sections', [])
    if secs:
        print(f"\n  Structure:")
        for s in secs:
            print(f"    {s['name']:<8} bars {s['start_bar']:>3}-{s['end_bar']:>3}")

    mp = profile.get('mix_profile', {})
    if mp:
        print(f"\n  Mix profile:")
        sw = mp.get('stereo_width', {})
        print(f"    Stereo width:  sub={sw.get('sub',0)} low={sw.get('low',0)} mid={sw.get('mid',0)} high={sw.get('high',0)} air={sw.get('air',0)}")
        eq = mp.get('eq_balance_db', {})
        print(f"    EQ balance dB: sub={eq.get('sub',0)} low={eq.get('low',0)} mid={eq.get('mid',0)} high={eq.get('high',0)} air={eq.get('air',0)}")
        print(f"    Reverb decay:  {mp.get('reverb_decay_s', 0)}s")
        vp = mp.get('vocal_presence', [])
        if vp:
            print(f"    Vocal presence:")
            for v in vp:
                print(f"      {v['name']:<8} ratio={v['ratio']}")

    print("=" * W)


# ============================================================================
# MAIN
# ============================================================================

def main(input_path, output_dir=None):
    if not os.path.isfile(input_path):
        sys.exit(f"File not found: {input_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing: {input_path}")

    # 1 — stems
    stems = step1_separate(input_path, output_dir)

    # 2 — global
    props = step2_global(input_path)
    bpm = props['bpm']

    # 3 — chords
    chord_cycle, voicings, bar_chords = [], [], []
    if 'bass' in stems and 'other' in stems:
        chord_cycle, voicings, bar_chords = step3_chords(
            stems['bass'], stems['other'], bpm)

    # 4 — bass
    bass_pattern, bass_roots = [], []
    if 'bass' in stems:
        bass_pattern, bass_roots = step4_bass(stems['bass'], bpm, chord_cycle)

    # 5 — drums + swing
    drum_pattern = {}
    swing_pct = 0.0
    if 'drums' in stems:
        drum_pattern, swing_pct = step5_drums(stems['drums'], bpm)

    # 6 — melody
    melody_layers = {}
    if 'other' in stems:
        melody_layers = step6_melody(stems['other'], bpm)

    # 7 — structure + energy curve
    sections, energy_curve = step7_structure(input_path, bpm)

    # 8 — mix profile
    mix_profile = step8_mix_profile(input_path, stems, bpm, sections)

    # 9 — genre / vibe
    genre, mood, energy_level = step9_genre(props, drum_pattern, swing_pct)

    # 10 — assemble profile
    n_bars = int(props['duration_s'] / (60.0 / bpm * 4))

    profile = {
        'source_file':      os.path.basename(input_path),
        'bpm':              bpm,
        'key':              props['key'],
        'scale':            props['scale'],
        'time_signature':   '4/4',
        'duration_s':       props['duration_s'],
        'n_bars':           n_bars,
        'loudness_lufs':    props['loudness_lufs'],
        'spectral_centroid': props['spectral_centroid'],
        'danceability':     props['danceability'],
        'onset_rate':       props['onset_rate'],
        'beats_loudness':   props['beats_loudness'],
        'swing_pct':        swing_pct,
        'chords':           chord_cycle,
        'chord_voicings':   voicings,
        'bass_roots':       bass_roots,
        'bass_pattern':     bass_pattern,
        'drum_pattern':     drum_pattern,
        'melody_layers':    melody_layers,
        'sections':         sections,
        'energy_curve':     energy_curve,
        'mix_profile':      mix_profile,
        'genre':            genre,
        'mood':             mood,
        'energy_level':     energy_level,
    }

    json_path = os.path.join(output_dir, 'ref_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    print_report(profile)
    return profile


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
