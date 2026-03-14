#!/usr/bin/env python3
"""
instruments_scan.py — Build instruments_index.json

Scans all audio files in the instruments directory and extracts:
  - Format, sample rate, channels, duration
  - Peak amplitude, RMS loudness
  - Attack time, tail silence length
  - Tonal vs percussive classification
  - Root MIDI note + note name (tonal samples only)
  - Spectral brightness (centroid)
  - Dominant frequency band
  - Sample type (drum_oneshot, melodic_oneshot, drum_loop, melodic_loop)
  - BPM (loops only)
  - Pack name (top-level folder)
  - Crest factor (punch measurement)
  - Sub energy % (energy below 80 Hz)
  - Decay ms (time from peak to -30 dB)
  - Clipping detection
  - DC offset
  - Scale / mode (major, minor, dorian, etc.) via chroma templates
  - Key + scale via Essentia (more accurate, when available)
  - Loop cleanliness (zero-crossing at boundaries)
  - Stereo width / phase correlation (stereo files)
  - Pitch stability / vibrato (tonal samples)
  - Recommended max pitch shift semitones
  - Sound-trimmed duration (minus tail silence)

Usage:
  python instruments_scan.py                  # scan everything
  python instruments_scan.py --pack flute_sound_kit
  python instruments_scan.py --rescan         # force re-analyze all files
  python instruments_scan.py --missing        # only scan files not in index yet

Output:
  instruments_index.json  (same directory as this script)
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import time
from math import gcd, log2
from pathlib import Path

import hashlib
import numpy as np
import soundfile as sf
from scipy import signal

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    HAS_ESSENTIA = False

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
INDEX_PATH = ROOT / 'instruments_index.json'
ANALYSIS_SR = 22050   # downsample for analysis (faster, sufficient)

AUDIO_EXTS = {'.wav', '.mp3', '.aif', '.aiff', '.flac', '.ogg', '.wv'}
SKIP_EXTS  = {
    '.fxp', '.txt', '.jpg', '.jpeg', '.png', '.gif', '.url', '.zpa',
    '.nfo', '.NFO', '.zip', '.rar', '.7z', '.DS_Store', '.pdf',
    '.mid', '.midi', '.sfz', '.sf2', '.rex', '.rx2', '.xp', '.preset',
    '.ini', '.xml', '.html', '.htm',
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ─── Audio loading ─────────────────────────────────────────────────────────────

def _resample(mono, orig_sr, target_sr):
    if orig_sr == target_sr:
        return mono
    g = gcd(target_sr, orig_sr)
    return signal.resample_poly(mono, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path, target_sr=ANALYSIS_SR):
    """
    Load any audio file to mono float32 at target_sr.
    Handles: standard WAV, OGG-in-WAV (FL Studio), WavPack (.wv), MP3, AIFF.
    Returns (mono_array, sample_rate, format_tag) or (None, None, error_str).
    """
    p = Path(path)
    ext = p.suffix.lower()

    # ── WavPack ──────────────────────────────────────────────────────────────
    if ext == '.wv':
        tmp = tempfile.mktemp(suffix='.wav')
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', str(path), '-ar', str(target_sr), '-ac', '1', tmp],
                capture_output=True, timeout=20)
            if r.returncode != 0:
                return None, None, 'wv_convert_failed'
            data, sr = sf.read(tmp, dtype='float32', always_2d=True)
            mono = data.mean(axis=1)
            return _resample(mono, sr, target_sr), target_sr, 'wavpack'
        except Exception as e:
            return None, None, f'wv_error:{e}'
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    # ── Standard soundfile load ───────────────────────────────────────────────
    try:
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        mono = data.mean(axis=1)
        fmt = 'wav' if ext == '.wav' else ext.lstrip('.')
        return _resample(mono, sr, target_sr), target_sr, fmt
    except Exception:
        pass

    # ── OGG-in-WAV fallback (FL Studio keyboard/Rhodes samples) ──────────────
    if ext == '.wav':
        tmp = tempfile.mktemp(suffix='.wav')
        try:
            r = subprocess.run(
                ['ffmpeg', '-f', 'ogg', '-i', str(path),
                 '-ar', str(target_sr), '-ac', '1', tmp],
                capture_output=True, timeout=20)
            if r.returncode == 0:
                data, sr = sf.read(tmp, dtype='float32', always_2d=True)
                mono = data.mean(axis=1)
                return _resample(mono, sr, target_sr), target_sr, 'ogg_in_wav'
        except Exception:
            pass
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    # ── MP3 / AIFF / other via ffmpeg ─────────────────────────────────────────
    tmp = tempfile.mktemp(suffix='.wav')
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', str(path), '-ar', str(target_sr), '-ac', '1', tmp],
            capture_output=True, timeout=20)
        if r.returncode != 0:
            return None, None, 'ffmpeg_failed'
        data, sr = sf.read(tmp, dtype='float32', always_2d=True)
        mono = data.mean(axis=1)
        return _resample(mono, sr, target_sr), target_sr, ext.lstrip('.')
    except Exception as e:
        return None, None, f'load_error:{e}'
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


# ─── Analysis functions ────────────────────────────────────────────────────────

def peak_amplitude(mono):
    return float(np.abs(mono).max())


def rms_db(mono):
    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 1e-9:
        return -120.0
    return float(20 * np.log10(rms))


def attack_ms(mono, sr):
    """Time from start to peak amplitude (envelope follower)."""
    window = max(1, int(sr * 0.005))   # 5ms window
    env = np.convolve(np.abs(mono), np.ones(window) / window, mode='same')
    peak_idx = int(np.argmax(env))
    return round(peak_idx / sr * 1000, 1)


def tail_silence_s(mono, sr, threshold_db=-60):
    """How many seconds of near-silence at the end of the file."""
    thresh = 10 ** (threshold_db / 20)
    above = np.where(np.abs(mono) > thresh)[0]
    if len(above) == 0:
        return round(len(mono) / sr, 3)
    last_active = above[-1]
    return round((len(mono) - last_active) / sr, 3)


def spectral_flatness(mono):
    """
    Ratio of geometric mean to arithmetic mean of power spectrum.
    Near 0 = tonal (melodic), near 1 = noise-like (percussive).
    """
    win = min(2048, len(mono))
    spectrum = np.abs(np.fft.rfft(mono[:win] * np.hanning(win))) ** 2
    spectrum = np.maximum(spectrum, 1e-10)
    geo = np.exp(np.mean(np.log(spectrum)))
    ari = np.mean(spectrum)
    return float(geo / ari)


def is_tonal(mono, flatness_threshold=0.05):
    """True if sample is pitched/tonal rather than percussive."""
    flat = spectral_flatness(mono)
    return flat < flatness_threshold, flat


def detect_root_midi(mono, sr, min_hz=60, max_hz=2000):
    """
    Pitch detection using librosa.yin (YIN algorithm), which correctly identifies
    the fundamental frequency even when harmonics are stronger than the fundamental.
    Falls back to FFT peak if yin fails.
    Returns (midi_note, note_name, frequency) or (None, None, None).
    """
    # Use the loudest non-silent portion (skip attack transient, skip tail silence)
    env = np.abs(mono)
    peak_idx = int(np.argmax(env))
    # Analyze a window around the sustain region (after attack, before tail)
    start = min(peak_idx, max(0, int(len(mono) * 0.1)))
    end   = min(len(mono), start + int(sr * 0.5))  # up to 0.5s of analysis audio
    segment = mono[start:end]
    if len(segment) < 512:
        segment = mono

    if HAS_LIBROSA:
        try:
            f0 = librosa.yin(segment, fmin=min_hz, fmax=max_hz, sr=sr)
            # Filter out unvoiced frames (yin returns 0 for unvoiced)
            voiced = f0[(f0 > min_hz) & (f0 < max_hz)]
            if len(voiced) > 0:
                # Use median to reject outliers
                peak_hz = float(np.median(voiced))
                midi = int(round(12 * log2(peak_hz / 440) + 69))
                midi = max(0, min(127, midi))
                note = NOTE_NAMES[midi % 12] + str((midi // 12) - 1)
                return midi, note, round(peak_hz, 1)
        except Exception:
            pass

    # FFT fallback
    win = min(65536, len(segment))
    windowed = segment[:win] * np.hanning(win)
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not mask.any():
        return None, None, None
    peak_hz = float(freqs[mask][np.argmax(fft[mask])])
    midi = int(round(12 * log2(peak_hz / 440) + 69))
    midi = max(0, min(127, midi))
    note = NOTE_NAMES[midi % 12] + str((midi // 12) - 1)
    return midi, note, round(peak_hz, 1)


def spectral_centroid_hz(mono, sr):
    """Brightness: frequency-weighted mean of the spectrum."""
    win = min(4096, len(mono))
    spectrum = np.abs(np.fft.rfft(mono[:win] * np.hanning(win)))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    total = spectrum.sum()
    if total < 1e-9:
        return 0.0
    return float((freqs * spectrum).sum() / total)


def dominant_freq_band(centroid_hz):
    """Classify into sub / bass / low-mid / mid / high."""
    if centroid_hz < 80:    return 'sub'
    if centroid_hz < 250:   return 'bass'
    if centroid_hz < 800:   return 'low-mid'
    if centroid_hz < 4000:  return 'mid'
    return 'high'


def detect_bpm(mono, sr):
    """BPM via librosa tempo estimation. Returns float or None."""
    if not HAS_LIBROSA:
        return None
    try:
        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        # Sanity check — only return if plausible
        return round(bpm, 1) if 40 <= bpm <= 220 else None
    except Exception:
        return None


def crest_factor(mono):
    """Peak-to-RMS ratio in linear scale. Higher = punchier transient."""
    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 1e-9:
        return 0.0
    return round(float(np.abs(mono).max() / rms), 2)


def sub_energy_pct(mono, sr):
    """Fraction of total energy below 80 Hz (0–100). High = real sub."""
    win = min(4096, len(mono))
    spectrum = np.abs(np.fft.rfft(mono[:win] * np.hanning(win))) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    sub = spectrum[freqs < 80].sum()
    return round(float(sub / total) * 100, 1)


def decay_ms(mono, sr, threshold_db=-30):
    """Time from peak amplitude to threshold_db below peak."""
    env = np.abs(mono)
    peak_idx = int(np.argmax(env))
    peak_val = env[peak_idx]
    if peak_val < 1e-9:
        return 0.0
    thresh = peak_val * (10 ** (threshold_db / 20))
    after_peak = env[peak_idx:]
    below = np.where(after_peak < thresh)[0]
    if len(below) == 0:
        return round(len(after_peak) / sr * 1000, 1)
    return round(below[0] / sr * 1000, 1)


def is_clipping(mono, threshold=0.999):
    """True if any sample exceeds threshold (likely clipped)."""
    return bool(np.abs(mono).max() >= threshold)


def dc_offset(mono):
    """Mean value of waveform. Non-zero causes clicks and wastes headroom."""
    return round(float(np.mean(mono)), 6)


# Mode templates: 12-element binary vectors (semitones from root)
_MODES = {
    'major':         [1,0,1,0,1,1,0,1,0,1,0,1],
    'natural_minor': [1,0,1,1,0,1,0,1,1,0,1,0],
    'dorian':        [1,0,1,1,0,1,0,1,0,1,1,0],
    'phrygian':      [1,1,0,1,0,1,0,1,1,0,1,0],
    'lydian':        [1,0,1,0,1,0,1,1,0,1,0,1],
    'mixolydian':    [1,0,1,0,1,1,0,1,0,1,1,0],
    'harmonic_minor':[1,0,1,1,0,1,0,1,1,0,0,1],
    'phrygian_dom':  [1,1,0,0,1,1,0,1,1,0,1,0],  # Spanish/flamenco ♭2
}
_MODE_ARRAYS = {k: np.array(v, dtype=float) for k, v in _MODES.items()}


def detect_scale(mono, sr):
    """
    Chroma-based key + mode detection.
    Returns (root_note, mode, confidence_0_to_1) or (None, None, 0).
    Uses Essentia's key detector when available (more accurate),
    falls back to chroma template matching.
    """
    # ── Essentia key detector (most accurate) ────────────────────────────────
    if HAS_ESSENTIA:
        try:
            mono_es = mono.astype(np.float32)
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(mono_es)
            # Map Essentia scale names to our mode names
            scale_map = {'major': 'major', 'minor': 'natural_minor'}
            mode = scale_map.get(scale, scale)
            return key, mode, round(float(strength), 3)
        except Exception:
            pass

    # ── Chroma template matching fallback ────────────────────────────────────
    if not HAS_LIBROSA:
        return None, None, 0.0
    try:
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        profile = chroma.mean(axis=1)  # 12-element vector
        profile = profile / (profile.sum() + 1e-9)

        best_score, best_root, best_mode = -1, None, None
        for mode_name, template in _MODE_ARRAYS.items():
            for root in range(12):
                rotated = np.roll(template, root)
                rotated = rotated / (rotated.sum() + 1e-9)
                score = float(np.dot(profile, rotated))
                if score > best_score:
                    best_score, best_root, best_mode = score, root, mode_name

        root_name = NOTE_NAMES[best_root] if best_root is not None else None
        return root_name, best_mode, round(best_score, 3)
    except Exception:
        return None, None, 0.0


def loop_cleanliness(mono, sr, window_ms=5):
    """
    How cleanly the loop start and end connect.
    Returns a score 0–1: 1 = perfectly seamless, 0 = harsh click.
    Based on: zero-crossing proximity + amplitude match at boundaries.
    """
    n = max(1, int(window_ms / 1000 * sr))
    start_rms = float(np.sqrt(np.mean(mono[:n] ** 2)))
    end_rms   = float(np.sqrt(np.mean(mono[-n:] ** 2)))
    # Level match: how similar are start and end RMS
    if max(start_rms, end_rms) < 1e-9:
        return 1.0
    level_match = 1.0 - abs(start_rms - end_rms) / max(start_rms, end_rms)
    # Zero-crossing proximity: first zero crossing from start
    signs = np.sign(mono[:n * 4])
    crossings = np.where(np.diff(signs) != 0)[0]
    zc_score = 1.0 if len(crossings) == 0 else max(0, 1 - crossings[0] / (n * 4))
    return round(float((level_match + zc_score) / 2), 3)


def stereo_width(path, sr):
    """
    For stereo files: L/R correlation (-1=out-of-phase, 0=wide, 1=mono).
    Returns None for mono files.
    """
    try:
        data, orig_sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception:
        return None
    if data.shape[1] < 2:
        return None
    L, R = data[:, 0], data[:, 1]
    n = min(len(L), int(orig_sr * 2))  # up to 2s
    L, R = L[:n], R[:n]
    denom = (np.sqrt(np.mean(L**2)) * np.sqrt(np.mean(R**2)))
    if denom < 1e-9:
        return None
    corr = float(np.mean(L * R) / denom)
    return round(corr, 3)


def pitch_stability(mono, sr):
    """
    For tonal samples: how stable is the pitch over time?
    Returns std deviation of detected F0 in cents (0 = rock-solid, >50 = heavy vibrato).
    Also returns vibrato_hz (modulation rate) if vibrato is present.
    """
    if not HAS_LIBROSA:
        return None, None
    try:
        f0 = librosa.yin(mono, fmin=60, fmax=2000, sr=sr)
        voiced = f0[(f0 > 60) & (f0 < 2000)]
        if len(voiced) < 4:
            return None, None
        # Convert to cents relative to median
        median_f0 = np.median(voiced)
        cents = 1200 * np.log2(voiced / median_f0)
        stability_cents = round(float(np.std(cents)), 1)
        # Estimate vibrato rate via autocorrelation of cent curve
        vibrato_hz = None
        if len(cents) > 8:
            ac = np.correlate(cents - cents.mean(), cents - cents.mean(), mode='full')
            ac = ac[len(ac)//2:]
            ac /= ac[0] + 1e-9
            # Frame rate for yin at sr=22050
            hop = 512
            frame_rate = sr / hop
            peaks, _ = signal.find_peaks(ac[1:], height=0.2)
            if len(peaks) > 0:
                vibrato_hz = round(float(frame_rate / (peaks[0] + 1)), 2)
        return stability_cents, vibrato_hz
    except Exception:
        return None, None


def max_pitch_shift(mono, sr, duration_s):
    """
    Estimate how many semitones up/down this sample can be shifted cleanly.
    Heuristic: longer + lower-pitched = more headroom; short transients = less.
    """
    # Longer samples have more spectral information = better shift quality
    dur_score  = min(duration_s / 2.0, 1.0)   # saturates at 2s
    # Spectral complexity: simple tones shift better
    win = min(4096, len(mono))
    spec = np.abs(np.fft.rfft(mono[:win] * np.hanning(win)))
    spec_norm = spec / (spec.max() + 1e-9)
    complexity = float(np.sum(spec_norm > 0.1)) / len(spec_norm)  # fraction of bins above 10%
    complexity_score = 1.0 - min(complexity * 4, 1.0)
    combined = (dur_score * 0.6 + complexity_score * 0.4)
    max_st = int(round(combined * 12))   # 0–12 semitones
    return max(1, max_st)


def classify_type(duration_s, tonal, centroid_hz):
    """Guess sample type from duration and tonality."""
    loop = duration_s > 1.8
    if tonal:
        return 'melodic_loop' if loop else 'melodic_oneshot'
    else:
        return 'drum_loop' if loop else 'drum_oneshot'


def content_hash(mono, n_samples=8192):
    """MD5 of first N audio samples (quantized to 16-bit).
    Identical audio from different packs will produce the same hash."""
    chunk = mono[:min(n_samples, len(mono))]
    quantized = (chunk * 32767).astype(np.int16)
    return hashlib.md5(quantized.tobytes()).hexdigest()


def pack_name(rel_path):
    """Top-level folder name = pack."""
    parts = Path(rel_path).parts
    return parts[0] if parts else ''


# ─── Per-file analysis ─────────────────────────────────────────────────────────

def analyze(path, rel):
    mono, sr, fmt = load_audio(path)
    if mono is None:
        return {'error': fmt, 'path': rel}

    dur  = round(len(mono) / sr, 3)
    peak = round(peak_amplitude(mono), 4)
    rms  = round(rms_db(mono), 1)
    atk  = attack_ms(mono, sr)
    tail = tail_silence_s(mono, sr)
    cent = round(spectral_centroid_hz(mono, sr), 0)
    band = dominant_freq_band(cent)
    tonal, flatness = is_tonal(mono)
    flatness = round(flatness, 4)

    # New fields
    crest   = crest_factor(mono)
    sub_pct = sub_energy_pct(mono, sr)
    dec_ms  = decay_ms(mono, sr)
    clipped = is_clipping(mono)
    dc      = dc_offset(mono)
    trimmed_dur = round(dur - tail, 3)
    max_shift   = max_pitch_shift(mono, sr, trimmed_dur)
    c_hash      = content_hash(mono)

    root_midi = root_note = root_hz = None
    if tonal:
        root_midi, root_note, root_hz = detect_root_midi(mono, sr)

    sample_type = classify_type(dur, tonal, cent)

    bpm = None
    if 'loop' in sample_type:
        bpm = detect_bpm(mono, sr)

    # Scale / key (all tonal samples and loops)
    key_root = key_mode = key_confidence = None
    if tonal or 'loop' in sample_type:
        key_root, key_mode, key_confidence = detect_scale(mono, sr)

    # Loop cleanliness (loops only)
    loop_clean = None
    if 'loop' in sample_type:
        loop_clean = loop_cleanliness(mono, sr)

    # Pitch stability / vibrato (tonal one-shots and short melodic loops)
    stab_cents = vibrato_hz = None
    if tonal and dur < 8.0:
        stab_cents, vibrato_hz = pitch_stability(mono, sr)

    # Raw file info (native SR + channels before downsampling)
    try:
        info = sf.info(str(path))
        native_sr = info.samplerate
        channels  = info.channels
    except Exception:
        native_sr = sr
        channels  = 1

    # Stereo width (only if native file is stereo)
    width = stereo_width(path, sr) if channels >= 2 else None

    entry = {
        'pack':             pack_name(rel),
        'format':           fmt,
        'channels':         channels,
        'sample_rate':      native_sr,
        'duration_s':       dur,
        'trimmed_duration_s': trimmed_dur,
        'peak':             peak,
        'rms_db':           rms,
        'crest_factor':     crest,
        'sub_energy_pct':   sub_pct,
        'attack_ms':        atk,
        'decay_ms':         dec_ms,
        'tail_silence_s':   tail,
        'clipping':         clipped,
        'dc_offset':        dc,
        'is_tonal':         tonal,
        'flatness':         flatness,
        'brightness_hz':    int(cent),
        'freq_band':        band,
        'type':             sample_type,
        'max_pitch_shift_st': max_shift,
        'content_hash':     c_hash,
    }
    if root_midi is not None:
        entry['root_midi'] = root_midi
        entry['root_note'] = root_note
        entry['root_hz']   = root_hz
    if bpm is not None:
        entry['bpm'] = bpm
    if key_root is not None:
        entry['key']            = key_root
        entry['scale']          = key_mode
        entry['key_confidence'] = key_confidence
    if loop_clean is not None:
        entry['loop_clean'] = loop_clean
    if stab_cents is not None:
        entry['pitch_stability_cents'] = stab_cents
    if vibrato_hz is not None:
        entry['vibrato_hz'] = vibrato_hz
    if width is not None:
        entry['stereo_width'] = width

    return entry


# ─── File discovery ────────────────────────────────────────────────────────────

def discover(root, pack_filter=None):
    files = []
    for dirpath, _, filenames in os.walk(root):
        # Skip the converted cache dir (already standard WAV)
        rel_dir = Path(dirpath).relative_to(root)
        top = rel_dir.parts[0] if rel_dir.parts else ''
        if pack_filter and top != pack_filter:
            continue
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_EXTS or ext not in AUDIO_EXTS:
                continue
            full = Path(dirpath) / fname
            rel  = str(full.relative_to(root))
            files.append((str(full), rel))
    return sorted(files, key=lambda x: x[1])


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build instruments_index.json')
    parser.add_argument('--pack',    help='Only scan a specific top-level pack folder')
    parser.add_argument('--rescan',  action='store_true', help='Re-analyze all files')
    parser.add_argument('--missing', action='store_true', help='Only scan files not yet indexed')
    args = parser.parse_args()

    # Load existing index
    index = {}
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            index = json.load(f)
        print(f'Loaded existing index: {len(index)} entries')

    files = discover(ROOT, pack_filter=args.pack)
    print(f'Found {len(files)} audio files to process\n')

    errors = []
    updated = 0
    skipped = 0
    t0 = time.time()

    for i, (full_path, rel) in enumerate(files):
        # Decide whether to skip
        if not args.rescan and rel in index:
            if args.missing or 'error' not in index[rel]:
                skipped += 1
                continue

        # Progress line
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 0.1)
        remaining = (len(files) - i - 1) / rate
        print(f'[{i+1:>5}/{len(files)}] {rel[:80]:<80}', end=' ', flush=True)

        entry = analyze(full_path, rel)

        if 'error' in entry:
            errors.append(rel)
            print(f'ERROR: {entry["error"]}')
        else:
            print(f'{entry["type"]:<18} {entry["duration_s"]:.2f}s  '
                  f'{entry.get("root_note","—"):>4}  {entry["rms_db"]:>6.1f}dBFS')
            updated += 1

        index[rel] = entry

        # Save every 50 files so progress isn't lost on interrupt
        if (i + 1) % 50 == 0:
            _save(index)

    _save(index)

    elapsed = time.time() - t0
    print(f'\n{"─"*60}')
    print(f'Done in {elapsed:.0f}s')
    print(f'  Analyzed : {updated}')
    print(f'  Skipped  : {skipped} (already indexed)')
    print(f'  Errors   : {len(errors)}')
    if errors:
        print('\nFiles that could not be read:')
        for e in errors[:20]:
            print(f'  {e}')
    print(f'\nIndex saved to: {INDEX_PATH}')
    print(f'Total entries: {len(index)}')


def _save(index):
    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
