"""
sample_analysis.py -- Sample analysis and loop extraction utilities.

Detect tempo, key, loop period, extract bar-aligned loops, and score
candidate loop positions by loop-ability and simplicity.

Usage:
    from sample_analysis import detect_sample_tempo, detect_sample_key,
        detect_loop_period, extract_loop_auto, extract_loop_at
"""

import numpy as np
import librosa

SR = 44100


def analyze_sample_character(audio, sr=SR):
    """Spectral analysis: centroid, warmth, brightness, onset rate, etc."""
    centroid = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean())
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    low_energy = float(spec[freqs < 400].sum())
    high_energy = float(spec[freqs > 4000].sum())
    mid_energy = float(spec[(freqs >= 400) & (freqs <= 4000)].sum())
    total_energy = low_energy + mid_energy + high_energy + 1e-9
    warmth = low_energy / total_energy
    brightness = high_energy / total_energy
    mid_presence = mid_energy / total_energy
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    onset_rate = len(onsets) / (len(audio) / sr) if len(audio) > 0 else 0
    flatness = float(librosa.feature.spectral_flatness(y=audio).mean())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return {
        'centroid': centroid, 'bandwidth': bandwidth,
        'warmth': warmth, 'brightness': brightness,
        'mid_presence': mid_presence, 'onset_rate': onset_rate,
        'flatness': flatness, 'rms': rms,
    }


def detect_sample_tempo(audio, sr=SR, target_bpm=90.0, bpm_range=(70, 200)):
    """Detect tempo with half/double compensation toward target BPM."""
    tempo_default = float(np.atleast_1d(librosa.beat.tempo(y=audio, sr=sr))[0])
    tempo_hinted = float(np.atleast_1d(
        librosa.beat.tempo(y=audio, sr=sr, start_bpm=target_bpm))[0])
    candidates = []
    for t in [tempo_default, tempo_hinted]:
        for mult in [0.5, 1.0, 2.0]:
            adj = t * mult
            if bpm_range[0] < adj < bpm_range[1]:
                candidates.append((abs(adj - target_bpm), adj, t))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    return tempo_default


def detect_sample_key(audio, sr=SR):
    """Detect key via chroma energy distribution."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return key_names[int(np.argmax(chroma.mean(axis=1)))]


def detect_loop_period(audio, sr=SR):
    """Detect loop repetition period using chroma autocorrelation.
    Returns period in seconds (at least 2.5s)."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
    hop_sec = 512 / sr
    n_frames = chroma.shape[1]
    max_lag = min(n_frames // 2, int(25 / hop_sec))
    min_lag = int(2.5 / hop_sec)
    corrs = []
    for lag in range(min_lag, max_lag):
        c1 = chroma[:, :n_frames - lag]
        c2 = chroma[:, lag:]
        sim = np.sum(c1 * c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9)
        corrs.append((lag * hop_sec, sim))
    corrs.sort(key=lambda x: -x[1])
    return corrs[0][0]


def score_loop_candidates(raw_sample, loop_period, sr=SR, top_n=5):
    """Score candidate positions by loop-ability + simplicity + energy.
    Returns list of (score, pos, loop_sim, rms, simplicity)."""
    loop_samp = int(loop_period * sr)
    chroma_hop = 512
    step = int(0.5 * sr)
    candidates = []
    for pos in range(0, len(raw_sample) - loop_samp, step):
        seg = raw_sample[pos:pos + loop_samp]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms < 0.01:
            continue
        ch = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=chroma_hop)
        n_compare = min(8, ch.shape[1] // 4)
        ch_start = ch[:, :n_compare].mean(axis=1)
        ch_end = ch[:, -n_compare:].mean(axis=1)
        loop_sim = float(np.dot(ch_start, ch_end) /
                         (np.linalg.norm(ch_start) * np.linalg.norm(ch_end) + 1e-9))
        ch_mean = ch.mean(axis=1)
        ch_norm = ch_mean / (ch_mean.sum() + 1e-9)
        entropy = -float(np.sum(ch_norm * np.log2(ch_norm + 1e-9)))
        simplicity = 1.0 - (entropy / np.log2(12))
        score = 0.45 * loop_sim + 0.25 * simplicity + 0.30 * (rms / 0.2)
        candidates.append((score, pos, loop_sim, rms, simplicity))
    candidates.sort(key=lambda x: -x[0])
    return candidates[:top_n]


def extract_bar_aligned_loop(chunk, sr=SR, target_dur=None):
    """Beat-track a chunk and extract a bar-aligned loop.
    Returns (loop_audio, n_bars, bar_times)."""
    _, local_beat_frames = librosa.beat.beat_track(y=chunk, sr=sr)
    local_beat_samps = librosa.frames_to_samples(local_beat_frames)
    local_bar_samps = local_beat_samps[::4]
    local_bar_times = local_bar_samps / sr

    bar_durs = np.diff(local_bar_times)
    cumulative = np.concatenate([[0], np.cumsum(bar_durs)])
    end_idx = int(np.argmin(np.abs(cumulative - target_dur)))
    if end_idx < 2:
        end_idx = min(2, len(local_bar_samps) - 1)
    n_bars = end_idx

    loop_start = int(local_bar_samps[0])
    loop_end = int(local_bar_samps[end_idx])
    loop = chunk[loop_start:loop_end].copy()
    return loop, n_bars, local_bar_times[:end_idx + 1], loop_start


def extract_loop_at(raw_sample, loop_start_s, sr=SR, loop_end_s=None):
    """Extract a loop starting at a specific time. Uses local beat tracking."""
    chunk_start = int(loop_start_s * sr)
    chunk_end = min(chunk_start + int(30.0 * sr), len(raw_sample))
    chunk = raw_sample[chunk_start:chunk_end].copy()

    if loop_end_s is not None:
        target_dur = loop_end_s - loop_start_s
    else:
        target_dur = detect_loop_period(chunk, sr)
        print(f'  Auto-detected loop period: {target_dur:.2f}s')

    loop, n_bars, bar_times, loop_start_samp = extract_bar_aligned_loop(chunk, sr, target_dur)
    abs_start = loop_start_s + loop_start_samp / sr
    print(f'  Loop: {abs_start:.2f}s ({len(loop)/sr:.2f}s, {n_bars} bars)')
    print(f'  Bar boundaries: {bar_times}')
    return loop, n_bars


def extract_loop_auto(raw_sample, sr=SR):
    """Auto-detect best loop position and extract bar-aligned loop."""
    print('  Auto-detecting loop period ...')
    mid = len(raw_sample) // 3
    detect_chunk = raw_sample[mid:mid + min(int(60 * sr), len(raw_sample) - mid)]
    loop_period = detect_loop_period(detect_chunk, sr)
    print(f'  Detected period: {loop_period:.2f}s')

    candidates = score_loop_candidates(raw_sample, loop_period, sr)
    for i, c in enumerate(candidates):
        ts = c[1] / sr
        mins, secs = int(ts // 60), ts % 60
        print(f'  #{i+1}: {mins}:{secs:05.2f} (score={c[0]:.3f}, loop={c[2]:.3f}, '
              f'rms={c[3]:.4f}, simple={c[4]:.3f})')

    best_pos = candidates[0][1]
    local_end = min(best_pos + int(loop_period * 2.5 * sr), len(raw_sample))
    local_chunk = raw_sample[best_pos:local_end]
    loop, n_bars, bar_times, loop_start_samp = extract_bar_aligned_loop(
        local_chunk, sr, loop_period)
    abs_start = best_pos / sr + loop_start_samp / sr
    print(f'  Best section: {abs_start:.2f}s ({len(loop)/sr:.2f}s, {n_bars} bars)')
    return loop, n_bars
