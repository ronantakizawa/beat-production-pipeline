"""
sample_scout.py -- Sample Quality Scoring Pipeline

Search YouTube for trap loop samples, download candidates, stem-separate,
score audio quality, and rank. Picks the best samples for beat production.

Usage:
  python sample_scout.py search "dark guitar trap loop"                  # list results
  python sample_scout.py scout "dark guitar trap loop" --top 5           # full pipeline
  python sample_scout.py rank samples/stems/htdemucs/                    # rank existing
  python sample_scout.py score path/to/no_drums.wav                      # score one file
"""

import argparse
import json
import os
import subprocess
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal
import librosa

SR = 44100
# BASE_DIR defaults to cwd; override with --output-dir
BASE_DIR = os.getcwd()
SAMPLES = os.path.join(BASE_DIR, 'samples')


# ============================================================================
# Scoring metrics (each returns 0.0 - 1.0, higher = better)
# ============================================================================

def score_tonal_clarity(audio, sr):
    """Low spectral flatness = tonal/melodic content. Range 0-1."""
    flat = librosa.feature.spectral_flatness(y=audio, hop_length=512)[0]
    # Flatness near 0 = tonal, near 1 = noise
    clarity = 1.0 - float(np.median(flat))
    return max(0.0, min(1.0, clarity))


def score_loopability(audio, sr):
    """Best chroma autocorrelation peak. High = repeating pattern."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
    hop_sec = 512 / sr
    n_frames = chroma.shape[1]
    # Search for repetition between 2s and 15s
    min_lag = int(2.0 / hop_sec)
    max_lag = min(n_frames // 2, int(15.0 / hop_sec))
    if min_lag >= max_lag:
        return 0.0
    best_sim = 0.0
    for lag in range(min_lag, max_lag, max(1, (max_lag - min_lag) // 200)):
        c1 = chroma[:, :n_frames - lag]
        c2 = chroma[:, lag:]
        sim = float(np.sum(c1 * c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-9))
        if sim > best_sim:
            best_sim = sim
    return max(0.0, min(1.0, best_sim))


def score_tempo_consistency(audio, sr):
    """Low variance in onset intervals = steady tempo. Range 0-1."""
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
    if len(onsets) < 4:
        return 0.0
    intervals = np.diff(onsets)
    # Remove outliers (> 2x median)
    med = np.median(intervals)
    intervals = intervals[(intervals > med * 0.3) & (intervals < med * 3.0)]
    if len(intervals) < 3:
        return 0.0
    cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    # CV of 0 = perfect, CV of 1+ = erratic
    return max(0.0, min(1.0, 1.0 - cv))


def score_frequency_balance(audio, sr):
    """Penalize samples that are too bass-heavy or too bright.
    Ideal for trap: energy concentrated in mids (200-4000 Hz)."""
    S = np.abs(librosa.stft(audio, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr)

    low_mask = freqs < 200
    mid_mask = (freqs >= 200) & (freqs < 4000)
    high_mask = freqs >= 4000

    low_e = float(S[low_mask].sum()) if low_mask.any() else 0
    mid_e = float(S[mid_mask].sum()) if mid_mask.any() else 0
    high_e = float(S[high_mask].sum()) if high_mask.any() else 0
    total = low_e + mid_e + high_e + 1e-9

    mid_ratio = mid_e / total
    low_ratio = low_e / total
    # Ideal: mid_ratio > 0.5, low_ratio < 0.3
    # Penalize bass-heavy samples (they'll clash with 808s)
    balance = mid_ratio - 0.3 * max(0, low_ratio - 0.2)
    return max(0.0, min(1.0, balance))


def score_dynamic_range(audio, sr):
    """Moderate dynamic range is ideal. Too flat or too spiky = bad."""
    # Compute RMS in 0.5s windows
    win = int(0.5 * sr)
    n_wins = len(audio) // win
    if n_wins < 2:
        return 0.5
    rms_vals = []
    for i in range(n_wins):
        chunk = audio[i * win:(i + 1) * win]
        rms_vals.append(float(np.sqrt(np.mean(chunk ** 2))))
    rms_arr = np.array(rms_vals)
    rms_arr = rms_arr[rms_arr > 0.005]  # skip near-silence
    if len(rms_arr) < 2:
        return 0.3
    cv = float(np.std(rms_arr) / (np.mean(rms_arr) + 1e-9))
    # CV 0.1-0.4 = good dynamic range. Too low = flat. Too high = inconsistent.
    if cv < 0.1:
        return 0.5  # too compressed
    elif cv > 0.6:
        return max(0.0, 1.0 - cv)  # too dynamic
    else:
        return 1.0  # sweet spot


def score_duration(audio, sr):
    """Prefer samples 15s-180s. Too short = not enough material. Too long = pack with filler."""
    dur = len(audio) / sr
    if dur < 5:
        return 0.1
    elif dur < 15:
        return 0.4
    elif dur <= 180:
        return 1.0
    elif dur <= 300:
        return 0.7
    else:
        return 0.4


def score_sample(audio, sr, verbose=False):
    """Compute all metrics and weighted total. Returns dict."""
    metrics = {}
    metrics['tonal_clarity'] = score_tonal_clarity(audio, sr)
    metrics['loopability'] = score_loopability(audio, sr)
    metrics['tempo_consistency'] = score_tempo_consistency(audio, sr)
    metrics['freq_balance'] = score_frequency_balance(audio, sr)
    metrics['dynamic_range'] = score_dynamic_range(audio, sr)
    metrics['duration'] = score_duration(audio, sr)

    # Weighted total
    weights = {
        'tonal_clarity':     0.20,
        'loopability':       0.30,
        'tempo_consistency': 0.15,
        'freq_balance':      0.15,
        'dynamic_range':     0.10,
        'duration':          0.10,
    }
    total = sum(metrics[k] * weights[k] for k in weights)
    metrics['total'] = total

    # Extra info
    metrics['duration_s'] = len(audio) / sr
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    metrics['centroid_hz'] = int(np.median(centroid))

    if verbose:
        print(f'    Tonal clarity:     {metrics["tonal_clarity"]:.3f}')
        print(f'    Loop-ability:      {metrics["loopability"]:.3f}')
        print(f'    Tempo consistency: {metrics["tempo_consistency"]:.3f}')
        print(f'    Freq balance:      {metrics["freq_balance"]:.3f}')
        print(f'    Dynamic range:     {metrics["dynamic_range"]:.3f}')
        print(f'    Duration:          {metrics["duration"]:.3f}  ({metrics["duration_s"]:.0f}s)')
        print(f'    Centroid:          {metrics["centroid_hz"]} Hz')
        print(f'    -- TOTAL:          {metrics["total"]:.3f}')

    return metrics


# ============================================================================
# YouTube search (reuses yt-dlp)
# ============================================================================

def search_youtube(query, max_results=10):
    cmd = [
        'yt-dlp', f'ytsearch{max_results}:{query}',
        '--dump-json', '--flat-playlist', '--no-download',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        print(f'  yt-dlp search error: {proc.stderr[:200]}')
        return []
    results = []
    for line in proc.stdout.strip().split('\n'):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        dur = data.get('duration') or 0
        views = data.get('view_count') or 0
        m, s = divmod(int(dur), 60)
        results.append({
            'title': data.get('title', 'Unknown'),
            'url': data.get('url') or data.get('webpage_url') or f"https://youtube.com/watch?v={data.get('id', '')}",
            'id': data.get('id', ''),
            'duration': dur,
            'dur_str': f'{m}:{s:02d}',
            'views': views,
        })
    return results


def download_youtube(url, out_dir):
    """Download audio from YouTube URL as WAV. Returns path."""
    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, '%(title)s.%(ext)s')
    cmd = [
        'yt-dlp', url,
        '-x', '--audio-format', 'wav', '--audio-quality', '0',
        '-o', out_tmpl, '--no-playlist',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f'    Download failed: {proc.stderr[:200]}')
        return None
    import glob as g
    wavs = sorted(g.glob(os.path.join(out_dir, '*.wav')), key=os.path.getmtime, reverse=True)
    if wavs:
        return wavs[0]
    all_f = sorted(g.glob(os.path.join(out_dir, '*')), key=os.path.getmtime, reverse=True)
    return all_f[0] if all_f else None


def separate_drums(audio_path, out_dir):
    """Run demucs --two-stems drums. Returns path to no_drums.wav."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'demucs',
        '--two-stems', 'drums',
        '-d', 'mps',
        '-o', out_dir,
        '--float32',
        audio_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        # Fallback to CPU
        cmd[cmd.index('-d') + 1] = 'cpu'
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f'    Demucs failed: {proc.stderr[:200]}')
            return None
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    import glob as g
    pattern = os.path.join(out_dir, 'htdemucs', track_name, 'no_drums.wav')
    matches = g.glob(pattern)
    if not matches:
        matches = g.glob(os.path.join(out_dir, '**', track_name, 'no_drums.wav'), recursive=True)
    return matches[0] if matches else None


def load_mono(path):
    """Load audio file as mono float32 at SR."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g_val = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g_val, orig_sr // g_val)
    return mono.astype(np.float32)


# ============================================================================
# Commands
# ============================================================================

def cmd_search(query, max_results=10):
    """Search YouTube and list results."""
    print(f'Searching YouTube: "{query}" ...\n')
    results = search_youtube(query, max_results)
    if not results:
        print('No results found.')
        return results
    for i, r in enumerate(results, 1):
        views_k = r['views'] / 1000 if r['views'] else 0
        print(f'  [{i:2d}] {r["title"]}')
        print(f'       {r["dur_str"]}  |  {views_k:.0f}K views  |  {r["url"]}')
    return results


def cmd_score(path, verbose=True):
    """Score a single audio file."""
    if not os.path.exists(path):
        print(f'File not found: {path}')
        return None
    print(f'Scoring: {os.path.basename(path)}')
    audio = load_mono(path)
    return score_sample(audio, SR, verbose=verbose)


def cmd_rank(directory):
    """Score and rank all no_drums.wav (or _prepared.wav) files in a directory tree."""
    import glob as g
    # Find candidate files
    patterns = [
        os.path.join(directory, '**', 'no_drums.wav'),
        os.path.join(directory, '**', '*_prepared.wav'),
    ]
    files = []
    for pat in patterns:
        files.extend(g.glob(pat, recursive=True))
    # Deduplicate and prefer no_drums over prepared if both exist
    seen_dirs = {}
    for f in files:
        d = os.path.dirname(f)
        basename = os.path.basename(f)
        if d not in seen_dirs or basename == 'no_drums.wav':
            seen_dirs[d] = f
    files = list(seen_dirs.values())

    if not files:
        print(f'No audio files found in {directory}')
        return []

    print(f'Found {len(files)} samples to score:\n')
    results = []
    for f in sorted(files):
        # Get a short name from the parent directory
        parent = os.path.basename(os.path.dirname(f))
        short_name = parent[:60] if len(parent) > 60 else parent
        print(f'  [{len(results)+1}] {short_name}')
        try:
            audio = load_mono(f)
            metrics = score_sample(audio, SR, verbose=True)
            metrics['path'] = f
            metrics['name'] = short_name
            results.append(metrics)
        except Exception as e:
            print(f'    ERROR: {e}')
        print()

    if not results:
        return []

    # Rank by total score
    results.sort(key=lambda x: -x['total'])
    print('=' * 70)
    print('RANKED RESULTS')
    print('=' * 70)
    for i, r in enumerate(results, 1):
        print(f'  #{i}  {r["total"]:.3f}  {r["name"]}')
        print(f'       tonal={r["tonal_clarity"]:.2f}  loop={r["loopability"]:.2f}  '
              f'tempo={r["tempo_consistency"]:.2f}  freq={r["freq_balance"]:.2f}  '
              f'dyn={r["dynamic_range"]:.2f}  dur={r["duration_s"]:.0f}s')
    return results


def cmd_scout(query, top=5):
    """Full pipeline: search -> download top N -> separate -> score -> rank."""
    print(f'=== SAMPLE SCOUT: "{query}" ===\n')

    # 1. Search
    results = search_youtube(query, max_results=top * 2)
    if not results:
        print('No results found.')
        return

    # Filter: prefer 30s-300s duration (skip very short or very long)
    candidates = [r for r in results if 30 <= r['duration'] <= 300]
    if not candidates:
        candidates = results
    candidates = candidates[:top]

    print(f'Downloading & analyzing top {len(candidates)} results:\n')

    scored = []
    scout_dir = os.path.join(SAMPLES, 'scout')

    for i, r in enumerate(candidates, 1):
        print(f'--- [{i}/{len(candidates)}] {r["title"]} ({r["dur_str"]}) ---')

        # Download
        raw_dir = os.path.join(scout_dir, 'raw')
        wav_path = download_youtube(r['url'], raw_dir)
        if not wav_path:
            print('    Skipping (download failed)')
            continue

        # Stem separate
        stems_dir = os.path.join(scout_dir, 'stems')
        print(f'    Separating drums ...')
        no_drums = separate_drums(wav_path, stems_dir)
        if not no_drums:
            # Fall back to scoring the raw file
            print('    Separation failed, scoring raw audio')
            no_drums = wav_path

        # Score
        try:
            audio = load_mono(no_drums)
            metrics = score_sample(audio, SR, verbose=True)
            metrics['path'] = no_drums
            metrics['name'] = r['title']
            metrics['url'] = r['url']
            metrics['views'] = r['views']
            scored.append(metrics)
        except Exception as e:
            print(f'    Scoring failed: {e}')
        print()

    if not scored:
        print('No samples scored successfully.')
        return

    # Rank
    scored.sort(key=lambda x: -x['total'])
    print('=' * 70)
    print('RANKED RESULTS')
    print('=' * 70)
    for i, s in enumerate(scored, 1):
        winner = ' <-- BEST' if i == 1 else ''
        print(f'  #{i}  {s["total"]:.3f}  {s["name"]}{winner}')
        print(f'       tonal={s["tonal_clarity"]:.2f}  loop={s["loopability"]:.2f}  '
              f'tempo={s["tempo_consistency"]:.2f}  freq={s["freq_balance"]:.2f}  '
              f'dyn={s["dynamic_range"]:.2f}  dur={s["duration_s"]:.0f}s')
        print(f'       {s["url"]}')
        print(f'       {s["path"]}')

    print(f'\nBest sample: {scored[0]["name"]}')
    print(f'  Path: {scored[0]["path"]}')
    return scored


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sample quality scoring pipeline for trap beat production.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "dark guitar trap loop"
  %(prog)s scout "dark guitar trap loop" --top 5
  %(prog)s rank samples/stems/htdemucs/
  %(prog)s score path/to/no_drums.wav
        """)

    parser.add_argument('command', choices=['search', 'scout', 'rank', 'score'],
                        help='Command to run')
    parser.add_argument('query', help='Search query, directory path, or file path')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of candidates to evaluate (default: 5)')
    parser.add_argument('--max-results', type=int, default=10,
                        help='Max YouTube search results (default: 10)')

    args = parser.parse_args()

    if args.command == 'search':
        cmd_search(args.query, args.max_results)
    elif args.command == 'score':
        cmd_score(args.query, verbose=True)
    elif args.command == 'rank':
        cmd_rank(args.query)
    elif args.command == 'scout':
        cmd_scout(args.query, top=args.top)


if __name__ == '__main__':
    main()
