"""
sample_fetch.py — Internet Sample Sourcing Tool

Search, download, and process audio samples from YouTube, Freesound, or any URL.
Optionally isolate vocals/stems using Meta's Demucs (MPS-accelerated).

Usage:
  python sample_fetch.py youtube "tech house vocal chop" --vocals-only
  python sample_fetch.py youtube-url "https://youtube.com/watch?v=XXX" --vocals-only --trim 0:30-1:00
  python sample_fetch.py freesound "vocal chop house" --license cc0
  python sample_fetch.py url "https://example.com/sample.wav"
  python sample_fetch.py separate /path/to/audio.wav --stems vocals
  python sample_fetch.py vowels /path/to/vocal.wav --top 5 --max-dur 1.0
  python sample_fetch.py timesig /path/to/sample.wav --from-sig 3 --to-sig 4 --bpm 120
"""

import argparse
import json
import os
import re
import subprocess
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal
import librosa

# BASE_DIR defaults to cwd; set before calling functions if needed
BASE_DIR = os.getcwd()
SAMPLES  = os.path.join(BASE_DIR, 'samples')
SR       = 44100


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def search_youtube(query, max_results=5):
    """Search YouTube via yt-dlp and return top results."""
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
        m, s = divmod(int(dur), 60)
        results.append({
            'title':    data.get('title', 'Unknown'),
            'url':      data.get('url') or data.get('webpage_url') or f"https://youtube.com/watch?v={data.get('id', '')}",
            'id':       data.get('id', ''),
            'duration': dur,
            'dur_str':  f'{m}:{s:02d}',
        })
    return results


def download_youtube(url, trim=None):
    """Download audio from YouTube URL as WAV. Returns path to file."""
    out_dir = os.path.join(SAMPLES, 'youtube', 'raw')
    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, '%(title)s.%(ext)s')

    cmd = [
        'yt-dlp', url,
        '-x', '--audio-format', 'wav', '--audio-quality', '0',
        '-o', out_tmpl,
        '--no-playlist',
    ]
    if trim:
        cmd += ['--download-sections', f'*{trim}']

    print(f'  Downloading: {url}')
    if trim:
        print(f'  Trim: {trim}')
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f'  Download failed: {proc.stderr[:300]}')
        return None

    # Find the downloaded file
    import glob as g
    wavs = sorted(g.glob(os.path.join(out_dir, '*.wav')), key=os.path.getmtime, reverse=True)
    if not wavs:
        # yt-dlp may have output a different format, try to find any recent file
        all_files = sorted(g.glob(os.path.join(out_dir, '*')), key=os.path.getmtime, reverse=True)
        if all_files:
            return all_files[0]
        print('  No output file found')
        return None
    print(f'  Saved: {os.path.basename(wavs[0])}')
    return wavs[0]


# ---------------------------------------------------------------------------
# Direct URL (SoundCloud, Bandcamp, or any URL yt-dlp supports)
# ---------------------------------------------------------------------------

def download_url(url):
    """Download audio from any URL. Uses yt-dlp (supports 1000+ sites)."""
    out_dir = os.path.join(SAMPLES, 'url', 'raw')
    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, '%(title)s.%(ext)s')

    cmd = [
        'yt-dlp', url,
        '-x', '--audio-format', 'wav', '--audio-quality', '0',
        '-o', out_tmpl,
        '--no-playlist',
    ]
    print(f'  Downloading: {url}')
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        # Fallback: try wget/curl for direct file URLs
        filename = url.split('/')[-1].split('?')[0] or 'download.wav'
        out_path = os.path.join(out_dir, filename)
        proc2 = subprocess.run(['curl', '-L', '-o', out_path, url],
                               capture_output=True, text=True, timeout=120)
        if proc2.returncode == 0 and os.path.exists(out_path):
            print(f'  Saved (curl): {filename}')
            return out_path
        print(f'  Download failed: {proc.stderr[:300]}')
        return None

    import glob as g
    wavs = sorted(g.glob(os.path.join(out_dir, '*.wav')), key=os.path.getmtime, reverse=True)
    if wavs:
        print(f'  Saved: {os.path.basename(wavs[0])}')
        return wavs[0]
    all_files = sorted(g.glob(os.path.join(out_dir, '*.*')), key=os.path.getmtime, reverse=True)
    if all_files:
        return all_files[0]
    return None


# ---------------------------------------------------------------------------
# Freesound
# ---------------------------------------------------------------------------

def _get_freesound_key():
    """Load Freesound API key from env or .env file."""
    key = os.environ.get('FREESOUND_API_KEY')
    if key:
        return key
    env_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('FREESOUND_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return None


def search_freesound(query, license_filter='cc0', max_results=5):
    """Search Freesound.org API. Returns list of sound metadata."""
    api_key = _get_freesound_key()
    if not api_key:
        print('  WARNING: No FREESOUND_API_KEY found.')
        print('  Set env var or add to .env: FREESOUND_API_KEY=your_key')
        print('  Get one at: https://freesound.org/apiv2/apply/')
        return []

    license_map = {
        'cc0':         'Creative Commons 0',
        'attribution': 'Attribution',
        'any':         '',
    }
    lic = license_map.get(license_filter, '')

    import urllib.request
    import urllib.parse
    params = {
        'query': query,
        'token': api_key,
        'fields': 'id,name,duration,tags,license,previews',
        'page_size': str(max_results),
    }
    if lic:
        params['filter'] = f'license:"{lic}"'

    url = f'https://freesound.org/apiv2/search/text/?{urllib.parse.urlencode(params)}'
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f'  Freesound API error: {e}')
        return []

    results = []
    for s in data.get('results', []):
        previews = s.get('previews', {})
        preview_url = previews.get('preview-hq-mp3') or previews.get('preview-lq-mp3', '')
        results.append({
            'id':       s['id'],
            'name':     s['name'],
            'duration': round(s.get('duration', 0), 1),
            'tags':     s.get('tags', [])[:6],
            'license':  s.get('license', ''),
            'preview':  preview_url,
        })
    return results


def download_freesound(sound, output_dir=None):
    """Download a Freesound preview. Takes a result dict from search_freesound."""
    out_dir = output_dir or os.path.join(SAMPLES, 'freesound', 'raw')
    os.makedirs(out_dir, exist_ok=True)

    url = sound.get('preview', '')
    if not url:
        print('  No preview URL available')
        return None

    safe_name = re.sub(r'[^\w\-.]', '_', sound['name'])[:80]
    ext = 'mp3' if 'mp3' in url else 'ogg'
    out_path = os.path.join(out_dir, f"{sound['id']}_{safe_name}.{ext}")

    import urllib.request
    print(f'  Downloading: {sound["name"]} ({sound["duration"]}s)')
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        print(f'  Download failed: {e}')
        return None

    print(f'  Saved: {os.path.basename(out_path)}')
    return out_path


# ---------------------------------------------------------------------------
# Stem Separation (Demucs)
# ---------------------------------------------------------------------------

def separate_stems(audio_path, stem='vocals'):
    """Run demucs to isolate a stem. Returns path to isolated stem WAV."""
    source_type = 'youtube' if 'youtube' in audio_path else 'url'
    out_dir = os.path.join(SAMPLES, source_type, 'stems')
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'demucs',
        '--two-stems', stem,
        '-d', 'mps',          # Apple Silicon GPU
        '-o', out_dir,
        '--float32',
        audio_path,
    ]
    print(f'  Separating stems ({stem}) with Demucs (MPS) ...')
    print(f'  This may take a minute ...')
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        # Fallback to CPU if MPS fails
        print(f'  MPS failed, retrying on CPU ...')
        cmd[cmd.index('-d') + 1] = 'cpu'
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f'  Demucs failed: {proc.stderr[:300]}')
            return None

    # Find the output stem
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    import glob as g
    pattern = os.path.join(out_dir, 'htdemucs', track_name, f'{stem}.wav')
    matches = g.glob(pattern)
    if not matches:
        # Try without model subfolder
        pattern2 = os.path.join(out_dir, '**', track_name, f'{stem}.wav')
        matches = g.glob(pattern2, recursive=True)
    if matches:
        print(f'  Stem isolated: {matches[0]}')
        return matches[0]

    print(f'  Stem file not found at expected path: {pattern}')
    return None


# ---------------------------------------------------------------------------
# Sample Preparation
# ---------------------------------------------------------------------------

def prepare_sample(audio_path, target_sr=44100):
    """Convert to 44100Hz mono float32, normalize, trim silence. Returns path."""
    data, orig_sr = sf.read(audio_path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)

    # Resample if needed
    if orig_sr != target_sr:
        g_val = gcd(target_sr, orig_sr)
        mono = signal.resample_poly(mono, target_sr // g_val, orig_sr // g_val)
    mono = mono.astype(np.float32)

    # Trim leading/trailing silence (threshold: -40dB)
    threshold = 10 ** (-40 / 20)
    above = np.where(np.abs(mono) > threshold)[0]
    if len(above) > 0:
        # Keep 50ms padding
        pad = int(0.05 * target_sr)
        start = max(0, above[0] - pad)
        end = min(len(mono), above[-1] + pad)
        mono = mono[start:end]

    # Normalize peak to -1dB
    peak = np.abs(mono).max()
    if peak > 0:
        target_peak = 10 ** (-1 / 20)  # -1dB
        mono = mono * (target_peak / peak)

    # Save prepared version
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.dirname(audio_path)
    out_path = os.path.join(out_dir, f'{base}_prepared.wav')
    sf.write(out_path, mono, target_sr, subtype='FLOAT')
    dur = len(mono) / target_sr
    print(f'  Prepared: {os.path.basename(out_path)} ({dur:.1f}s, peak={np.abs(mono).max():.3f})')
    return out_path


# ---------------------------------------------------------------------------
# Vowel Targeting
# ---------------------------------------------------------------------------

def analyze_vowels(audio, sr, hop_length=512):
    """Compute per-frame vowel score. Vowels = high energy, low spectral flatness."""
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    # Normalize RMS to 0-1
    rms_norm = rms / (rms.max() + 1e-9)
    # Score: loud + tonal (low flatness) = vowel-like
    scores = rms_norm * (1.0 - flatness)
    # Smooth with a small window to avoid frame-level noise
    kernel = np.ones(7) / 7
    scores = np.convolve(scores, kernel, mode='same')
    return scores.astype(np.float32)


def extract_vowel_chops(audio, sr, scores, top_n=5, max_dur=1.0, hop_length=512):
    """Find top vowel-score peaks and extract chops around them."""
    from scipy.signal import find_peaks
    min_dist_frames = int(0.15 * sr / hop_length)  # 150ms min between peaks
    peaks, props = find_peaks(scores, distance=min_dist_frames, height=scores.max() * 0.25)
    # Sort by score descending
    order = np.argsort(scores[peaks])[::-1]
    peaks = peaks[order]

    max_samp = int(max_dur * sr)
    half = max_samp // 2
    fade = int(0.005 * sr)  # 5ms fade
    chops = []

    for p in peaks[:top_n * 3]:  # oversample then deduplicate
        center = p * hop_length
        start = max(0, center - half)
        end = min(len(audio), start + max_samp)
        start = max(0, end - max_samp)

        # Snap to zero crossings for clean cuts
        zc = np.where(np.diff(np.signbit(audio[max(0, start - 200):start + 200])))[0]
        if len(zc) > 0:
            start = max(0, start - 200) + zc[0]
        zc_end = np.where(np.diff(np.signbit(audio[max(0, end - 200):min(len(audio), end + 200)])))[0]
        if len(zc_end) > 0:
            end = max(0, end - 200) + zc_end[-1]
        end = min(end, len(audio))

        # Check no overlap with existing chops
        overlap = False
        for cs, ce, _ in chops:
            if start < ce and end > cs:
                overlap = True
                break
        if overlap:
            continue

        score = float(scores[p])
        chops.append((int(start), int(end), score))
        if len(chops) >= top_n:
            break

    # Sort by time position
    chops.sort(key=lambda x: x[0])

    # Apply fades
    result = []
    for start, end, score in chops:
        chunk = audio[start:end].copy()
        f = min(fade, len(chunk) // 4)
        if f > 0:
            chunk[:f] *= np.linspace(0, 1, f)
            chunk[-f:] *= np.linspace(1, 0, f)
        result.append((start, end, score, chunk))
    return result


def save_chops(chops, sr, output_dir, base_name='chop'):
    """Write individual chop WAV files. Returns list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, (start, end, score, chunk) in enumerate(chops, 1):
        t = start / sr
        fname = f'{base_name}_{i:02d}_{t:.2f}s_score{score:.2f}.wav'
        path = os.path.join(output_dir, fname)
        sf.write(path, chunk, sr, subtype='FLOAT')
        paths.append(path)
        print(f'  [{i}] t={t:.2f}s  dur={len(chunk)/sr:.3f}s  score={score:.2f}  -> {fname}')
    return paths


# ---------------------------------------------------------------------------
# Time-Signature Flipping
# ---------------------------------------------------------------------------

def detect_beats(audio, sr, bpm_hint=None):
    """Detect beat positions using librosa. Returns sample indices."""
    kwargs = {}
    if bpm_hint:
        kwargs['start_bpm'] = bpm_hint
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, **kwargs)
    beat_samples = librosa.frames_to_samples(beat_frames)
    detected_bpm = float(np.atleast_1d(tempo)[0])
    print(f'  Detected BPM: {detected_bpm:.1f}  |  {len(beat_samples)} beats')
    return beat_samples, detected_bpm


def flip_time_sig(audio, sr, beat_samples, from_sig, to_sig):
    """Regroup beats from one time signature to another via time-stretching."""
    # Add audio end as final boundary
    boundaries = np.append(beat_samples, len(audio))

    # Split into individual beat chunks
    chunks = []
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        if e > s:
            chunks.append(audio[s:e])

    if len(chunks) < max(from_sig, to_sig):
        print(f'  Not enough beats ({len(chunks)}) for {from_sig}/{to_sig} conversion')
        return audio

    # Regroup: take `from_sig` beats, redistribute into `to_sig` slots
    ratio = from_sig / to_sig  # stretch factor per beat
    fade = int(0.005 * sr)
    output_parts = []

    n_measures = len(chunks) // from_sig
    print(f'  {n_measures} measures of {from_sig}/4 -> regrouping as {to_sig}/4')

    for m in range(n_measures):
        measure_audio = np.concatenate(chunks[m * from_sig:(m + 1) * from_sig])
        # Total measure duration stays the same, but we slice into `to_sig` equal parts
        target_beat_len = len(measure_audio) / to_sig
        for b in range(to_sig):
            src_start = int(b * len(measure_audio) / to_sig)
            src_end = int((b + 1) * len(measure_audio) / to_sig)
            beat_chunk = measure_audio[src_start:src_end]
            # Resample to uniform beat length (keeps groove but reinterprets meter)
            target_len = int(target_beat_len)
            if len(beat_chunk) != target_len and len(beat_chunk) > 0:
                beat_chunk = signal.resample(beat_chunk, target_len).astype(np.float32)
            # Crossfade at joins
            if output_parts and fade > 0 and len(beat_chunk) > fade:
                overlap = min(fade, len(output_parts[-1]))
                beat_chunk[:overlap] *= np.linspace(0, 1, overlap).astype(np.float32)
                output_parts[-1][-overlap:] *= np.linspace(1, 0, overlap).astype(np.float32)
            output_parts.append(beat_chunk)

    result = np.concatenate(output_parts)
    print(f'  Output: {len(result)/sr:.2f}s ({n_measures} measures of {to_sig}/4)')
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Search, download, and process audio samples from the internet.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s youtube "tech house vocal chop" --vocals-only
  %(prog)s youtube-url "https://youtube.com/watch?v=XXX" --vocals-only --trim 0:30-1:00
  %(prog)s freesound "vocal chop" --license cc0
  %(prog)s url "https://example.com/sample.wav"
  %(prog)s separate /path/to/audio.wav --stems vocals
        """)

    parser.add_argument('source', choices=['youtube', 'youtube-url', 'freesound', 'url', 'separate',
                                           'vowels', 'timesig'],
                        help='Source to fetch from (or processing command)')
    parser.add_argument('query', help='Search query, URL, or file path')
    parser.add_argument('--vocals-only', action='store_true',
                        help='Run Demucs to isolate vocals')
    parser.add_argument('--stems', default='vocals',
                        choices=['vocals', 'drums', 'bass', 'other'],
                        help='Which stem to extract (default: vocals)')
    parser.add_argument('--trim', default=None,
                        help='Trim time range (e.g. 0:30-1:00)')
    parser.add_argument('--max-results', type=int, default=5,
                        help='Number of search results (default: 5)')
    parser.add_argument('--license', default='cc0',
                        choices=['cc0', 'attribution', 'any'],
                        help='Freesound license filter (default: cc0)')
    parser.add_argument('--list', action='store_true',
                        help='Just list search results, don\'t download')
    parser.add_argument('--pick', type=int, default=None,
                        help='Pick result by number (1-indexed) instead of interactive')
    parser.add_argument('--no-prepare', action='store_true',
                        help='Skip sample preparation step')
    # Vowel extraction options
    parser.add_argument('--top', type=int, default=5,
                        help='Number of vowel chops to extract (default: 5)')
    parser.add_argument('--max-dur', type=float, default=1.0,
                        help='Max chop duration in seconds (default: 1.0)')
    # Time-sig flipping options
    parser.add_argument('--from-sig', type=int, default=3,
                        help='Source time signature beats per bar (default: 3)')
    parser.add_argument('--to-sig', type=int, default=4,
                        help='Target time signature beats per bar (default: 4)')
    parser.add_argument('--bpm', type=float, default=None,
                        help='BPM hint for beat detection')

    args = parser.parse_args()

    print(f'\n=== sample_fetch: {args.source} ===\n')

    audio_path = None

    # --- YouTube search ---
    if args.source == 'youtube':
        results = search_youtube(args.query, args.max_results)
        if not results:
            print('No results found.')
            return

        print(f'Found {len(results)} results:\n')
        for i, r in enumerate(results, 1):
            print(f'  [{i}] {r["title"]}  ({r["dur_str"]})')
            print(f'      {r["url"]}')

        if args.list:
            return

        # Pick which to download
        pick = args.pick
        if pick is None:
            try:
                pick = int(input('\nPick a result (number): '))
            except (ValueError, EOFError):
                pick = 1
        if pick < 1 or pick > len(results):
            print(f'Invalid pick: {pick}')
            return

        chosen = results[pick - 1]
        print(f'\nDownloading: {chosen["title"]}')
        audio_path = download_youtube(chosen['url'], trim=args.trim)

    # --- YouTube direct URL ---
    elif args.source == 'youtube-url':
        audio_path = download_youtube(args.query, trim=args.trim)

    # --- Freesound ---
    elif args.source == 'freesound':
        results = search_freesound(args.query, args.license, args.max_results)
        if not results:
            print('No results found.')
            return

        print(f'Found {len(results)} results:\n')
        for i, r in enumerate(results, 1):
            tags = ', '.join(r['tags'][:4])
            print(f'  [{i}] {r["name"]}  ({r["duration"]}s)  [{tags}]')

        if args.list:
            return

        pick = args.pick
        if pick is None:
            try:
                pick = int(input('\nPick a result (number): '))
            except (ValueError, EOFError):
                pick = 1
        if pick < 1 or pick > len(results):
            print(f'Invalid pick: {pick}')
            return

        audio_path = download_freesound(results[pick - 1])

    # --- Direct URL ---
    elif args.source == 'url':
        audio_path = download_url(args.query)

    # --- Local file stem separation ---
    elif args.source == 'separate':
        audio_path = args.query
        if not os.path.exists(audio_path):
            print(f'File not found: {audio_path}')
            return
        stem_path = separate_stems(audio_path, stem=args.stems)
        if stem_path and not args.no_prepare:
            prepare_sample(stem_path)
        return

    # --- Vowel extraction ---
    elif args.source == 'vowels':
        if not os.path.exists(args.query):
            print(f'File not found: {args.query}')
            return
        audio, file_sr = sf.read(args.query, dtype='float32', always_2d=True)
        mono = audio.mean(axis=1)
        if file_sr != SR:
            g_val = gcd(SR, file_sr)
            mono = signal.resample_poly(mono, SR // g_val, file_sr // g_val).astype(np.float32)
        print(f'  Loaded: {len(mono)/SR:.2f}s')
        scores = analyze_vowels(mono, SR)
        chops = extract_vowel_chops(mono, SR, scores, top_n=args.top, max_dur=args.max_dur)
        if not chops:
            print('  No vowel chops found.')
            return
        out_dir = os.path.join(os.path.dirname(args.query), 'vowels')
        paths = save_chops(chops, SR, out_dir)
        print(f'\n  {len(paths)} vowel chops saved to {out_dir}')
        return

    # --- Time-signature flipping ---
    elif args.source == 'timesig':
        if not os.path.exists(args.query):
            print(f'File not found: {args.query}')
            return
        audio, file_sr = sf.read(args.query, dtype='float32', always_2d=True)
        mono = audio.mean(axis=1)
        if file_sr != SR:
            g_val = gcd(SR, file_sr)
            mono = signal.resample_poly(mono, SR // g_val, file_sr // g_val).astype(np.float32)
        print(f'  Loaded: {len(mono)/SR:.2f}s')
        beat_samps, detected_bpm = detect_beats(mono, SR, bpm_hint=args.bpm)
        flipped = flip_time_sig(mono, SR, beat_samps, args.from_sig, args.to_sig)
        # Normalize peak to -1dB
        peak = np.abs(flipped).max()
        if peak > 0:
            flipped = flipped * (10 ** (-1 / 20) / peak)
        base = os.path.splitext(os.path.basename(args.query))[0]
        out_path = os.path.join(os.path.dirname(args.query),
                                f'{base}_{args.from_sig}to{args.to_sig}.wav')
        sf.write(out_path, flipped, SR, subtype='FLOAT')
        print(f'\n  Saved: {out_path}')
        print(f'  Duration: {len(flipped)/SR:.2f}s')
        return

    # --- Post-download processing ---
    if audio_path is None:
        print('\nDownload failed.')
        return

    print(f'\nDownloaded: {audio_path}')

    # Stem separation
    if args.vocals_only or args.stems != 'vocals':
        stem = args.stems
        stem_path = separate_stems(audio_path, stem=stem)
        if stem_path:
            audio_path = stem_path

    # Prepare for use in render scripts
    if not args.no_prepare and audio_path:
        prepare_sample(audio_path)

    print('\nDone!')


if __name__ == '__main__':
    main()
