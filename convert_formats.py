"""
Convert unreadable audio files in-place to standard PCM WAV.

Handles two FL Studio formats:
  1. .wv (WavPack lossless) → decoded via wvunpack
  2. .wav with OGG Vorbis inside WAV container → extract OGG, decode, re-save as PCM WAV

Files are converted in-place (original replaced with PCM WAV).
A backup is NOT made — run on a copy if you want to preserve originals.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent

# Count stats
stats = {'wv_ok': 0, 'wv_fail': 0, 'ogg_ok': 0, 'ogg_fail': 0, 'skipped': 0}


def is_readable(path):
    """Check if soundfile can read the file."""
    try:
        import soundfile as sf
        sf.read(str(path), dtype='float32', frames=1)
        return True
    except Exception:
        return False


def convert_wavpack(path):
    """Convert .wv → .wav using wvunpack."""
    wav_path = str(path).rsplit('.wv', 1)[0] + '.wav'
    result = subprocess.run(
        ['wvunpack', '-y', str(path), '-o', wav_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and os.path.exists(wav_path):
        os.remove(path)  # remove .wv after successful conversion
        stats['wv_ok'] += 1
        return wav_path
    else:
        stats['wv_fail'] += 1
        return None


def convert_ogg_in_wav(path):
    """Extract OGG Vorbis from WAV container and re-save as PCM WAV."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()

        ogg_start = raw.find(b'OggS')
        if ogg_start < 0:
            stats['ogg_fail'] += 1
            return None

        # Extract OGG data to temp file
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
            tmp.write(raw[ogg_start:])
            tmp_ogg = tmp.name

        # Decode OGG and re-save as PCM WAV via ffmpeg
        tmp_wav = tmp_ogg.replace('.ogg', '.wav')
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_ogg, '-acodec', 'pcm_s16le', tmp_wav],
            capture_output=True, text=True,
        )

        os.remove(tmp_ogg)

        if result.returncode == 0 and os.path.exists(tmp_wav):
            # Replace original with converted file
            os.replace(tmp_wav, str(path))
            stats['ogg_ok'] += 1
            return str(path)
        else:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            stats['ogg_fail'] += 1
            return None

    except Exception as e:
        stats['ogg_fail'] += 1
        return None


def main():
    # Find all files to convert
    wv_files = list(ROOT.rglob('*.wv'))
    print(f'Found {len(wv_files)} .wv files')

    # Find bad .wav files (OGG-in-WAV)
    bad_wavs = []
    all_wavs = list(ROOT.rglob('*.wav'))
    print(f'Checking {len(all_wavs)} .wav files for OGG-in-WAV format …')

    for i, wav in enumerate(all_wavs):
        if i % 500 == 0 and i > 0:
            print(f'  checked {i}/{len(all_wavs)} …')
        if not is_readable(wav):
            bad_wavs.append(wav)

    print(f'Found {len(bad_wavs)} unreadable .wav files')
    print(f'\nTotal to convert: {len(wv_files) + len(bad_wavs)}')
    print()

    # Convert .wv files
    if wv_files:
        print(f'Converting {len(wv_files)} WavPack files …')
        for i, wv in enumerate(wv_files):
            if i % 100 == 0:
                print(f'  [{i+1}/{len(wv_files)}] {wv.name}')
            convert_wavpack(wv)
        print(f'  Done: {stats["wv_ok"]} converted, {stats["wv_fail"]} failed')

    # Convert OGG-in-WAV files
    if bad_wavs:
        print(f'\nConverting {len(bad_wavs)} OGG-in-WAV files …')
        for i, wav in enumerate(bad_wavs):
            if i % 100 == 0:
                print(f'  [{i+1}/{len(bad_wavs)}] {wav.name}')
            convert_ogg_in_wav(wav)
        print(f'  Done: {stats["ogg_ok"]} converted, {stats["ogg_fail"]} failed')

    # Summary
    total_ok   = stats['wv_ok'] + stats['ogg_ok']
    total_fail = stats['wv_fail'] + stats['ogg_fail']
    print(f'\n=== Summary ===')
    print(f'  Converted: {total_ok}')
    print(f'  Failed:    {total_fail}')

    if total_ok > 0:
        print(f'\n⚠  Re-run instruments_scan.py to update the index with converted files.')


if __name__ == '__main__':
    main()
