"""
add_vocals.py -- Add AI-generated rap vocals over a beat.

Supports 3 TTS providers for comparison:
  1. elevenlabs  — highest quality, API-based (needs key)
  2. chatterbox  — open source, runs locally on MPS
  3. bark        — open source, expressive with ad-libs

Pipeline:
  1. Load beat WAV
  2. Generate vocal WAV from lyrics via selected provider
  3. Apply vocal effects (HPF, compressor, reverb, LPF)
  4. Place vocals at beat section positions (hook1, verse, hook2)
  5. Mix vocals over beat
  6. Master and export

Usage:
  python add_vocals.py --beat Mosey_Guitar_v5.wav --name "Mosey_Guitar" \
      --bpm 140 --lyrics lyrics_mosey.txt --provider elevenlabs
"""

import argparse
import os
import sys
import io
import re
import tempfile
import numpy as np
from math import gcd
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import pedalboard as pb
import pyloudnorm as pyln
import glob as _glob

SR = 44100
OUTPUT_DIR = '/Users/ronantakizawa/Documents/Melodic_Trap_Beats'
ELEVENLABS_KEY = 'sk_d1baf45f63570b9ee0a211de3db55260f0a7070e1b114a66'


# ============================================================================
# HELPERS
# ============================================================================

def load_audio(path):
    """Load audio file to stereo float32 numpy array (NSAMP, 2)."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    if data.shape[1] == 1:
        data = np.column_stack([data[:, 0], data[:, 0]])
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        left = signal.resample_poly(data[:, 0], SR // g, orig_sr // g)
        right = signal.resample_poly(data[:, 1], SR // g, orig_sr // g)
        data = np.column_stack([left, right]).astype(np.float32)
    return data


def load_mono(path):
    """Load audio file to mono float32 numpy array."""
    data, orig_sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if orig_sr != SR:
        g = gcd(SR, orig_sr)
        mono = signal.resample_poly(mono, SR // g, orig_sr // g)
    return mono.astype(np.float32)


def apply_pb(arr2ch, board):
    out = board(arr2ch.T.astype(np.float32), SR)
    return out.T.astype(np.float32)


def parse_lyrics(lyrics_path):
    """Parse lyrics file into sections: {section_name: text}."""
    sections = {}
    current = None
    lines = []
    with open(lyrics_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^\[(\w+)\]$', line)
            if m:
                if current and lines:
                    sections[current] = ' '.join(lines)
                current = m.group(1)
                lines = []
            elif line and current:
                lines.append(line)
    if current and lines:
        sections[current] = ' '.join(lines)
    return sections


# ============================================================================
# TTS PROVIDERS
# ============================================================================

def generate_elevenlabs(text, output_path):
    """Generate vocal using ElevenLabs API."""
    from elevenlabs.client import ElevenLabs

    print('    Connecting to ElevenLabs ...')
    client = ElevenLabs(api_key=ELEVENLABS_KEY)

    # Use a built-in voice with hip-hop energy
    # "Adam" is a deep male voice good for rap delivery
    voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam

    print(f'    Generating with voice Adam ...')
    audio_gen = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Collect all chunks
    audio_bytes = b''.join(audio_gen)

    # Save MP3 then convert to WAV
    tmp_mp3 = output_path.replace('.wav', '_tmp.mp3')
    with open(tmp_mp3, 'wb') as f:
        f.write(audio_bytes)

    seg = AudioSegment.from_mp3(tmp_mp3)
    seg.export(output_path, format='wav')
    os.remove(tmp_mp3)
    print(f'    Saved: {output_path} ({len(seg)/1000:.1f}s)')


def generate_chatterbox(text, output_path):
    """Generate vocal using Chatterbox TTS (local, MPS)."""
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f'    Loading Chatterbox on {device} ...')
    model = ChatterboxTTS.from_pretrained(device=device)

    print('    Generating speech ...')
    wav = model.generate(text, exaggeration=1.2, cfg_weight=0.5)
    torchaudio.save(output_path, wav, model.sr)

    dur = wav.shape[1] / model.sr
    print(f'    Saved: {output_path} ({dur:.1f}s)')


def generate_bark(text, output_path):
    """Generate vocal using Bark TTS (local, expressive)."""
    from bark import SAMPLE_RATE, generate_audio, preload_models

    print('    Loading Bark models ...')
    preload_models()

    print('    Generating speech ...')
    # Bark works best with shorter segments
    audio = generate_audio(text, history_prompt="v2/en_speaker_6")

    # Save as WAV
    wavfile.write(output_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    dur = len(audio) / SAMPLE_RATE
    print(f'    Saved: {output_path} ({dur:.1f}s)')


PROVIDERS = {
    'elevenlabs': generate_elevenlabs,
    'chatterbox': generate_chatterbox,
    'bark': generate_bark,
}


# ============================================================================
# VOCAL PROCESSING
# ============================================================================

def process_vocal(vocal_mono):
    """Apply vocal effects chain: HPF, compressor, reverb, LPF."""
    vocal_stereo = np.column_stack([vocal_mono, vocal_mono]).astype(np.float32)

    vocal_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=100),
        pb.Compressor(threshold_db=-12, ratio=3.0, attack_ms=4, release_ms=120),
        pb.Reverb(room_size=0.30, damping=0.50, wet_level=0.15,
                  dry_level=0.90, width=0.70),
        pb.LowpassFilter(cutoff_frequency_hz=12000),
        pb.Gain(gain_db=1.0),
        pb.Limiter(threshold_db=-2.0),
    ])
    return apply_pb(vocal_stereo, vocal_board)


def place_vocal(beat_buf, vocal_stereo, start_sec, gain=0.40):
    """Mix vocal into beat buffer at a given time position."""
    s = int(start_sec * SR)
    n = min(len(vocal_stereo), len(beat_buf) - s)
    if n <= 0:
        return
    beat_buf[s:s + n] += vocal_stereo[:n] * gain


# ============================================================================
# MIX & MASTER
# ============================================================================

def master_and_export(mix, beat_name, provider, bpm):
    """Master the vocal+beat mix and export."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Version
    pattern = f'{beat_name}_Vocal_{provider}_v*.mp3'
    existing = _glob.glob(os.path.join(OUTPUT_DIR, pattern))
    version = max([int(os.path.basename(p).split('_v')[1].split('.')[0])
                   for p in existing], default=0) + 1
    vstr = f'v{version}'

    OUT_WAV = os.path.join(OUTPUT_DIR, f'{beat_name}_Vocal_{provider}_{vstr}.wav')
    OUT_MP3 = os.path.join(OUTPUT_DIR, f'{beat_name}_Vocal_{provider}_{vstr}.mp3')

    print(f'\n  Mastering {provider} ...')

    # Master chain
    master_board = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=25),
        pb.LowpassFilter(cutoff_frequency_hz=19000),
        pb.Compressor(threshold_db=-14, ratio=2.0, attack_ms=15, release_ms=200),
        pb.Gain(gain_db=-0.5),
    ])
    mix = apply_pb(mix, master_board)

    # Fade out last 4 bars
    BAR_DUR = (60.0 / bpm) * 4
    NBARS = int(len(mix) / SR / BAR_DUR)
    fade_bars = min(4, NBARS // 4)
    fade_start = int((NBARS - fade_bars) * BAR_DUR * SR)
    fade_end = len(mix)
    fade_len = fade_end - fade_start
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len) ** 2
        mix[fade_start:fade_end, 0] *= fade_curve
        mix[fade_start:fade_end, 1] *= fade_curve

    # LUFS normalization
    ln_meter = pyln.Meter(SR, block_size=0.400)
    limit_board = pb.Pedalboard([pb.Limiter(threshold_db=-1.0)])
    mix = apply_pb(mix, limit_board)

    body_start = int(8 * BAR_DUR * SR)  # skip intro
    body_end = fade_start
    if body_end > body_start:
        lufs = ln_meter.integrated_loudness(mix[body_start:body_end])
        if np.isfinite(lufs):
            gain_db = -14.0 - lufs
            mix = mix * (10 ** (gain_db / 20.0))
            print(f'    LUFS: {lufs:.1f} -> applied {gain_db:+.1f} dB')

    # Export
    out_i16 = (mix * 32767).clip(-32767, 32767).astype(np.int16)
    wavfile.write(OUT_WAV, SR, out_i16)
    seg = AudioSegment.from_wav(OUT_WAV)
    seg.export(OUT_MP3, format='mp3', bitrate='192k', tags={
        'title': f'{beat_name} Vocal ({provider}) {vstr}',
        'artist': 'Claude Code',
        'genre': 'Trap',
    })
    m, s = divmod(int(len(seg) / 1000), 60)
    print(f'    {os.path.basename(OUT_MP3)}: {os.path.getsize(OUT_MP3)/1e6:.1f} MB | {m}:{s:02d}')
    return OUT_MP3


# ============================================================================
# MAIN
# ============================================================================

def run_provider(provider, lyrics_sections, beat_buf, bpm, beat_name, tmp_dir):
    """Generate vocals with one provider and mix over beat."""
    print(f'\n=== Provider: {provider} ===')

    BAR_DUR = (60.0 / bpm) * 4
    hook_text = lyrics_sections.get('hook', '')
    verse_text = lyrics_sections.get('verse', '')

    # Generate hook and verse vocals
    hook_path = os.path.join(tmp_dir, f'{provider}_hook.wav')
    verse_path = os.path.join(tmp_dir, f'{provider}_verse.wav')

    gen_fn = PROVIDERS[provider]

    try:
        print(f'  Generating hook vocal ...')
        gen_fn(hook_text, hook_path)

        print(f'  Generating verse vocal ...')
        gen_fn(verse_text, verse_path)
    except Exception as e:
        print(f'  ERROR: {provider} failed: {e}')
        return None

    # Load generated vocals
    hook_mono = load_mono(hook_path)
    verse_mono = load_mono(verse_path)

    # Process vocals (effects chain)
    print(f'  Processing vocals ...')
    hook_processed = process_vocal(hook_mono)
    verse_processed = process_vocal(verse_mono)

    # Create mix buffer (copy of beat)
    mix = beat_buf.copy()

    # Place vocals at section positions
    # Hook1: bar 8, Verse: bar 24, Hook2: bar 40
    hook1_sec = 8 * BAR_DUR
    verse_sec = 24 * BAR_DUR
    hook2_sec = 40 * BAR_DUR

    print(f'  Placing vocals: hook@{hook1_sec:.1f}s, verse@{verse_sec:.1f}s, hook2@{hook2_sec:.1f}s')
    place_vocal(mix, hook_processed, hook1_sec, gain=0.45)
    place_vocal(mix, verse_processed, verse_sec, gain=0.42)
    place_vocal(mix, hook_processed, hook2_sec, gain=0.45)

    # Master and export
    out_path = master_and_export(mix, beat_name, provider, bpm)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add AI rap vocals over a beat')
    parser.add_argument('--beat', required=True, help='Beat WAV file path')
    parser.add_argument('--name', required=True, help='Beat name')
    parser.add_argument('--bpm', required=True, type=float, help='BPM')
    parser.add_argument('--lyrics', required=True, help='Lyrics text file')
    parser.add_argument('--provider', default='all',
                        choices=['elevenlabs', 'chatterbox', 'bark', 'all'],
                        help='TTS provider (default: all)')
    args = parser.parse_args()

    # Resolve beat path
    beat_path = args.beat
    if not os.path.isabs(beat_path):
        beat_path = os.path.join(OUTPUT_DIR, beat_path)

    # Resolve lyrics path
    lyrics_path = args.lyrics
    if not os.path.isabs(lyrics_path):
        lyrics_path = os.path.join('/Users/ronantakizawa/Documents/instruments', lyrics_path)

    print(f'Beat: {beat_path}')
    print(f'Lyrics: {lyrics_path}')
    print(f'BPM: {args.bpm}')

    # Load beat
    print('\nLoading beat ...')
    beat_buf = load_audio(beat_path)
    print(f'  {len(beat_buf)/SR:.1f}s stereo')

    # Parse lyrics
    lyrics_sections = parse_lyrics(lyrics_path)
    print(f'  Sections: {list(lyrics_sections.keys())}')
    for sec, text in lyrics_sections.items():
        print(f'    [{sec}]: {len(text)} chars')

    # Run providers
    providers = list(PROVIDERS.keys()) if args.provider == 'all' else [args.provider]

    results = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for prov in providers:
            out = run_provider(prov, lyrics_sections, beat_buf, args.bpm,
                               args.name, tmp_dir)
            if out:
                results[prov] = out

    # Summary
    print('\n' + '=' * 50)
    print('RESULTS:')
    for prov, path in results.items():
        print(f'  {prov}: {path}')
    if not results:
        print('  No providers succeeded.')
    print('=' * 50)
