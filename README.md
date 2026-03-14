# Beat Production Pipeline

End-to-end toolchain for analyzing reference tracks and producing beats programmatically. Compose MIDI arrangements in Python, render with real samples and DSP, then compare against reference tracks.

## Pipeline Overview

```
Reference Track (.mp3/.wav)
        |
   analyze_reference.py    --> ref_analysis.json + separated stems
        |
   compose_*.py            --> MIDI stems + full arrangement
        |
   render_*.py             --> mixed WAV/MP3 with effects, samples, mastering
        |
   compare_beats.py        --> similarity score vs reference (6 dimensions)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `analyze_reference.py` | 8-step reference analyzer: Demucs stem separation, BPM/key/loudness detection (essentia), chord progression, bass pattern (basic-pitch), drum transcription, melody extraction, song structure detection. Outputs JSON profile. |
| `skeleton_compose.py` | Compose template. Define chords, bass, drums, melody layers as data tables, outputs MIDI stems via music21. |
| `skeleton_render.py` | Render template. Loads MIDI + audio samples, applies DSP (FAUST synths, pedalboard effects, sidechain compression), exports mixed WAV/MP3. |
| `skeleton_compare.py` | Beat similarity scorer. Compares two audio files across 6 dimensions (tempo, harmony, dynamics, spectral, texture, groove) using essentia's MusicExtractor (~400 features). |
| `instruments_scan.py` | Scans sample directories and builds a searchable JSON index of all audio files with metadata. |
| `instruments_query.py` | Searches the sample index by instrument type, key, BPM, etc. |
| `convert_formats.py` | Batch audio format conversion utility. |

## Setup

```bash
pip install -r requirements.txt
```

**Note:** `demucs` and `basic-pitch` are only required for `analyze_reference.py`. The compose/render scripts work without them.

If basic-pitch fails to load its model, install the ONNX runtime:
```bash
pip install onnxruntime
```

## Usage

### Analyze a reference track
```bash
python analyze_reference.py /path/to/reference.mp3
# Outputs: ref_analysis.json + ref_stems/ directory
```

### Compose a beat
```bash
# Copy skeleton_compose.py, fill in chord tables and melody patterns
python compose_mybeat.py
# Outputs: MIDI stems + full arrangement
```

### Render to audio
```bash
# Copy skeleton_render.py, configure samples and effects
python render_mybeat.py
# Outputs: versioned WAV + MP3
```

### Compare against reference
```bash
python skeleton_compare.py my_beat.wav reference.wav
# Outputs: similarity report across 6 dimensions
```

## Workflow

See [workflow.md](workflow.md) for the full production workflow (research, compose, sound selection, render, mix check, compare, export).
