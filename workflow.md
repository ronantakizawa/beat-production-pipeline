# Beat Production Workflow

## Step 1: Research — Pick ONE Source of Truth

You have two research paths. Pick one per project and commit to it.

### Path A: Reference Track Analysis (preferred)
Run `analyze_reference.py` on a reference track you want to match.
```bash
python analyze_reference.py /path/to/reference.mp3
```
This gives you everything you need to clone the vibe:
- **BPM, key, scale** — use these exactly, don't guess
- **Chord progression + voicings** — copy the cycle and voice chords at the given octaves
- **Drum pattern** — kick/clap/hh grid positions, swing %, triplet hats, crash interval
- **Bass pattern** — rhythm + roots per chord bar
- **Melody layers** — notes per register (pad/main/top), cycle length
- **Song structure** — section boundaries (intro/verse/hook/bridge) with bar ranges
- **Genre + mood + energy** — confirms what you're building
- **Instruments** — bass type (808/sub/synth/pluck), drum sub-types, melody timbres (strings/pluck/lead_synth/bell)
- **Mix profile** — stereo width per band, EQ balance, reverb decay, vocal presence
- **Energy curve** — per-bar dynamics to shape your arrangement

Use `ref_analysis.json` as your spec. Every decision in compose/render should trace back to a field in this JSON.

### Path B: YouTube Tutorial Transcripts
Pull transcripts via the youtube-transcript MCP for the target genre.
```
get_transcript(video_id="S4CdACq0TuI")
```
Extract and follow:
- BPM, key, scale choices
- Chord voicing techniques (half-step tension, suspended chords, inversions)
- Drum pattern construction order (snare first, then hats, then 808)
- Mixing levels (kick+808 loudest, snare -2/-3dB, melody -6dB)
- Arrangement structure (intro bars, hook length, bridge placement)
- Genre-specific techniques (dembow rim accent, trap hat rolls, sidechain timing)

Tutorials give you *why* and *how*. Analysis gives you *what*. If a tutorial contradicts the analysis data, trust the analysis — tutorials can be wrong about key/tempo.

### Do NOT mix paths without declaring priority
If you use both, state which one wins on conflicts. Default: analysis data overrides tutorial claims.

## Step 2: Compose (compose_*.py)

### If using Path A (reference analysis):
- Set BPM, key, scale from `ref_analysis.json`
- Copy chord cycle verbatim from `chords` + `chord_voicings`
- Replicate drum grid positions from `drum_pattern` (kick/clap/hh arrays)
- Match swing % from `swing_pct`
- Bass rhythm from `bass_pattern`, roots from `bass_roots`
- Melody: use note density and register split from `melody_layers`
- Structure: match section boundaries from `sections`
- Energy: follow `energy_curve` shape for dynamics across bars

### If using Path B (tutorial transcripts):
- Follow the tutorial's BPM, key, chord progression
- Build drums in the order the tutorial specifies
- Apply the mixing level hierarchy from the tutorial
- Use the arrangement structure from the tutorial

### General compose rules:
- **Chord voicing reference (ALWAYS use):** Before writing any chord voicings, check `/Users/ronantakizawa/Documents/instruments/chordprogressions/` for the target key. This directory contains:
  - `GitHub Free Progressions/{KEY}/{KEY} Minor/` — minor key triads with proper voice leading
  - `GitHub Free Progressions/{KEY}/{KEY} Ninths/` — extended 5-note jazz voicings
  - `GitHub Free Progressions/{KEY}/{KEY} Sevenths/` — 7th chord voicings
  - `More Genres/Trap Dark.mid`, `Trap Hard.mid`, `Trap Melodic.mid` — genre-specific trap voicings
  - `More Genres/Reggae *.mid` — reggae chord patterns
  - `EDM Progressions/` — EDM chord progressions
  - Analyze the MIDI files with mido to extract note names, octave ranges, and voicing density. Match your voicings to the reference register (typically octave 4 for triads) and include all chord tones (don't drop the 3rd — it defines major/minor).
- Melody source: derive melodies from chord top notes + passing tones from the key's scale, using the chordprogressions reference voicings as the harmonic foundation
- Keep melodies quiet and lowkey — low velocity (38-58), sparse notes, lots of rests
- Melody should sit behind the drums and bass, not compete with them
- Use music21 for MIDI generation with separate parts per instrument
- Export individual MIDI stems + full arrangement MIDI (track in git for re-rendering)
- Add inline timbre guard comments for critical design decisions (e.g., "MELODY TIMBRE RULE: sine+triangle, not saw")
- Hook octave pattern: within each 16-bar hook (4 cycles of 4 bars), shift melody octaves as [base, base, +1, +1]

## Step 3: Sound Selection

### If using Path A (reference analysis):
- Check `instruments.bass.type` — if "808", pick an 808 sample; if "sub_bass", pick a sub; etc.
- Check `instruments.drums.*_type` — match kick/snare/hh sub-types
- Check `instruments.melody.*.type` — match pad/main/top timbres (strings/pluck/lead_synth/bell)
- If `instruments.*.closest_samples` exists, start with those matches from your library
- Use `instruments.*.features` to verify: brightness, attack, decay, sub energy should be in the same ballpark
- Use `SampleSelector` from `instruments_query.py` with the detected genre:
  ```python
  from instruments_query import SampleSelector
  sel = SampleSelector(genre=ref['genre'], key=ref['key'])
  KICK = sel.pick('kick')
  ```

### If using Path B (tutorial transcripts):
- Follow sample type recommendations from the tutorial
- Match samples to genre (don't use trap 808s on reggaeton, don't use reggaeton rims on trap)

### General sound selection rules:
- Rotate sample kits between projects — avoid reusing the same kick/snare/hat across beats
- Consider pitch-shifting samples for character (e.g., hi-hats down -2/-3st for darker feel)

## Step 4: Render (render_*.py)

### If using Path A (reference analysis):
- Target `mix_profile.eq_balance_db` values for your EQ settings
- Match `mix_profile.stereo_width` per band — keep sub mono, widen highs
- Set reverb send to approximate `mix_profile.reverb_decay_s`
- If `vocal_presence` is low across all sections, skip vocal processing chain
- Target the reference's `loudness_lufs` for your master level

### General render rules:
- Step 0: MIDI fix — strip duplicate tempo events, save as *_FIXED.mid
- FAUST DSP for synths (pad, lead) via dawdreamer
- Pedalboard for effects chains (EQ, compression, reverb, limiting)
- Sidechain bass/pad/lead to kick envelope
- EQ separation: pad HPF at 150Hz, lead HPF at 200Hz, bass LPF at 1200Hz
- Humanize timing (+/-8-14ms jitter) and velocity (+/-3-7 range) on hats, snares, melody
- Use reproducible random seed (seed=42) so renders are deterministic
- Auto-increment version numbers — scan for existing *_v*.mp3, output v{N+1}
- Add atmosphere layers where genre calls for it (vinyl crackle, ambient pads, risers before drops)

## Step 5: Mix Check
- LUFS normalization pipeline (order matters):
  1. Master chain: HPF + LPF + compressor + saturation + gain — NO limiter yet
  2. Fade out last 4 bars (quadratic curve)
  3. Measure LUFS on body only (exclude intro + fade-out so quiet sections don't skew)
  4. Apply gain to hit -14 LUFS target
  5. Final true-peak limiter AFTER normalization at -1.0 dB ceiling
- Per-section loudness: measure LUFS/peak/RMS for each section (intro, hook, verse, bridge)
  - Flag any section >3 dB over target or with peak above -0.5 dB
- Artifact detection (automated, runs every render):
  - Clipping: count samples above 0.99
  - DC offset: mean of each channel should be <0.005
  - Phase cancellation: mono RMS vs stereo RMS — flag if >3 dB loss
  - Silent sections: flag any section with peak <0.01
- librosa analysis (subprocess due to dawdreamer/numba LLVM conflict):
  - Spectral centroid — should be 1000-3000Hz for balanced mix
  - Spectral bandwidth — watch for too narrow (muddy) or too wide (harsh)
  - RMS — typical range -18 to -12 dB
- If warnings are raised, fix stems and re-render before calling it done

## Step 6: Compare Against Reference
- Use compare_beats.py pattern (Essentia MusicExtractor, ~400 features) to score the beat against a real reference track
- Score across 6 dimensions: Tempo/Rhythm (25%), Key/Harmony (20%), Spectral (20%), Dynamics (15%), Texture (10%), Groove (10%)
- Use the scores to identify which aspects need work before calling it done

## Step 7: Export
- Export full mix as WAV + MP3 with metadata tags (title, artist, album, genre)
- Write MP3 tags with version number for tracking

## Step 8: File Organization
- .gitignore: *.wav, *.mp3, *_FIXED.mid, __pycache__/
- Track in git: compose_*.py, render_*.py, individual MIDI stems, full arrangement MIDI
- Each project gets its own directory and GitHub repo

## Reminders
- Avoid keys of C (C major, C minor) — overused and generic-sounding. Pick keys with more character (F minor, G minor, D minor, Ab major, etc.)
- Melodies should be quiet and lowkey — never louder than drums/bass
- Don't default to the same samples every time — explore the kits
- Run the full analytics pipeline (LUFS + spectral) on every render
- Pad and lead must be high-passed to stay out of the bass frequency range
- Add timbre guard comments on critical mix decisions to prevent regressions
- Every render should produce a new version number, never overwrite previous renders
