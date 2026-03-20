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

## Step 3: Sound Selection (MANDATORY — use instruments_query.py)

**ALWAYS use `SampleSelector` for sample-based instruments. NEVER hardcode sample paths.**

```python
from instruments_query import SampleSelector

sel = SampleSelector(genre='reggaeton', beat='MyBeat_v1', key='A', seed=42)

# Drums
KICK   = sel.pick('kick')
SNARE  = sel.pick('snare')
HH_CL  = sel.pick('hat_closed')
HH_OP  = sel.pick('hat_open')
PERC   = sel.pick('perc')

# Melodic (sample-based — prefer over FAUST for realism)
MELODY = sel.pick('melody')
PAD    = sel.pick('pad')
BASS   = sel.pick('bass')

# Auto pitch/gain info:
sel.info['melody']['pitch_st']   # semitones to shift to match song key
sel.info['melody']['gain_db']    # dB correction for target loudness
sel.info['melody']['root_midi']  # for sample-based pitch shifting

sel.save()  # logs to usage_log.json — next beat auto-avoids these
```

### Sample vs FAUST decision:
- **Use samples from catalog** when the index has good matches (melodic_oneshot with is_tonal=True, good key_confidence). The catalog has 5,557 melodic one-shots and 3,059 melodic loops — flutes, synths, bells, guitars, pianos, strings. These sound more realistic than FAUST oscillators.
- **Use FAUST DSP only** for sub bass (sine osc below 200Hz) or when no catalog sample fits. FAUST is a last resort for melody/pad, not the default.
- **Use SF2 soundfonts** for instruments not in the WAV catalog (brass, choir, tubular bells). Query presets with `fluidsynth.Synth.sfpreset_name()`.

### Sample-based melody rendering pattern:
```python
# Instead of faust_render(dsp, freq, gate, gain):
from instruments_query import SampleSelector, pitch_offset_st
sample = load_sample(sel.pick('melody'))
root_midi = sel.info['melody']['root_midi']

for note_start, note_num, vel, dur in melody_notes:
    semitones = note_num - root_midi  # pitch shift from sample's root
    shifted = pitch_shift_sample(sample, semitones)
    place(buf_L, buf_R, shifted, int(note_start * SR), gain=vel/127.0)
```

### If using Path A (reference analysis):
- Check `instruments.bass.type` — if "808", pick role='808'; if "sub_bass", use FAUST sub
- Check `instruments.drums.*_type` — match kick/snare/hh sub-types via role filters
- Check `instruments.melody.*.type` — query catalog with matching extra_filters
- Use `instruments.*.features` to verify: brightness, attack, decay should be in the same ballpark

### If using Path B (tutorial transcripts):
- Follow sample type recommendations from the tutorial
- Match samples to genre (don't use trap 808s on reggaeton, don't use reggaeton rims on trap)

### General sound selection rules:
- **SampleSelector auto-rotates**: it excludes paths/packs used in the last 4 beats via usage_log.json
- CLI check before starting: `python instruments_query.py --role melody --genre reggaeton --key A`
- Consider pitch-shifting samples for character (e.g., hi-hats down -2/-3st for darker feel)
- For stems/variants: query multiple candidates with `query(role, n=6)` and render each

## Step 4: Render (render_*.py)

### If using Path A (reference analysis):
- Target `mix_profile.eq_balance_db` values for your EQ settings
- Match `mix_profile.stereo_width` per band — keep sub mono, widen highs
- Set reverb send to approximate `mix_profile.reverb_decay_s`
- If `vocal_presence` is low across all sections, skip vocal processing chain
- Target the reference's `loudness_lufs` for your master level

### General render rules:
- Step 0: MIDI fix — strip duplicate tempo events, save as *_FIXED.mid
- Load samples via SampleSelector (Step 3) — NEVER hardcode paths
- Prefer catalog samples over FAUST for melody/pad (more realistic). Use FAUST only for sub bass or when no catalog match exists
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
- **All workflow/task files go in `/Users/ronantakizawa/Documents/instruments/`** — never in project subdirectories

## Sample-Based Tracks (render_groove.py)

- **Unique percussion per track**: Never use the same kick, clap, or hat across multiple tracks. Use sample spectral analysis (centroid, warmth, brightness, onset rate) to score pools of candidates, then hash the track name with per-category seeds to guarantee unique combinations.
- **Beat-aligned loops**: Use `librosa.beat.beat_track()` to find actual downbeats in the source sample. Snap loop start to detected bar positions — never use onset_strength grid alone, it drifts.
- **Tempo detection**: Always try multiple ranges (default + hinted at target BPM). Check half/double of detected tempo. Piano samples especially get misdetected (e.g. 161 BPM instead of ~80).
- **Underwater LPF sections**: Breakdowns must use real LPF sweeps (scipy butter, block-wise), not just volume ducks. Sweep 12kHz→350Hz→12kHz for contrast. Intro starts submerged, outro sinks.
- **Version numbering**: Check both base output dir AND subdirectories (e.g. Groove_Tracks/) when globbing for existing versions, or versions reset to v1 after moving files.

## Reminders
- **ALWAYS use `SampleSelector` from `instruments_query.py`** — never hardcode sample paths. The catalog has 15,644 instruments indexed with pitch, key, brightness, attack, and more
- **NEVER default to FAUST `os.osc(freq)` for melody/pad** — query the catalog first. FAUST sine/saw oscillators are a last resort, not the default. The catalog has 5,557 melodic one-shots (flutes, synths, bells, guitars, pianos) that sound far more realistic
- **NEVER reuse the 5x detuned saw pad** (the `PAD_DSP` with `os.sawtooth(freq) + os.sawtooth(freq * 1.009) + os.sawtooth(freq * 0.991) + os.sawtooth(freq * 1.018) + os.sawtooth(freq * 0.982)`) — it's been used on too many beats. Pick a different instrument: drawbar organ, FM pad, string ensemble sample, etc.
- **Call `sel.save()` after every render** — this logs usage so the next beat auto-avoids the same samples
- **Check `python instruments_query.py --history`** before starting a new beat to see what was used recently
- Avoid keys of C (C major, C minor) — overused and generic-sounding. Pick keys with more character (F minor, G minor, D minor, Ab major, etc.)
- Melodies should be quiet and lowkey — never louder than drums/bass
- Run the full analytics pipeline (LUFS + spectral) on every render
- Pad and lead must be high-passed to stay out of the bass frequency range
- Add timbre guard comments on critical mix decisions to prevent regressions
- Every render should produce a new version number, never overwrite previous renders
- Always use the sample's detected `root_midi` from the index for pitch shifting. Never hardcode ref_midi. Shift formula: `target_midi - root_midi`
- Buildups/transitions: **4 bars max**, not 8
- Transition FX: **short and minimal** — 1-2 elements max
