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

## Sample-Based Trap Tracks (render_trap.py)

- **Local beat-tracking for loop extraction**: NEVER beat-track the entire audio file to find loop boundaries — the global beat grid drifts and doesn't match local bar positions. Instead, extract a generous chunk starting at the user's loop start point, then call `librosa.beat.beat_track()` on just that chunk. Bar boundaries from local tracking are accurate; global tracking is not.
- **Cut loops to exact bar boundaries**: The loop must end at an exact beat-tracked bar boundary (every 4th beat). If the loop length isn't an exact multiple of the bar duration, the drum grid will drift when the loop repeats. Find the bar boundary closest to the target duration and cut there.
- **Derive BPM from loop length**: After cutting to an exact bar boundary, compute `bpm = n_bars * 4 * 60.0 / loop_duration`. This locks the drum grid precisely to the sample. Never use librosa's detected BPM directly — it's an estimate that won't match the actual loop length.
- **Don't trim musical content for alignment**: If the beat tracker says the first beat is at 1.07s, that doesn't mean 0-1.07s is silence. Listen first / check RMS. The music may already be playing. The correct start point is where the musical phrase begins, not where the beat tracker places beat 1.
- **Half-time feel**: Kick on beats 1 and 3, snare on beat 3 (0-indexed: beat 2). 808 bass follows kick with longer sustain. Hi-hats are 16th notes with 32nd rolls on hooks.
- **Trap BPM range**: Samples often detect at ~76 BPM (half-time). The actual trap feel is ~152 BPM (double-time) but the half-time value is what locks to the sample.
- **Metronome debugging**: When alignment is off, render the loop x2 with a metronome overlay to verify. Use the exact same beat interval derived from the loop (`loop_length / (n_bars * 4)`) so clicks stay locked across repeats.

## Sample Scouting Pipeline (sample_scout.py)

When sourcing samples from the internet, don't pick blindly by title or view count. Use `sample_scout.py` to search, download, and auto-rank candidates by audio quality before rendering.

### Commands:
```bash
# Search YouTube and list results
python sample_scout.py search "dark guitar trap loop"

# Full pipeline: search → download → stem-separate → score → rank
python sample_scout.py scout "dark guitar trap loop" --top 5

# Score and rank existing downloaded samples
python sample_scout.py rank samples/stems/htdemucs/

# Score a single file
python sample_scout.py score path/to/no_drums.wav
```

### Scoring metrics (each 0-1, higher = better):
| Metric | Weight | What it measures |
|--------|--------|-----------------|
| **Loop-ability** | 30% | Best chroma autocorrelation peak — does the sample repeat cleanly? |
| **Tonal clarity** | 20% | Spectral flatness (low = tonal/melodic, high = noisy) |
| **Tempo consistency** | 15% | Onset interval variance — steady tempo locks to the grid |
| **Frequency balance** | 15% | Energy in mids (200-4kHz) — penalizes bass-heavy samples that clash with 808s |
| **Dynamic range** | 10% | RMS variance — not too compressed, not too spiky |
| **Duration** | 10% | 15-180s ideal — enough material without filler |

### Sample selection rules:
- Always `scout` or `rank` before committing to a sample — never pick by title/views alone
- After stem separation, score the `no_drums.wav` (not the raw audio)
- Total score > 0.80 = good candidate, > 0.85 = excellent
- Low loop-ability (< 0.70) means the sample won't loop cleanly — avoid or manually set `--loop-start`

## Shared Modules (`/instruments/`)

Reusable code lives in shared modules — never duplicate across renderers.

| Module | Contents |
|--------|----------|
| `audio_utils.py` | `load_sample`, `place`, `apply_pb`, `stereo_widen`, `create_sample_bed`, `lpf_sweep`, `pitch_shift_sample`, `auto_gain_sample`, `adaptive_hpf`, `add_metronome` |
| `sample_analysis.py` | `analyze_sample_character`, `detect_sample_tempo`, `detect_sample_key`, `detect_loop_period`, `score_loop_candidates`, `extract_loop_auto`, `extract_loop_at` |
| `gross_beat.py` | `gb_reverse`, `gb_stutter`, `gb_gate`, `gb_scratch`, `gb_tape_stop`, `gb_halftime`, `apply_gross_beat` |
| `mix_master.py` | `master_chain`, `lufs_normalize`, `fade_out`, `export_audio`, `mix_analysis`, `master_and_export` |
| `skeleton_fetch.py` | YouTube search, download, stem-separate, prepare |
| `skeleton_scout.py` | Quality scoring pipeline (search → download → score → rank) |

Import pattern for new renderers:
```python
import sys; sys.path.insert(0, '/Users/ronantakizawa/Documents/instruments')
from audio_utils import load_sample, place, apply_pb, create_sample_bed, auto_gain_sample, adaptive_hpf
from sample_analysis import detect_sample_tempo, detect_sample_key, extract_loop_auto
from gross_beat import apply_gross_beat
from mix_master import master_and_export
```

## Sample Auto-Gain & Adaptive HPF

Every sample must be normalized before mixing so level and low-end are consistent regardless of source.

- **Auto-gain**: Normalize sample bed to **-18 dB RMS** via `auto_gain_sample()`. Loud samples get turned down, quiet ones up. The mix multiplier (e.g. `0.45`) then means the same thing for every sample.
- **Adaptive HPF**: Measure low-end energy ratio of the loop via `adaptive_hpf()`. If >20% of energy is below 300Hz, raise the HPF cutoff (up to 200-220Hz) to prevent the sample from muddying the 808/sub bass range. Baseline: 150Hz for trap, 120Hz for boom-bap.

## BPM Sanity Check (boom-bap / slow genres)

After deriving BPM from `n_bars * 4 * 60 / loop_dur`, the beat tracker may have counted too many bars (e.g. 3 bars at 125 BPM when it should be 2 bars at 83 BPM). **Always compare derived BPM against the target hint** and try all possible bar counts (1 to n_bars), picking the one closest to the target. This is critical for slow genres like boom-bap where the beat tracker tends to double-count.

## Gross Beat-Style FX (`/instruments/gross_beat.py`)

Time/volume manipulation effects applied to the **sample channel only** (drums play through unaffected). Effects are randomized per track (seeded by track name) and placed only at transition points.

### Available effects:
| Effect | Function | Wet/Dry |
|--------|----------|---------|
| **reverse** | `gb_reverse()` | 100% wet |
| **stutter** | `gb_stutter(divisions, wet=0.5)` | 50% blend (div 4/6/8) |
| **gate** | `gb_gate(rate, wet=0.5)` | 50% blend |

### Placement rules:
- **Only at transitions**: rises into hooks, end of song into outro
- **Never mid-section**: keep hooks/verses clean
- **Randomized**: `apply_gross_beat()` picks 1-N effects from the pool, seeded by track name — each song gets a different combo
- Effects on sample only, drums always play through
- Stutter and gate use 50% wet/dry blend for subtlety

## Old-School Boom-Bap Tracks (render_boombap.py)

- **BPM range**: 70-95 BPM. Use BPM sanity check (try all bar counts, pick closest to target) — beat tracker over-counts bars on slow samples.
- **Full-time drums**: Kick on beats 1 & 3, snare on beats 2 & 4 (NOT half-time like trap). No clap layer.
- **Simple hi-hats**: Straight 8th notes only — no 16th/32nd rolls. Open hat on "and of 4" every 2-4 bars.
- **Sub bass, not 808**: Pure sine oscillator via `generate_sub_bass()` with subtle 2nd harmonic. LPF at 200Hz.
- **Drum break layer**: Time-stretch a break from `oldschoolhiphop/Loops/` to match BPM, mix at 20-30% under individual hits, LPF at 8kHz. Only in hooks.
- **20ms humanization jitter**: Much larger than trap's 4ms — simulates a live drummer.
- **Lo-fi processing**: `lofi_process()` on sample bus — bit-crush (12-bit), tape saturation, high-shelf cut. Plus `vinyl_noise()` crackle overlay on full mix.
- **Parallel drum compression**: 70% dry + 30% heavy-compressed for tape-style grit.
- **Vintage sample EQ**: HPF 120-200Hz (adaptive), LPF 12kHz, cut highs for authentic feel.
- **Arrangement**: 64 bars — Intro(8) → Hook(16) → Verse(16) → Hook(16) → Outro(8). Gradual element entry in intro (sample → hats → kick → snare+bass).
- **Master LPF at 16kHz** (darker than trap's 18kHz).

## Roadmap / Future Improvements

### Sample Pipeline
- **Chord detection on samples** — we detect key but not the chord progression. Bass currently follows a generic `[0, 0, -5, -3]` pattern. If we detected the actual chords, bass and any added layers would follow the harmony.
- **Smarter loop alignment** — beat tracker still miscounts bars regularly. Could use onset-based alignment (find the strongest transient) instead of relying purely on librosa's beat tracker.
- **Auto tempo-match without pitch shift** — currently BPM is locked from loop length. If the sample is 95 BPM but we want 122, we have no clean way to time-stretch without artifacts.

### Architecture
- **Unified renderer** — trap, boom-bap, and jazz house share 70% of their code. A single `render_beat.py` with a genre config (JSON/dict) for drum patterns, BPM range, jitter, arrangement, mix levels would eliminate 3 separate 800-line files.
- **Preset system** — `genre_presets.json` defining: BPM range, drum pattern (kick/snare/hat positions), arrangement structure, mix levels, FX chain, lo-fi intensity. Adding a new genre = adding a JSON block, not writing a new renderer.

## Reminders
- **BPM HARD CAP: 150 BPM maximum** — no genre should ever render above 150 BPM. If a sample detects above 150, halve it. Breakcore targets ~100 BPM, trap ~148, boom-bap ~90, jazz house ~122. Fast tempos sound rushed and unmusical.
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
- **Hi-hats: SIMPLE by default** — straight 8th notes with on-beat/off-beat velocity contrast. No 16th rolling, no 32nd rolls, no triplet hats unless a reference track analysis explicitly shows them. Complex hat patterns distract from melodies and sound overproduced.
- **Snares/claps: SIMPLE by default** — clap on 2 and 4 only. No ghost snares, no syncopated extra hits, no snare rolls mid-bar. The backbeat should be steady and predictable. Only add ghost notes if the reference analysis `drum_pattern` contains them.
