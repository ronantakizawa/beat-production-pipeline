# Beat Production Workflow

## Step 1: Research
- Pull YouTube tutorial transcripts for the target genre before writing any code
- Use transcripts as the primary guide for drum patterns, arrangement, and mixing decisions
- Note specific techniques mentioned (e.g., dembow rim accent on 2.5, trap hi-hat rolls, sidechain before snare)
- Download and analyze actual reference tracks with music-analysis MCP (chroma, BPM)
- Cross-check tutorial claims against empirical data — tutorials can be wrong about key/tempo

## Step 2: Compose (compose_*.py)
- Define key, BPM, chord progression, and song structure first
- Keep melodies quiet and lowkey — low velocity (38-58), sparse notes, lots of rests
- Melody should sit behind the drums and bass, not compete with them
- Use music21 for MIDI generation with separate parts per instrument
- Export individual MIDI stems + full arrangement MIDI (track in git for re-rendering)
- Add inline timbre guard comments for critical design decisions (e.g., "MELODY TIMBRE RULE: sine+triangle, not saw") to prevent regressions across versions
- Hook octave pattern: within each 16-bar hook (4 cycles of 4 bars), shift melody octaves as [base, base, +1, +1] — first 8 bars establish, last 8 bars lift. Both hooks must use the same pattern.

## Step 3: Sound Selection
- Rotate sample kits between projects — avoid reusing the same kick/snare/hat across beats
- Available kits: Obie Trap, Juicy Jules Stardust, REGGAETON 4, URBANO, reggaetondrums, and others in /instruments
- Match samples to genre (don't use trap 808s on reggaeton, don't use reggaeton rims on trap)
- Listen to reference tracks in the target genre to guide sample choice
- Consider pitch-shifting samples for character (e.g., hi-hats down -2/-3st for darker feel)

## Step 4: Render (render_*.py)
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
- Always check YouTube transcripts for genre-specific production tips
- Run the full analytics pipeline (LUFS + spectral) on every render
- Pad and lead must be high-passed to stay out of the bass frequency range
- Add timbre guard comments on critical mix decisions to prevent regressions
- Cross-validate tutorial info against real audio analysis — don't trust blindly
- Every render should produce a new version number, never overwrite previous renders
