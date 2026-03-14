# Instruments Library Reference
`/Users/ronantakizawa/Documents/instruments/`

Quick-reference for choosing samples when building a new beat. Organized by genre/use-case.

---

## Production Rules (Always Apply These)

### Melody
- **Keep it lowkey and quiet.** The melody should sit underneath the drums and bass, not on top. In the mix, melody/arp/bell layers should be lower in volume than the kick and snare — they add texture and color, not dominance.
- **Leave space.** Sparse is better than busy. Rests and silence are part of the melody. If every beat has a note, pull some out.
- **Avoid washing it out.** Reverb on melody should be moderate — enough for space, not so much the notes blur together. Dry level should stay prominent.

### Instrument Variety
- **Don't default to the same sounds every song.** Actively rotate between packs and instrument types. If the last beat used Rhodes, try flute, piano, or synplant textures next time. If it used a Reese bass, try a sub 808 or a picked bass instead.
- **Layer differently each time.** Pad + arp is one approach. Bell melody + minimal bass is another. Stab + no pad is another. Mix it up.
- **Check `instruments_index.json` before picking.** Query by `type`, `freq_band`, and `pack` to find something you haven't used recently rather than reaching for the same sample.

---

## Format Notes (Read First)

| Format | How to load |
|--------|-------------|
| `.wav` (standard) | `soundfile.read()` directly |
| `.wav` (OGG-in-WAV, tag 0x674F) | FL Studio Keyboard/Rhodes samples. `soundfile` fails silently. Fix: `ffmpeg -f ogg -i input.wav output_fixed.wav` |
| `.wv` (WavPack) | ModeAudio drum samples. `soundfile` fails. Fix: `ffmpeg -y -i input.wv output.wav` |
| `.zpa` | Sidecar metadata files (FL Studio). Ignore — not audio. |

> FL Studio's `Packs/Drums (ModeAudio)` is all `.wv`. Convert before use.
> FL Studio's `Packs/Instruments` keyboard samples (Rhodes, piano, etc.) are OGG-in-WAV.
> All converted files live in `FL_Studio/converted/`.

---

## By Genre / Use Case

### Reggaeton / Latin Urban

**`REGGAETON 4/`** — 87 files, clean one-shots
- `Kicks/` — punchy dembow kicks (el.obie style)
- `Snare/` — 20 numbered snares (`Snare @el.obie 1–20.wav`)
- `Reese Bass/` — Reese bass one-shot at C2 root (`Bass @el.obie 1.wav`)
- `Perc/`, `Loops/` — reggaeton perc loops
- **Best for:** Dembow kick, Reese bass (pitch-shift to A1/G1/F1/E1 for sub)

**`reggaetondrums/`** — 173 files, "No Me Conoce" kit
- `Bombos (Kicks)/` — kicks
- `Redoblantes (Snares)/` — snares, labeled by song (`NO ME CONOCE_SNARE_3.wav`)
- `Platos (Cymbals)/` — open hats, crashes
- `Timbales (Timbals)/` — timbales + fills (`NO ME CONOCE_TIMBAL_1.wav`, `TIMBAL FILL_1.wav`)
- `Percu/Bells & Guiros/` — cencerro, guiro (`NO ME CONOCE_CENCERRO_1.wav`, `NO ME CONOCE_GUIRO_1.wav`)
- `Percu/Palos & Toms/` — palitos
- `Percu/Rims & Bongos/` — bongos (`BONGO_1.wav`, `BONGO_2.wav`)
- `Percu/Vox/` — vox percussion
- `Percu/Rolls & Fills/`, `Percu/Special/`
- `Drum Loops/` — full loops + MIDI
- **Best for:** Authentic reggaeton percussion layer (Latin perc, cencerro, guiro, timbales)

**`reggaeton3/URBANITO Producer Bundle/`** — 104 files
- `URBANO Drum Kit/` — hats, claps, claves, kicks, snares
  - `Hats/Hat 1–Hat 30` (closed), `Hat 21` (open hat)
  - `Claps/Clap 1–10`
- `URBANO Bass One-Shots Kit/`, `URBANO Sub One-Shots Kit/` — bass one-shots
- `URBANO Drum Loops/`, `URBANO Percussion Loops/` — loops
- `URBANO Guitar Loops/`, `URBANO Melody Loops/` — melodic loops
- `URBANO Basslines MIDI Kit/`, `URBANO Melody MIDI Kit/` — MIDI files
- **Best for:** Urban/urbano hats, claps, claves as secondary layers

**`reggaeton5/`** — 361 files, XXL Reggaeton Drums (Prime Loops)
- `XXL Drum Samples/XXL Kicks/` — 95 kicks (`25_Kick.wav`, `31_Low_Drum.wav`)
- `XXL Drum Samples/XXL Snares/` — 74 snares + rims (`54_Snare.wav`, `72_Rim.wav`)
- `XXL Drum Samples/XXL Hi Hats/` — 69 hi-hats (`45_Hat.wav`, `48_Hat2.wav`)
- `XXL Drum Samples/XXL Percussions/` — 83 perc (cowbell, timbal, shaker, sticks, perc hits)
- `XXL Drum Samples/XXL Claps/` — 14 claps
- `XXL Drum Samples/XXL Toms/` — 9 toms (low toms, laser tom)
- `XXL Drum Samples/XXL FX & Rolls/` — 17 FX (reverse snare, perc rolls, bells)
- **Best for:** Large reggaeton drum palette, numbered samples for easy browsing, cowbell/timbal/shaker variety

**`reggaeton2/`** — 2609 files, superset of reggaeton sources
- Contains the same REGGAETON 4 content + Obie ALL GENRE KIT content
- **Use:** Same as REGGAETON 4 + Obie, just a different copy

---

### Trap / Hip-Hop

**`Obie - ALL GENRE KIT PT 2/`** — 2522 files, multi-genre
- `1. TRAP_NEWAGE_ETC/MODERN TRAP/` — trap 808s, hi-hats, kicks, snares
- `1. TRAP_NEWAGE_ETC/NEWAGE TRAP/` — newer trap sounds
- `1. TRAP_NEWAGE_ETC/JERSEY CLUB/` — Jersey club perc
- `1. TRAP_NEWAGE_ETC/NEWJAZZ/` — jazz-trap hybrid
- `R&B/90s_2000s R&B/`, `R&B/MODERN R&B/`
- `LOFI/` — lo-fi kicks, hats, snares, bass
- `DRILL/` — 808s, kicks, snares (UK Drill style)
- `DISCO_POP/` — claps, cymbals, hats, kicks, snares, toms
- `PUNK ROCK/` — distorted drums, cymbals, guitar loops
- **Best for:** Modern trap, R&B, drill, lo-fi, multi-genre production

**`VIRION - BLESSDEEKIT [JERK DRUMKIT]/`** — 65 files
- `808/`, `Kick/`, `Snare/`, `Hi-Hat/`, `Open Hat/`, `Crash/`, `Perc/`, `FX/`
- `MIDI Snare/` — MIDI snare patterns
- **Best for:** West Coast jerk beats (EsDeeKid style), ghost snares, bounce patterns

**`lex_luger_drum_kit/`** — 51 files
- `LEX Clap (1–8).wav`, `LEX Clap&Snare.wav`
- `LEX Hat (1–6).wav`, `LEX HEAVY Kick LV1.wav`, `LEX Drum (1–2).wav`
- `LEX Chant.wav`, `LEX Crash.wav`
- **Best for:** Hard trap, Waka Flocka/Rick Ross style big-room trap

**`Metro Boomin - #MetroWay Sound Kit/`** — 149 files
- `808s/`, `Kicks/`, `Snares/`, `Hats/`, `OpenHats/`, `Claps/`
- `Percs/`, `Chants + Vocals/Laughters/`, `FX/`, `Samples/`
- `Midi/` — MIDI patterns
- **Best for:** Metro Boomin-style trap, dark cinematic beats

**`☆ Juicy Jules - Stardust ☆/`** — 220 files
- `☆ 808s/` — named 808s: Big, Cash, Choppa, Dark, Distorted, Eisen, etc.
- `☆ Kicks/`, `☆ Snares/`, `☆ Closed Hats/`, `☆ Open Hats/`
- `☆ Claps/`, `☆ Percs/`, `☆ Crashes/`
- `☆ Loops/`, `☆ Risers/`, `☆ FX/`
- **Best for:** Modern trap/pop-rap 808s and drums

**`rap2/`** — 70 files, clean modern rap one-shots
- `808s/`, `Kicks/`, `Snares/`, `Claps/`, `HiHats/`, `Open Hats/`, `Percs_Rims/`
- `Melodies/` — melodic one-shots/loops included alongside drums
- Named with character: `(HH) - Insomnia.wav`, `(OH) - Tardy.wav`, `(HH) - BBQ.wav`
- **Best for:** Modern rap/trap drums with personality, good hat variety

**`rap3/`** — 152 files, Lil Baby song-sourced drum kit
- Organized by song: `Lil Baby All In/`, `Lil Baby Emotionally Scarred/`, `Lil Baby We Paid/`, etc. (26 songs)
- Each folder has stems: 808s, kicks, snares, hats, claps, FX, open hats
- Named descriptively: `Snare [Plugg].wav`, `Kick [Vybe].wav`, `Clap [Luger Slap].wav`
- **Best for:** Lil Baby / plugg / melodic trap style drums, authentic Atlanta sound

**`rap4/`** — 111 files, two sub-packs
- `lil drum kit/` — full drum kit: 808, clap, hats (closed/open/crash), kick, snare, perc, SFX, stomp, one-shots, drum loops
- `boof/` — miscellaneous
- `sum other loops i liked/` — melodic loops with BPM + key in filenames: `{SOSA} waters [162 BPM G# minor].wav`, `{SOSA} cassiopeia [165 BPM F# minor].wav`
- **Best for:** High-BPM drill/plugg drums, melodic loops already labeled by key and tempo

---

### House / Electronic

**`Wave Point - Sample Packs/`** — 1013 files
- `Wave Point - House Essentials Vol. 1/`
  - `DRUMS/DRUMS - LOOPS/`, `DRUMS/DRUMS - ONE SHOTS/`
  - `MUSIC/`, `VOCALS/`, `FX & FOLEY/`
- `Wave Point - House Essentials Vol. 2/`
  - `Wave Point - Drum Shots/`, `Wave Point - Foley/`
- **Best for:** House music drums, loops, music elements

**`FL_Studio/Packs/`** — 4933 files total (⚠️ many are `.wv` — convert first)
- `Drums/` — standard kick/snare/hat/cymbal/tom/SFX folders
- `Drums (ModeAudio)/` — ⚠️ all `.wv` WavPack format
  - Sub-packs: Downstream (hats), Firecracker (clap), HouseGen (shaker), Volt (clave)
  - Convert: `ffmpeg -y -i input.wv output.wav`
- `Instruments/` — ⚠️ OGG-in-WAV format (Rhodes, piano, etc.)
  - Convert: `ffmpeg -f ogg -i input.wav output.wav`
  - `Rhodes_1–4.wav` (converted, in `FL_Studio/converted/`)
- `Loops/`, `Risers/`, `SFX/`, `Shapes/`, `Vocals/`, `FLEX/`, `Legacy/`

**`FL_Studio/converted/`** — Ready-to-use converted samples:
```
CHat_Downstream07.wav   — closed hi-hat (crisp, modern)
OHat_Downstream05.wav   — open hi-hat
Clap_Firecracker.wav    — snappy clap (layerable on snare)
Clave_Volt03.wav        — clean clave accent
Shaker_HouseGen03.wav   — 8th-note shaker
Rhodes_1.wav            — FL Rhodes one-shot (~A3, 0.48s)
Rhodes_2.wav            — FL Rhodes one-shot (~0.50s)
Rhodes_3.wav            — FL Rhodes one-shot (~0.52s)
Rhodes_4.wav            — FL Rhodes one-shot (~0.55s) ← used in Sola
```
> Detect Rhodes root pitch via FFT before pitch-shifting:
> `_peak = freq[mask][argmax(fft[mask])]` → `MIDI = round(12*log2(peak/440)+69)`

---

### Synthwave / Retro Electronic

**`synthwaveSamples/`** — 412 files, vintage drum machine samples
- **Machines included:** E-mu Drumulator, Korg DDD-1, Linn Linn9000, Linn LinnDrum, Oberheim DMX, Roland R8, Roland TR-626, Roland TR-707, SCI DrumTraks, Simmons SDSV, Yamaha RX5
- Each machine has: `Kick/`, `Snare/`, `Hi Hat/`, `Cymbals/`, `Toms/`, `Percussion/`
- **Best for:** Synthwave, retro 80s, lo-fi electronic, chillwave

---

### Melodic / Instrumental One-Shots

**`flute_sound_kit/`** — 24 files, chromatic flute one-shots
- Two series: `FluteClean_` and `FluteClean2_` (slightly different tone/mic)
- Notes covered: C2, C3, C4, D#2, D#3, F#2, F#3, F#4, A#3, A#4
- Filenames encode pitch: `FluteClean2_C4.wav` = C4
- **Best for:** Pitch-shifting to any note (use closest sample to minimize artifacts), melodic top lines, lo-fi/chill flute textures
- **Loading tip:** Pick the sample nearest to your target pitch, then use `pb.PitchShift(semitones=...)` — max ±6 semitones for clean results

**`Piano Sample Pack/`** — 17 files, piano one-shots
- Files named `Piano Sample 3.wav` through `Piano Sample 18.wav` (no pitch info in name)
- One-shots of varying character — audition to find preferred tone
- **Best for:** Piano stabs, keys textures, layering under synth pads
- **Loading tip:** Use `detect_midi_root()` to find each sample's pitch before shifting

**`LSTBT_FSMD Free Samples Monthly Delivery Pack 7-2020/`** — 9 files, mixed melodic
- Key and BPM encoded in filenames: `LSTBT_Keys_85_Abm_21.wav`, `LSTBT_Tenor_Sax_90_Cm_153.wav`
- Contents: keys loop, tenor sax (trill + melodic), kalimba (with reverb FX), sliced flute loop, guitar loop, one-shots
- **Best for:** Lo-fi jazz-hop, chill hop, quick melodic accents with key already labeled

---

### Melodic / Loops / One-Shots

**`Care Package/`** — 100 files, melodic loops with metadata in filenames
- Format: `[EDptk] Name_keyAmin_143bpm.wav`
- Keys and BPMs encoded in filenames — easy to filter
- Contents: synth leads, guitar arps, Rhodes loops, drum rimshots, guitar loops
- Examples: `ADHD_keyAmin_143bpm`, `Celestial_keyF#min_120bpm`, `Escape_keyG#min_103bpm`
- Also includes `GUIT-STAB_Bmin.wav`, `Guitar_EchoStrings_keyAmin.wav`
- **Best for:** Ready-made melodic loops (check key + BPM in filename)

**`Free Melody Samples/`** — 20 files
- `Loop/Keys/`, `Loop/Multi_Instrument/`, `Loop/Synth/`
- `One_Shot/Keys/`
- **Best for:** Quick melodic loop references

**`NIKOS & GSTONE_SA - LET'S GO TO SPACE/`** — 180 files, full song pack
- `01. Acapella Vocals/`, `02. Instrumentals/`
- `03. Melody Loops/`, `04. Drum Loops/`, `05. Drumless Edits/`
- `06. Melody Midis/` — MIDI melody files
- `07. Drum Kit/` — 808s, claps, hats, kicks, open hats, perc, snares
- **Best for:** Space/atmospheric hip-hop, full stem references, MIDI melodies

**`Best Free Synths Samples/`** — 26 files
- `Loop/Bass/`, `Loop/Brass_Woodwind/`, `Loop/Synth/`
- `One_Shot/Bass/`
- **Best for:** Synth bass loops, vintage synth textures

**`Free 808 Samples/`** — 20 files
- `Loop/Bass/`, `One_Shot/Bass/`
- **Best for:** 808 sub bass one-shots and loops

**`Loop/`** — 20 files
- `Bass/`, `Brass_Woodwind/`, `Drums/`, `Synth/`
- **Best for:** Quick loop references

**`92elm - synplant sounds/`** — 27 files, experimental synth textures
- `nice and good synth.wav`, `seagull synth.wav`, `space blocks.wav`, `nice bell.wav`, `simple bass.wav`
- **Best for:** Atmospheric textures, sound design

**`Free Melody Samples/`**, **`synthwaveSamples/`** — see above sections

---

### Vocals

**`Vocalpack - free samples/`** — 41 files, Hiskee vocal chops organized by key + BPM
- `Am - 130 bpm/` — 19 vocal samples in A minor at 130 BPM
- `E - 110 bpm/` — 20 vocal samples in E at 110 BPM (includes 2 "Grain" variants)
- `F - 110 bpm/` — 2 vocal samples in F at 110 BPM
- Filename format: `{BPM}bpm {Key} - {NN} - Hiskee Vocalpack.wav`
- **Best for:** Vocal chops, vox textures, atmospheric vocal layers (key + BPM pre-labeled)

---

### Synth Presets (VST — not audio files)

**`Presets/`** — 30,477 files, massive Nexus 2 VST preset collection
- ⚠️ **These are `.fxp` synthesizer preset files, not audio samples** — cannot be loaded with `soundfile` or `load_sample()`
- Require the **Nexus 2 VST plugin** inside a DAW (FL Studio, Ableton, etc.)
- Genre coverage: Trap, Drill, Jersey Club, R&B, Lo-fi, House, EDM, Trance, Dubstep, Psytrance, Synthwave, Orchestral, Cinematic, and more
- Notable sub-collections: Metro Boomin XP, 808 Mafia, London On Da Track, ASAP Rocky XP, Chicago Drill Vols 1–3, Trap God 1–3, Juicy Jules, Wrecks On The Beat Vols 1–6
- Instrument types: leads, pads, arps, 808s, bass, plucks, bells, brass, strings, FX, sequences
- **Best for:** FL Studio / Nexus-based production workflows — not usable in Python render scripts

---

### Breakbeats

**`Breakbeats - Chopped to Tempo/`** — BPM-organized breakbeat chops
- BPMs: 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170
- Breaks per BPM: Amen, Baby Got Bach, Big Daddy, Kink, Skull Snaps, Stick, Think
- Each break has: `Akaized/` (whole loop) + `Chops/` (individual hits)
- **Best for:** Jungle, DnB, boom-bap, sampling-style beats

---

## Quick Lookup by Sample Type

| Need | Best Options |
|------|-------------|
| **Dembow kick** | `REGGAETON 4/Kicks/`, `reggaetondrums/Bombos/`, `reggaeton5/XXL Kicks/` |
| **Trap kick** | `Obie/MODERN TRAP/`, `Metro Boomin/Kicks/`, `Juicy Jules/☆ Kicks/`, `rap3/`, `rap4/lil drum kit/KICK/` |
| **Snare (reggaeton)** | `REGGAETON 4/Snare/`, `reggaetondrums/Redoblantes/`, `reggaeton5/XXL Snares/` |
| **Snare (trap)** | `Obie/MODERN TRAP/`, `VIRION/Snare/`, `Lex Luger/`, `rap2/Snares/`, `rap3/` |
| **Plugg/Atlanta trap drums** | `rap3/` (Lil Baby stems by song), `rap2/` |
| **Drill/high-BPM drums** | `rap4/lil drum kit/`, `Obie/DRILL/` |
| **Melodic loops (keyed)** | `rap4/sum other loops i liked/` (BPM+key in filename), `Care Package/` |
| **808 bass** | `Juicy Jules/☆ 808s/`, `Metro Boomin/808s/`, `Free 808 Samples/` |
| **Reese bass** | `REGGAETON 4/Reese Bass/Bass @el.obie 1.wav` |
| **Closed hi-hat** | `FL_Studio/converted/CHat_Downstream07.wav`, `URBANITO/Hats/` |
| **Open hi-hat** | `FL_Studio/converted/OHat_Downstream05.wav`, `URBANITO/Hats/Hat 21` |
| **Clap** | `FL_Studio/converted/Clap_Firecracker.wav`, `Metro Boomin/Claps/` |
| **Cencerro / bell** | `reggaetondrums/Percu/Bells & Guiros/NO ME CONOCE_CENCERRO_1.wav` |
| **Guiro** | `reggaetondrums/Percu/Bells & Guiros/NO ME CONOCE_GUIRO_1.wav` |
| **Timbales** | `reggaetondrums/Timbales/NO ME CONOCE_TIMBAL_1.wav` |
| **Bongos** | `reggaetondrums/Percu/Rims & Bongos/NO ME CONOCE_BONGO_1.wav` |
| **Palito** | `reggaetondrums/Percu/Palos & Toms/NO ME CONOCE_PALITO_1.wav` |
| **Clave** | `FL_Studio/converted/Clave_Volt03.wav`, `URBANITO/` |
| **Rhodes** | `FL_Studio/converted/Rhodes_1–4.wav` (⚠️ FFT pitch-detect before shifting) |
| **Vintage drum machine** | `synthwaveSamples/` (DMX, TR-707, LinnDrum, etc.) |
| **House drums** | `Wave Point/House Essentials/`, `FL_Studio/Packs/Drums (ModeAudio)/` |
| **Breakbeats** | `Breakbeats - Chopped to Tempo/{BPM}/` |
| **Melodic loops** | `Care Package/` (key+BPM in filename), `NIKOS/Melody Loops/` |
| **Guitar loops** | `Care Package/`, `reggaeton3/URBANO Guitar Loops/` |
| **Flute** | `flute_sound_kit/FluteClean2_C4.wav` etc. (pitch in filename, shift ±6st max) |
| **Piano** | `Piano Sample Pack/Piano Sample N.wav` (use FFT to detect root first) |
| **Kalimba** | `LSTBT_FSMD.../LSTBT_PE1_kalimba_130_Cm_56revefx.wav` |
| **Tenor sax** | `LSTBT_FSMD.../LSTBT_Tenor_Sax_90_Cm_153.wav` |
| **Vocal chops** | `Vocalpack/` (key+BPM labeled, Hiskee) |
| **Cowbell** | `reggaeton5/XXL Percussions/91_Cowbell.wav`, `reggaetondrums/Percu/Bells & Guiros/` |
| **Shaker** | `reggaeton5/XXL Percussions/` (multiple shakers), `FL_Studio/converted/Shaker_HouseGen03.wav` |
| **Synth presets (VST)** | `Presets/` — Nexus 2 `.fxp` files, DAW only, not Python-loadable |

---

## instruments_index.json

Run `python instruments_scan.py` to (re)build the index. Options:
```
python instruments_scan.py               # scan everything new
python instruments_scan.py --rescan      # re-analyze all files
python instruments_scan.py --missing     # only files not yet indexed
python instruments_scan.py --pack flute_sound_kit  # one pack only
```

The index has **15,443 entries** (5,973 drum one-shots · 5,432 melodic one-shots · 2,973 melodic loops · 933 drum loops). 132 files could not be read (proprietary or corrupted).

Each entry contains:
```json
{
  "pack": "flute_sound_kit",
  "format": "wav",
  "channels": 1,
  "sample_rate": 44100,
  "duration_s": 3.385,
  "peak": 0.3247,
  "rms_db": -15.0,
  "attack_ms": 238.7,
  "tail_silence_s": 0.0,
  "is_tonal": true,
  "flatness": 0.0002,
  "brightness_hz": 2288,
  "freq_band": "mid",
  "type": "melodic_loop",
  "root_midi": 60,
  "root_note": "C4",
  "root_hz": 261.6,
  "bpm": 129.2
}
```

Query example — find all kick one-shots with strong sub:
```python
import json
with open('instruments_index.json') as f:
    idx = json.load(f)

kicks = [
    (path, e) for path, e in idx.items()
    if e.get('type') == 'drum_oneshot'
    and e.get('freq_band') in ('sub', 'bass')
    and e.get('attack_ms', 999) < 15
]
kicks.sort(key=lambda x: x[1]['rms_db'], reverse=True)
```

---

## Loading Tips for Python

```python
import soundfile as sf
from math import gcd
from scipy import signal
import subprocess, os

def load_sample(path, target_sr=44100):
    """Load any standard WAV, with resampling."""
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        mono = signal.resample_poly(mono, target_sr // g, sr // g)
    return mono.astype('float32')

def convert_wv(wv_path, out_dir):
    """Convert WavPack .wv to WAV (ModeAudio FL Studio drums)."""
    out = os.path.join(out_dir, os.path.basename(wv_path).replace('.wv', '.wav'))
    subprocess.run(['ffmpeg', '-y', '-i', wv_path, '-ar', '44100', '-ac', '1', out], check=True)
    return out

def convert_ogg_in_wav(ogg_wav_path, out_dir):
    """Convert FL Studio OGG-in-WAV (Rhodes, piano keyboard samples)."""
    out = os.path.join(out_dir, os.path.basename(ogg_wav_path))
    subprocess.run(['ffmpeg', '-f', 'ogg', '-i', ogg_wav_path, out], check=True)
    return out

def detect_midi_root(sample, sr=44100):
    """FFT-based fundamental frequency → MIDI note number."""
    import numpy as np
    fft  = np.abs(np.fft.rfft(sample * np.hanning(len(sample))))
    freq = np.fft.rfftfreq(len(sample), 1 / sr)
    mask = (freq > 80) & (freq < 2000)
    peak = freq[mask][np.argmax(fft[mask])]
    return int(round(12 * np.log2(peak / 440) + 69))
```
