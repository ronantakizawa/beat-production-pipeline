"""Render MyLove melody and pad MIDI through SF2 soundfonts."""
import os, sys, struct, subprocess
import numpy as np
import soundfile as sf
import mido

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/Documents/instruments")
OUT  = os.path.expanduser("~/Documents/MyLove_Beat/stems")
LEAD_MIDI = os.path.expanduser("~/Documents/MyLove_Beat/MyLove_lead.mid")
PAD_MIDI  = os.path.expanduser("~/Documents/MyLove_Beat/MyLove_pad.mid")
SR = 44100
BPM = 100

# ── SF2 configs: (path, bank, prog, label, use_for) ───────────────────
SF2S = [
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/Tubular Bells.sf2",
     0, 14, "tubularBells", "melody"),
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/Jose Guapo Bells.sf2",
     0, 0, "kalimbells", "melody"),
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/Karate Chop Sequence.sf2",
     0, 0, "saturnSiren", "melody"),
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/Future Speaks Brass.sf2",
     0, 61, "brass", "melody"),
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/Metro Choir.sf2",
     0, 52, "choir", "both"),
    (f"{BASE}/Metro Boomin - #MetroWay Sound Kit [Nexus XP]/Other Presets/19 & Boomin Zay Pad.sf2",
     0, 0, "zayPad", "pad"),
    (f"{BASE}/FL_Studio/Soundfonts/STR_Ensemble.sf2",
     0, 0, "strEnsemble", "both"),
]


def midi_to_events(midi_path):
    """Parse MIDI file into list of (time_seconds, type, note, vel)."""
    mid = mido.MidiFile(midi_path)
    events = []
    # Find the track with actual notes (skip metadata tracks)
    for track in mid.tracks:
        has_notes = any(m.type in ('note_on', 'note_off') for m in track)
        if not has_notes:
            continue
        abs_time = 0.0
        for msg in track:
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, mido.bpm2tempo(BPM))
            if msg.type == 'note_on':
                if msg.velocity > 0:
                    events.append((abs_time, 'on', msg.note, msg.velocity))
                else:
                    events.append((abs_time, 'off', msg.note, 0))
            elif msg.type == 'note_off':
                events.append((abs_time, 'off', msg.note, 0))
    events.sort(key=lambda e: e[0])
    return events


def extend_notes(events, sustain_add=0.2):
    """Delay note_off events to give instruments more sustain/release time."""
    # Pair note_on/off, extend off time, then flatten back
    on_times = {}   # note -> on_time
    pairs = []
    other = []
    for t, etype, note, vel in events:
        if etype == 'on':
            on_times[note] = (t, vel)
        elif etype == 'off' and note in on_times:
            on_t, on_vel = on_times.pop(note)
            new_off = t + sustain_add
            pairs.append((on_t, 'on', note, on_vel))
            pairs.append((new_off, 'off', note, 0))
    # Re-sort by time
    pairs.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
    return pairs


def apply_reverb(audio, decay=0.4, wet=0.3):
    """Simple convolution reverb using exponential decay IR."""
    ir_len = int(SR * decay)
    ir = np.exp(-np.linspace(0, 5, ir_len)).astype(np.float32)
    ir /= ir.sum()
    wet_l = np.convolve(audio[:, 0], ir, mode='full')[:len(audio)]
    wet_r = np.convolve(audio[:, 1], ir, mode='full')[:len(audio)]
    out = audio.copy()
    out[:, 0] = audio[:, 0] * (1 - wet) + wet_l * wet
    out[:, 1] = audio[:, 1] * (1 - wet) + wet_r * wet
    return out


def render_sf2(sf2_path, bank, prog, events, duration_sec):
    """Render MIDI events through an SF2 soundfont, return numpy audio array."""
    import fluidsynth

    # Extend note durations so they don't cut off abruptly
    events = extend_notes(events, sustain_add=0.25)

    fs = fluidsynth.Synth(samplerate=float(SR), gain=2.0)
    sfid = fs.sfload(sf2_path)
    if sfid < 0:
        print(f"  ERROR: Failed to load {os.path.basename(sf2_path)}")
        fs.delete()
        return None

    fs.program_select(0, sfid, bank, prog)
    # Enable built-in reverb and chorus
    fs.setting('synth.reverb.active', 1)
    fs.setting('synth.chorus.active', 1)

    total_samples = int(duration_sec * SR)
    chunk_size = 1024
    audio_chunks = []
    event_idx = 0
    sample_pos = 0

    while sample_pos < total_samples:
        current_time = sample_pos / SR

        # Trigger all events up to current time
        while event_idx < len(events) and events[event_idx][0] <= current_time:
            _, etype, note, vel = events[event_idx]
            if etype == 'on':
                fs.noteon(0, note, vel)
            else:
                fs.noteoff(0, note)
            event_idx += 1

        samples = fs.get_samples(chunk_size)
        audio_chunks.append(np.array(samples, dtype=np.int16))
        sample_pos += chunk_size

    fs.delete()

    raw = np.concatenate(audio_chunks)
    audio_f = raw.astype(np.float32) / 32768.0
    audio_f = audio_f.reshape(-1, 2)
    audio_f = audio_f[:total_samples]

    # Add reverb for smoothness
    audio_f = apply_reverb(audio_f, decay=0.6, wet=0.25)

    return audio_f


def export_mp3(audio, label, part):
    """Export float32 stereo array to MP3 via soundfile + lame."""
    wav_path = os.path.join(OUT, f"{part}_sf2_{label}.wav")
    out_path = os.path.join(OUT, f"{part}_sf2_{label}.mp3")
    sf.write(wav_path, audio, SR)
    subprocess.run(["/opt/homebrew/bin/lame", "-b", "320", "--silent",
                    wav_path, out_path], check=True)
    os.remove(wav_path)
    peak_db = 20 * np.log10(max(np.abs(audio).max(), 1e-10))
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(max(rms, 1e-10))
    print(f"  -> {os.path.basename(out_path)}  peak={peak_db:.1f}dB  rms={rms_db:.1f}dB")
    return out_path


def main():
    os.makedirs(OUT, exist_ok=True)

    # Parse MIDI files
    print("Parsing MIDI files...")
    lead_events = midi_to_events(LEAD_MIDI)
    pad_events = midi_to_events(PAD_MIDI)

    if not lead_events:
        print("WARNING: No note events found in lead MIDI")
    else:
        print(f"  Lead: {len(lead_events)} events, last at {lead_events[-1][0]:.1f}s")

    if not pad_events:
        print("WARNING: No note events found in pad MIDI")
    else:
        print(f"  Pad: {len(pad_events)} events, last at {pad_events[-1][0]:.1f}s")

    # Duration: 72 bars at 100 BPM = 72 * 4 * 60/100 = 172.8s + tail
    duration = 72 * 4 * 60.0 / BPM + 3.0  # extra 3s for release tails

    results = []

    for sf2_path, bank, prog, label, use_for in SF2S:
        if not os.path.exists(sf2_path):
            print(f"\nSKIP (missing): {label}")
            continue

        if use_for in ("melody", "both") and lead_events:
            print(f"\nRendering melody through {label} (bank={bank} prog={prog})...")
            audio = render_sf2(sf2_path, bank, prog, lead_events, duration)
            if audio is not None and np.abs(audio).max() > 0.001:
                path = export_mp3(audio, label, "melody")
                results.append(path)
            else:
                print(f"  SKIP: no audible output")

        if use_for in ("pad", "both") and pad_events:
            print(f"\nRendering pad through {label} (bank={bank} prog={prog})...")
            audio = render_sf2(sf2_path, bank, prog, pad_events, duration)
            if audio is not None and np.abs(audio).max() > 0.001:
                path = export_mp3(audio, label, "pad")
                results.append(path)
            else:
                print(f"  SKIP: no audible output")

    print(f"\n{'='*50}")
    print(f"Done! Rendered {len(results)} stems:")
    for r in results:
        print(f"  {os.path.basename(r)}")


if __name__ == "__main__":
    main()
