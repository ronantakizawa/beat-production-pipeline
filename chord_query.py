#!/usr/bin/env python3
"""
chord_query.py — Chord progression query tool for melody derivation.

Indexes all MIDI chord progressions in chordprogressions/ and provides
programmatic access to parsed chords, soprano lines, and bass lines.

Usage in a compose script:
    from chord_query import ChordBank
    bank = ChordBank()
    refs = bank.query(key='G', scale='Minor', n=3)
    soprano = refs[0].soprano_line   # [77, 74, 77, 79, ...]
    print(refs[0])                   # pretty-printed chords

Standalone CLI:
    python chord_query.py --key G --scale Minor --n 3
    python chord_query.py --key G --list-scales
    python chord_query.py --key G --scale Minor --soprano
    python chord_query.py --stats
"""

import os
import re
import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import mido

ROOT = Path(__file__).parent
CHORDS_DIR = ROOT / 'chordprogressions'

# ─── MIDI ↔ note name conversion ─────────────────────────────────────────────

_SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_NAMES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Keys that conventionally use flats
_FLAT_KEYS = {'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb',
              'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm'}


def midi_to_name(midi_num, prefer_flat=True):
    """Convert MIDI number to note name. E.g., 67 → 'G4', 70 → 'Bb4'."""
    octave = (midi_num // 12) - 1
    pc = midi_num % 12
    names = _FLAT_NAMES if prefer_flat else _SHARP_NAMES
    return f'{names[pc]}{octave}'


def name_to_midi(name):
    """Convert note name to MIDI number. E.g., 'G4' → 67, 'Bb4' → 70."""
    m = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', name)
    if not m:
        raise ValueError(f'Invalid note name: {name}')
    note_str = m.group(1)
    octave = int(m.group(2))
    note_str = note_str[0].upper() + note_str[1:]
    # Check both sharp and flat tables
    if note_str in _FLAT_NAMES:
        pc = _FLAT_NAMES.index(note_str)
    elif note_str in _SHARP_NAMES:
        pc = _SHARP_NAMES.index(note_str)
    else:
        raise ValueError(f'Unknown note: {note_str}')
    return (octave + 1) * 12 + pc


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Chord:
    notes: list          # MIDI numbers sorted low→high
    note_names: list     # e.g. ['G4', 'Bb4', 'D5']
    root: int            # lowest MIDI note
    top: int             # highest MIDI note (soprano)
    bar: int             # 0-indexed bar position

    def __repr__(self):
        return f'Chord(bar={self.bar}, {self.note_names})'


@dataclass
class Progression:
    file: str            # source MIDI path (relative to chordprogressions/)
    key: str             # e.g. 'G'
    scale: str           # e.g. 'Minor'
    collection: str      # e.g. 'GitHub Free Progressions'
    chords: list = field(default_factory=list)
    roman: str = ''      # roman numeral string from filename

    @property
    def soprano_line(self):
        """Top note of each chord — Step B of melody process."""
        return [c.top for c in self.chords]

    @property
    def soprano_names(self):
        """Soprano line as note names."""
        return [midi_to_name(c.top) for c in self.chords]

    @property
    def bass_line(self):
        """Lowest note of each chord."""
        return [c.root for c in self.chords]

    @property
    def bass_names(self):
        """Bass line as note names."""
        return [midi_to_name(c.root) for c in self.chords]

    def __repr__(self):
        header = f'{self.key} {self.scale} [{self.roman}]'
        lines = [header, f'  file: {self.file}']
        for c in self.chords:
            lines.append(f'  bar {c.bar}: {c.note_names}  (soprano={midi_to_name(c.top)})')
        lines.append(f'  soprano: {self.soprano_names}')
        lines.append(f'  bass:    {self.bass_names}')
        return '\n'.join(lines)


# ─── MIDI parsing ────────────────────────────────────────────────────────────

def parse_midi_chords(midi_path):
    """Parse a MIDI file and return a list of Chord objects."""
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    # Find the track with note data (usually track 1, but search all)
    note_track = None
    for track in mid.tracks:
        if any(msg.type == 'note_on' for msg in track):
            note_track = track
            break

    if note_track is None:
        return []

    # Group simultaneous note_on events by absolute tick
    abs_tick = 0
    active = {}       # tick → set of MIDI notes
    for msg in note_track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            if abs_tick not in active:
                active[abs_tick] = set()
            active[abs_tick].add(msg.note)

    # Convert to Chord objects, sorted by tick
    chords = []
    for i, tick in enumerate(sorted(active.keys())):
        notes = sorted(active[tick])
        if not notes:
            continue
        names = [midi_to_name(n) for n in notes]
        chords.append(Chord(
            notes=notes,
            note_names=names,
            root=notes[0],
            top=notes[-1],
            bar=i,
        ))

    return chords


def _extract_roman(filename):
    """Extract roman numeral progression from filename."""
    # Pattern: Key_Scale_...roman numerals..._Progression_N.mid
    # e.g. G_Minor_i_iv_VII_III_VI_iiĝ_V_i_Progression_1.mid
    m = re.search(r'_([iIvVx_ĝ]+(?:_[iIvVx_ĝ]+)*)_Progression', filename)
    if m:
        return m.group(1)
    # Fallback: strip key/scale prefix and progression suffix
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    # Remove first 2 (key, scale) and last 2 (Progression, N)
    if len(parts) > 4:
        return '_'.join(parts[2:-2])
    return ''


def _detect_key_from_notes(chords):
    """Detect key from chord notes using pitch class frequency."""
    if not chords:
        return 'C', 'Unknown'
    pc_count = [0] * 12
    for chord in chords:
        for note in chord.notes:
            pc_count[note % 12] += 1
    # Most common pitch class is likely the key
    root_pc = pc_count.index(max(pc_count))
    return _FLAT_NAMES[root_pc], 'Unknown'


# ─── ChordBank ───────────────────────────────────────────────────────────────

class ChordBank:
    """Query engine for chord progression MIDI files."""

    def __init__(self, root_dir=None):
        self._root = Path(root_dir) if root_dir else CHORDS_DIR
        self._catalog = []   # list of (key, scale, collection, rel_path)
        self._cache = {}     # rel_path → Progression
        self._scan()

    def _scan(self):
        """Scan the directory tree and build the catalog."""
        if not self._root.exists():
            raise FileNotFoundError(f'Chord progressions directory not found: {self._root}')

        # 1. GitHub Free Progressions: {KEY}/{KEY} {SCALE}/*.mid
        gfp = self._root / 'GitHub Free Progressions'
        if gfp.exists():
            for key_dir in sorted(gfp.iterdir()):
                if not key_dir.is_dir():
                    continue
                key_name = key_dir.name
                for scale_dir in sorted(key_dir.iterdir()):
                    if not scale_dir.is_dir():
                        continue
                    # Scale name = directory name minus the key prefix
                    scale_name = scale_dir.name.replace(f'{key_name} ', '', 1)
                    for f in sorted(scale_dir.glob('*.mid')):
                        rel = str(f.relative_to(self._root))
                        self._catalog.append((key_name, scale_name, 'GitHub Free Progressions', rel))

        # 2. New Free GitHub Chords: {KEY} Locrian/*.mid
        nfgc = self._root / 'New Free GitHub Chords'
        if nfgc.exists():
            for scale_dir in sorted(nfgc.iterdir()):
                if not scale_dir.is_dir():
                    continue
                # Parse "Ab Locrian" → key=Ab, scale=Locrian
                parts = scale_dir.name.rsplit(' ', 1)
                if len(parts) == 2:
                    key_name, scale_name = parts
                else:
                    key_name, scale_name = parts[0], 'Unknown'
                for f in sorted(scale_dir.glob('*.mid')):
                    rel = str(f.relative_to(self._root))
                    self._catalog.append((key_name, scale_name, 'New Free GitHub Chords', rel))

        # 3. EDM Progressions: *.mid (no key in path)
        edm = self._root / 'EDM Progressions'
        if edm.exists():
            for f in sorted(edm.glob('*.mid')):
                rel = str(f.relative_to(self._root))
                self._catalog.append(('*', 'EDM', 'EDM Progressions', rel))

        # 4. More Genres: *.mid
        mg = self._root / 'More Genres'
        if mg.exists():
            for f in sorted(mg.glob('*.mid')):
                rel = str(f.relative_to(self._root))
                # Try to extract genre from filename
                genre = f.stem.replace(' ', '_')
                self._catalog.append(('*', genre, 'More Genres', rel))

        # 5. Altered Dominant Chords: "{Key} Altered Dominant.mid"
        adc = self._root / 'Altered Dominant Chords'
        if adc.exists():
            for f in sorted(adc.glob('*.mid')):
                # "G Altered Dominant.mid" → key=G
                key_name = f.stem.split(' ')[0]
                rel = str(f.relative_to(self._root))
                self._catalog.append((key_name, 'Altered Dominant', 'Altered Dominant Chords', rel))

        # 6. Freemidis2025: *.mid
        fm = self._root / 'Freemidis2025'
        if fm.exists():
            for f in sorted(fm.glob('*.mid')):
                rel = str(f.relative_to(self._root))
                self._catalog.append(('*', 'Mixed', 'Freemidis2025', rel))

        # Also check Freemidis2025 2/
        fm2 = self._root / 'Freemidis2025 2'
        if fm2.exists():
            for f in sorted(fm2.glob('*.mid')):
                rel = str(f.relative_to(self._root))
                self._catalog.append(('*', 'Mixed', 'Freemidis2025', rel))

    def _load(self, key, scale, collection, rel_path):
        """Parse a MIDI file and return a Progression (cached)."""
        if rel_path in self._cache:
            return self._cache[rel_path]

        full_path = self._root / rel_path
        chords = parse_midi_chords(full_path)
        filename = os.path.basename(rel_path)
        roman = _extract_roman(filename)

        # For wildcard keys, detect from notes
        actual_key, actual_scale = key, scale
        if key == '*' and chords:
            actual_key, _ = _detect_key_from_notes(chords)

        prog = Progression(
            file=rel_path,
            key=actual_key,
            scale=actual_scale if scale != '*' else 'Unknown',
            collection=collection,
            chords=chords,
            roman=roman,
        )
        self._cache[rel_path] = prog
        return prog

    def query(self, key, scale='Minor', n=3, collection=None, seed=42):
        """
        Return n random progressions matching key/scale.

        Args:
            key:        Musical key, e.g. 'G', 'Ab', 'F#'
            scale:      Scale type, e.g. 'Minor', 'Major', 'Dorian', 'Sevenths'
            n:          Number of progressions to return
            collection: Filter by collection name (optional)
            seed:       Random seed for reproducibility
        """
        # Normalize key aliases
        key_norm = key.strip()

        matches = []
        for cat_key, cat_scale, cat_coll, rel_path in self._catalog:
            # Key match (case-insensitive, or wildcard)
            if cat_key != '*' and cat_key.lower() != key_norm.lower():
                continue
            # Scale match (case-insensitive, exact match)
            if cat_scale != '*' and cat_scale.lower() != scale.lower():
                continue
            # Collection filter
            if collection and collection.lower() not in cat_coll.lower():
                continue
            matches.append((cat_key, cat_scale, cat_coll, rel_path))

        if not matches:
            return []

        rng = random.Random(seed)
        selected = rng.sample(matches, min(n, len(matches)))

        results = []
        for cat_key, cat_scale, cat_coll, rel_path in selected:
            try:
                prog = self._load(cat_key, cat_scale, cat_coll, rel_path)
                if prog.chords:  # skip empty/broken files
                    results.append(prog)
            except Exception as e:
                print(f'  Warning: failed to parse {rel_path}: {e}')

        return results

    def soprano_pool(self, key, scale='Minor', n=3, seed=42):
        """
        Merged unique soprano notes from n progressions.
        Returns sorted list of MIDI numbers — ready for Step B.
        """
        refs = self.query(key, scale, n=n, seed=seed)
        pool = set()
        for prog in refs:
            pool.update(prog.soprano_line)
        return sorted(pool)

    def soprano_pool_named(self, key, scale='Minor', n=3, seed=42):
        """Soprano pool as note names."""
        return [midi_to_name(m) for m in self.soprano_pool(key, scale, n, seed)]

    def list_keys(self):
        """All available keys (excluding wildcard entries)."""
        keys = sorted(set(k for k, _, _, _ in self._catalog if k != '*'))
        return keys

    def list_scales(self, key):
        """Available scale types for a given key."""
        scales = sorted(set(
            s for k, s, _, _ in self._catalog
            if k.lower() == key.lower() and s != '*'
        ))
        return scales

    def list_collections(self):
        """All collection names."""
        return sorted(set(c for _, _, c, _ in self._catalog))

    def stats(self):
        """Count of files per collection."""
        counts = {}
        for _, _, coll, _ in self._catalog:
            counts[coll] = counts.get(coll, 0) + 1
        counts['total'] = len(self._catalog)
        return counts


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Query chord progression MIDI files for melody derivation')
    parser.add_argument('--key', help='Musical key, e.g. G, Ab, F#')
    parser.add_argument('--scale', default='Minor', help='Scale type (default: Minor)')
    parser.add_argument('--n', type=int, default=3, help='Number of progressions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--collection', help='Filter by collection name')
    parser.add_argument('--list-scales', action='store_true',
                        help='List available scales for --key')
    parser.add_argument('--list-keys', action='store_true',
                        help='List all available keys')
    parser.add_argument('--soprano', action='store_true',
                        help='Show merged soprano pool (Step B output)')
    parser.add_argument('--stats', action='store_true',
                        help='Show file counts per collection')
    args = parser.parse_args()

    bank = ChordBank()

    if args.stats:
        print('\nChord Progression Library Stats:\n')
        for coll, count in sorted(bank.stats().items()):
            if coll != 'total':
                print(f'  {coll:<30} {count:>5} files')
        print(f'  {"TOTAL":<30} {bank.stats()["total"]:>5} files')
        return

    if args.list_keys:
        print(f'\nAvailable keys: {", ".join(bank.list_keys())}')
        return

    if not args.key:
        parser.print_help()
        return

    if args.list_scales:
        scales = bank.list_scales(args.key)
        print(f'\nScales for {args.key}: {", ".join(scales)}')
        return

    if args.soprano:
        pool = bank.soprano_pool(args.key, args.scale, n=args.n, seed=args.seed)
        names = [midi_to_name(m) for m in pool]
        print(f'\nSoprano pool for {args.key} {args.scale} '
              f'(from {args.n} progressions):\n')
        print(f'  MIDI:  {pool}')
        print(f'  Names: {names}')
        return

    refs = bank.query(args.key, args.scale, n=args.n, seed=args.seed,
                      collection=args.collection)

    if not refs:
        print(f'\nNo progressions found for key={args.key} scale={args.scale}')
        return

    print(f'\n{len(refs)} progression(s) for {args.key} {args.scale}:\n')
    for prog in refs:
        print(prog)
        print()


if __name__ == '__main__':
    main()
