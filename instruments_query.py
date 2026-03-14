#!/usr/bin/env python3
"""
instruments_query.py — Sample selection from instruments_index.json

Replaces hardcoded sample paths in render scripts with index-driven selection.
Tracks usage across beats so the same samples don't repeat every song.

Usage in a render script:
    from instruments_query import SampleSelector
    sel = SampleSelector(genre='reggaeton', beat='MySong_v1', key='C#')
    KICK    = sel.pick('kick')
    SNARE   = sel.pick('snare')
    HH_CL   = sel.pick('hat_closed')
    BASS    = sel.pick('bass')
    MELODY  = sel.pick('melody')
    sel.save()   # write usage to log so next beat avoids these

    # Auto pitch/gain info available after pick():
    sel.info['kick']   # {'pitch_st': 0, 'gain_db': -2.3, 'rms_db': -10, ...}

Standalone query:
    python instruments_query.py --role kick --genre reggaeton
    python instruments_query.py --role melody --genre trap --key "C#" --scale minor
    python instruments_query.py --history
    python instruments_query.py --dedup-stats
"""

import os
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).parent
INDEX_PATH = ROOT / 'instruments_index.json'
USAGE_PATH = ROOT / 'usage_log.json'

# ─── Target RMS per role (dB) — for auto gain normalization ──────────────────
# Samples louder/quieter than target get a correction factor.
TARGET_RMS_DB = {
    'kick':       -10,
    'snare':      -14,
    'hat_closed': -20,
    'hat_open':   -22,
    'clap':       -16,
    'perc':       -18,
    'bass':       -10,
    '808':        -8,
    'melody':     -16,
    'pad':        -18,
    'arp':        -18,
}

# ─── Key → MIDI note mapping (for auto pitch-shift calculation) ──────────────
_KEY_TO_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def pitch_offset_st(sample_key, target_key):
    """Semitones to shift sample_key → target_key (shortest distance, ±6 max)."""
    if not sample_key or not target_key:
        return 0
    s = _KEY_TO_PITCH_CLASS.get(sample_key)
    t = _KEY_TO_PITCH_CLASS.get(target_key)
    if s is None or t is None:
        return 0
    diff = (t - s) % 12
    return diff if diff <= 6 else diff - 12

# ─── Genre presets ────────────────────────────────────────────────────────────
# Each role defines hard filters (must match) and soft scoring hints.
# Filters: type, freq_band, max_attack_ms, max_duration_s, min_duration_s,
#          is_tonal, clipping, min_crest, min_sub_pct

GENRE_PRESETS = {
    'reggaeton': {
        'kick': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['kick', 'kck', ' bd ', '_bd_', 'bd.', '(bd)', 'bassdrum', 'bass drum'],
            'max_attack_ms': 25,
            'min_duration_s': 0.15,
            'max_duration_s': 2.0,
            'min_sub_pct': 2,
            'clipping': False,
            'score_by': 'sub_energy_pct',
        },
        'snare': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['snare', 'snr', ' sn ', '_sn_', 'sn.', 'snr.'],
            'max_attack_ms': 35,
            'max_duration_s': 2.0,
            'clipping': False,
            'score_by': 'crest_factor',
        },
        'hat_closed': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['hh', 'hat', 'hihat', 'hi-hat', 'hi hat', 'closed'],
            'name_excludes': ['open', 'ohh', 'ohat', 'oh '],
            'max_duration_s': 0.8,
            'score_by': 'brightness_hz',
        },
        'hat_open': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['open', 'ohh', 'ohat', 'open hat', 'open hi', 'oh '],
            'min_duration_s': 0.15,
            'score_by': 'brightness_hz',
        },
        'clap': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['clap', 'clp'],
            'max_duration_s': 1.5,
            'score_by': 'crest_factor',
        },
        'perc': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['perc', 'cencerro', 'cowbell', 'guiro', 'shaker', 'tamb', 'rim', 'clave'],
            'max_duration_s': 1.2,
            'score_by': 'brightness_hz',
        },
        'bass': {
            'type': 'melodic_oneshot',
            'name_contains': ['bass', '808', 'sub', 'low', 'reese'],
            'is_tonal': True,
            'score_by': 'sub_energy_pct',
        },
        'melody': {
            'type': ['melodic_oneshot', 'melodic_loop'],
            'name_contains': ['melody', 'melo', 'lead', 'guitar', 'piano', 'flute',
                              'bell', 'keys', 'synth', 'pluck', 'loop', 'sample'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'clap', '808', 'perc'],
            'is_tonal': True,
            'max_rms_db': -8,
            'max_duration_s': 30.0,
            'max_sub_pct': 20,
            'score_by': 'key_confidence',
        },
        'pad': {
            'type': ['melodic_loop', 'melodic_oneshot'],
            'name_contains': ['pad', 'synth', 'atmosphere', 'ambient', 'string',
                              'warm', 'choir', 'texture'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'top', 'perc'],
            'is_tonal': True,
            'min_duration_s': 1.5,
            'max_rms_db': -10,
            'score_by': 'key_confidence',
        },
        'arp': {
            'type': ['melodic_oneshot', 'melodic_loop'],
            'name_contains': ['arp', 'pluck', 'bell', 'sequence', 'stab',
                              'key', 'piano', 'flute'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', '808', 'bass'],
            'is_tonal': True,
            'max_rms_db': -10,
            'score_by': 'pitch_stability_cents',
            'score_ascending': True,
        },
    },

    'trap': {
        'kick': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['kick', 'kck', ' bd ', '_bd_', 'bassdrum'],
            'max_attack_ms': 20,
            'min_duration_s': 0.15,
            'max_duration_s': 2.0,
            'min_sub_pct': 2,
            'clipping': False,
            'score_by': 'sub_energy_pct',
        },
        'snare': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['snare', 'snr', ' sn ', '_sn_'],
            'max_attack_ms': 25,
            'min_crest': 4.0,
            'score_by': 'crest_factor',
        },
        'hat_closed': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['hh', 'hat', 'hihat', 'hi-hat'],
            'name_excludes': ['open', 'ohh'],
            'max_duration_s': 0.4,
            'score_by': 'brightness_hz',
        },
        'hat_open': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['open', 'ohh', 'open hat', 'open hi'],
            'min_duration_s': 0.2,
            'max_duration_s': 2.0,
            'score_by': 'brightness_hz',
        },
        '808': {
            'type': 'melodic_oneshot',
            'name_contains': ['808', 'sub', 'bass', 'low'],
            'is_tonal': True,
            'min_sub_pct': 10,
            'score_by': 'sub_energy_pct',
        },
        'melody': {
            'type': ['melodic_oneshot', 'melodic_loop'],
            'name_contains': ['melody', 'melo', 'lead', 'guitar', 'piano', 'flute',
                              'bell', 'keys', 'synth', 'pluck', 'loop', 'sample'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'clap', '808', 'perc'],
            'is_tonal': True,
            'max_rms_db': -8,
            'max_duration_s': 30.0,
            'max_sub_pct': 20,
            'score_by': 'key_confidence',
        },
        'perc': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['perc', 'cencerro', 'cowbell', 'guiro', 'shaker',
                              'tamb', 'rim', 'clave', 'block', 'conga', 'bongo'],
            'max_duration_s': 0.8,
            'score_by': 'crest_factor',
        },
    },

    'jerk': {
        'kick': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['kick', 'kck', ' bd ', '_bd_', 'bassdrum'],
            'max_attack_ms': 15,
            'min_duration_s': 0.15,
            'max_duration_s': 2.0,
            'min_sub_pct': 2,
            'score_by': 'sub_energy_pct',
        },
        'snare': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['snare', 'snr', ' sn ', '_sn_'],
            'max_attack_ms': 20,
            'score_by': 'crest_factor',
        },
        'hat_closed': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['hh', 'hat', 'hihat', 'hi-hat'],
            'name_excludes': ['open', 'ohh'],
            'max_duration_s': 0.35,
            'score_by': 'brightness_hz',
        },
        '808': {
            'type': 'melodic_oneshot',
            'name_contains': ['808', 'sub', 'bass', 'low'],
            'is_tonal': True,
            'score_by': 'sub_energy_pct',
        },
        'melody': {
            'type': ['melodic_oneshot', 'melodic_loop'],
            'name_contains': ['melody', 'melo', 'lead', 'guitar', 'piano', 'flute',
                              'bell', 'keys', 'synth', 'pluck', 'loop', 'sample'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'clap', '808', 'perc'],
            'is_tonal': True,
            'max_rms_db': -8,
            'max_duration_s': 30.0,
            'max_sub_pct': 20,
            'score_by': 'key_confidence',
        },
    },

    'house': {
        'kick': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['kick', 'kck', ' bd ', '_bd_', 'bassdrum'],
            'max_attack_ms': 15,
            'min_duration_s': 0.15,
            'max_duration_s': 2.0,
            'min_sub_pct': 2,
            'max_decay_ms': 400,
            'clipping': False,
            'score_by': 'sub_energy_pct',
        },
        'snare': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['snare', 'snr', ' sn ', '_sn_'],
            'max_attack_ms': 30,
            'score_by': 'crest_factor',
        },
        'hat_closed': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['hh', 'hat', 'hihat', 'hi-hat'],
            'name_excludes': ['open', 'ohh'],
            'max_duration_s': 0.6,
            'score_by': 'brightness_hz',
        },
        'hat_open': {
            'type': ['drum_oneshot', 'melodic_oneshot'],
            'name_contains': ['open', 'ohh', 'open hat', 'open hi'],
            'min_duration_s': 0.25,
            'score_by': 'brightness_hz',
        },
        'bass': {
            'type': 'melodic_oneshot',
            'name_contains': ['bass', '808', 'sub', 'low', 'reese'],
            'is_tonal': True,
            'score_by': 'sub_energy_pct',
        },
        'pad': {
            'type': ['melodic_loop', 'melodic_oneshot'],
            'name_contains': ['pad', 'synth', 'atmosphere', 'ambient', 'string',
                              'warm', 'choir', 'texture'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'top', 'perc'],
            'is_tonal': True,
            'min_duration_s': 2.0,
            'max_rms_db': -10,
            'score_by': 'loop_clean',
        },
        'melody': {
            'type': ['melodic_oneshot', 'melodic_loop'],
            'name_contains': ['melody', 'melo', 'lead', 'guitar', 'piano', 'flute',
                              'bell', 'keys', 'synth', 'pluck', 'loop', 'sample'],
            'name_excludes': ['drum', 'kick', 'snare', 'hat', 'clap', '808', 'perc'],
            'is_tonal': True,
            'max_rms_db': -8,
            'max_duration_s': 30.0,
            'max_sub_pct': 20,
            'score_by': 'key_confidence',
        },
    },
}


# ─── Index loading ─────────────────────────────────────────────────────────────

def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f'Index not found: {INDEX_PATH}\nRun: python instruments_scan.py')
    with open(INDEX_PATH) as f:
        idx = json.load(f)
    # Strip meta key
    return {k: v for k, v in idx.items()
            if not k.startswith('_') and isinstance(v, dict) and 'type' in v}


def load_usage():
    if not USAGE_PATH.exists():
        return []
    with open(USAGE_PATH) as f:
        return json.load(f)


def save_usage(log):
    with open(USAGE_PATH, 'w') as f:
        json.dump(log, f, indent=2)


# ─── Filtering ────────────────────────────────────────────────────────────────

def matches(entry, filters, path=''):
    """Return True if an index entry satisfies all filter criteria."""
    # name_contains: list of strings — at least one must appear in the filename (case-insensitive)
    if 'name_contains' in filters:
        fname = path.lower()
        keywords = [k.lower() for k in filters['name_contains']]
        if not any(kw in fname for kw in keywords):
            return False

    # name_excludes: list of strings — none may appear in the filename
    if 'name_excludes' in filters:
        fname = path.lower()
        for kw in filters['name_excludes']:
            if kw.lower() in fname:
                return False

    # type (string or list)
    if 'type' in filters:
        allowed = filters['type'] if isinstance(filters['type'], list) else [filters['type']]
        if entry.get('type') not in allowed:
            return False

    # freq_band (string or list)
    if 'freq_band' in filters:
        allowed = filters['freq_band'] if isinstance(filters['freq_band'], list) else [filters['freq_band']]
        if entry.get('freq_band') not in allowed:
            return False

    if 'is_tonal' in filters and entry.get('is_tonal') != filters['is_tonal']:
        return False

    if 'clipping' in filters and filters['clipping'] is False and entry.get('clipping', False):
        return False

    if 'max_attack_ms' in filters:
        if entry.get('attack_ms', 9999) > filters['max_attack_ms']:
            return False

    if 'max_duration_s' in filters:
        if entry.get('trimmed_duration_s', entry.get('duration_s', 9999)) > filters['max_duration_s']:
            return False

    if 'min_duration_s' in filters:
        if entry.get('trimmed_duration_s', entry.get('duration_s', 0)) < filters['min_duration_s']:
            return False

    if 'max_rms_db' in filters:
        if entry.get('rms_db', 0) > filters['max_rms_db']:
            return False

    if 'min_crest' in filters:
        if entry.get('crest_factor', 0) < filters['min_crest']:
            return False

    if 'min_sub_pct' in filters:
        if entry.get('sub_energy_pct', 0) < filters['min_sub_pct']:
            return False

    if 'max_sub_pct' in filters:
        if entry.get('sub_energy_pct', 0) > filters['max_sub_pct']:
            return False

    if 'max_decay_ms' in filters:
        if entry.get('decay_ms', 9999) > filters['max_decay_ms']:
            return False

    if 'max_brightness_hz' in filters:
        if entry.get('brightness_hz', 9999) > filters['max_brightness_hz']:
            return False

    if 'min_brightness_hz' in filters:
        if entry.get('brightness_hz', 0) < filters['min_brightness_hz']:
            return False

    # Key / scale filters (optional, passed from caller)
    if 'key' in filters and filters['key'] is not None:
        if entry.get('key') != filters['key']:
            return False

    if 'scale' in filters and filters['scale'] is not None:
        if entry.get('scale') != filters['scale']:
            return False

    return True


def score_entry(entry, score_by, ascending=False):
    """Return a numeric score for ranking candidates."""
    val = entry.get(score_by, 0) or 0
    return -val if ascending else val


def recently_used_paths(usage_log, n_beats=4):
    """Paths used in the last n_beats entries."""
    recent = usage_log[-n_beats:] if len(usage_log) >= n_beats else usage_log
    used = set()
    for beat in recent:
        used.update(beat.get('samples', {}).values())
    return used


def recently_used_packs(usage_log, n_beats=4):
    """Pack names used in the last n_beats entries, with counts."""
    recent = usage_log[-n_beats:] if len(usage_log) >= n_beats else usage_log
    pack_counts = {}
    for beat in recent:
        for path in beat.get('samples', {}).values():
            pack = path.split('/')[0] if '/' in path else ''
            pack_counts[pack] = pack_counts.get(pack, 0) + 1
    return pack_counts


# ─── Core query ───────────────────────────────────────────────────────────────

def query(role, genre='reggaeton', key=None, scale=None,
          extra_filters=None, n=10, exclude_paths=None, exclude_packs=None):
    """
    Find the best candidates for a given role and genre.
    Returns list of (path, entry) tuples, best first.

    Args:
        role:           'kick', 'snare', 'melody', 'bass', etc.
        genre:          genre preset name
        key:            filter by musical key, e.g. 'C#'
        scale:          filter by scale, e.g. 'natural_minor'
        extra_filters:  dict of additional filter overrides
        n:              max number of candidates to return
        exclude_paths:  set of paths to exclude (recently used)
        exclude_packs:  dict of {pack: count} to deprioritize
    """
    idx = load_index()

    preset = GENRE_PRESETS.get(genre, {})
    if role not in preset:
        raise ValueError(f"Role '{role}' not defined for genre '{genre}'.\n"
                         f"Available roles: {list(preset.keys())}")

    filters = dict(preset[role])
    score_by     = filters.pop('score_by', 'rms_db')
    score_asc    = filters.pop('score_ascending', False)

    if key:   filters['key']   = key
    if scale: filters['scale'] = scale
    if extra_filters:
        filters.update(extra_filters)

    exclude_paths = exclude_paths or set()
    exclude_packs = exclude_packs or {}

    candidates = []
    for path, entry in idx.items():
        if path in exclude_paths:
            continue
        if not matches(entry, filters, path=path):
            continue
        candidates.append((path, entry))

    if not candidates:
        return []

    # Score: primary = role-specific metric, penalty = overused pack
    def rank(item):
        path, entry = item
        base = score_entry(entry, score_by, score_asc)
        pack = path.split('/')[0] if '/' in path else ''
        pack_penalty = exclude_packs.get(pack, 0) * 0.5
        return base - pack_penalty

    candidates.sort(key=rank, reverse=True)
    return candidates[:n]


# ─── SampleSelector (for use inside render scripts) ───────────────────────────

class SampleSelector:
    """
    Drop-in replacement for hardcoded sample paths in render scripts.

        sel = SampleSelector(genre='reggaeton', beat='Sola_v12', key='C#')
        KICK  = sel.pick('kick')
        SNARE = sel.pick('snare', avoid_pack='REGGAETON 4')
        sel.save()

        # Auto pitch/gain metadata available after pick():
        sel.info['kick']['pitch_st']   # semitones to shift this sample to match key
        sel.info['kick']['gain_db']    # dB correction to hit target loudness for role

    After calling save(), the chosen paths are logged so the next beat
    automatically avoids them, enforcing variety across songs.
    """

    def __init__(self, genre='reggaeton', beat=None, seed=None,
                 key=None, scale=None, avoid_recent_beats=4):
        self.genre   = genre
        self.beat    = beat or f'beat_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.key     = key
        self.scale   = scale
        self.rng     = random.Random(seed)
        self.chosen  = {}   # role → relative path
        self.info    = {}   # role → {path, pitch_st, gain_db, rms_db, root_midi, ...}

        self.usage_log    = load_usage()
        self.used_paths   = recently_used_paths(self.usage_log, avoid_recent_beats)
        self.used_packs   = recently_used_packs(self.usage_log, avoid_recent_beats)

    def pick(self, role, n_candidates=8, avoid_pack=None, extra_filters=None,
             verbose=True):
        """
        Select the best available sample for a role, avoiding recently used paths/packs.
        Returns an absolute path string ready to pass to load_sample().

        Also populates self.info[role] with:
          pitch_st  — semitones to shift sample key → song key (0 if non-tonal/unknown)
          gain_db   — dB correction to normalize RMS to role target (0 if unknown)
          rms_db    — sample's raw RMS
          root_midi — detected MIDI root note (None if percussive)
          bpm       — detected BPM (None if one-shot)
          key       — detected key
          scale     — detected scale
        """
        ep = set(self.used_paths)
        xp = dict(self.used_packs)
        if avoid_pack:
            xp[avoid_pack] = xp.get(avoid_pack, 0) + 99

        candidates = query(
            role, genre=self.genre,
            key=self.key, scale=self.scale,
            extra_filters=extra_filters,
            n=n_candidates,
            exclude_paths=ep,
            exclude_packs=xp,
        )

        if not candidates:
            candidates = query(role, genre=self.genre, key=self.key,
                               scale=self.scale, n=n_candidates)

        if not candidates:
            raise ValueError(f"No samples found for role='{role}' genre='{self.genre}'")

        pool = candidates[:max(1, len(candidates) // 2)]
        path, entry = self.rng.choice(pool)

        abs_path = str(ROOT / path)
        self.chosen[role] = path

        # ── Auto pitch-shift calculation ──
        p_st = 0
        sample_key = entry.get('key')
        if self.key and sample_key and entry.get('is_tonal'):
            p_st = pitch_offset_st(sample_key, self.key)

        # ── Auto gain normalization ──
        g_db = 0.0
        sample_rms = entry.get('rms_db')
        target_rms = TARGET_RMS_DB.get(role)
        if sample_rms is not None and target_rms is not None:
            g_db = round(target_rms - sample_rms, 1)
            # Clamp to ±18 dB to avoid extreme corrections
            g_db = max(-18.0, min(18.0, g_db))

        self.info[role] = {
            'path':      abs_path,
            'pitch_st':  p_st,
            'gain_db':   g_db,
            'rms_db':    sample_rms,
            'root_midi': entry.get('root_midi'),
            'bpm':       entry.get('bpm'),
            'key':       sample_key,
            'scale':     entry.get('scale'),
        }

        if verbose:
            key_info  = f"  key={sample_key}/{entry.get('scale','—')}" if sample_key else ''
            adj_info  = ''
            if p_st != 0:
                adj_info += f'  pitch={p_st:+d}st'
            if abs(g_db) > 0.5:
                adj_info += f'  gain={g_db:+.1f}dB'
            print(f"  [{role:<12}] {path}{key_info}{adj_info}")

        return abs_path

    def save(self):
        """Write this beat's sample choices to the usage log."""
        self.usage_log.append({
            'beat':    self.beat,
            'genre':   self.genre,
            'date':    datetime.now().isoformat(),
            'samples': self.chosen,
        })
        save_usage(self.usage_log)
        print(f"\n  Usage logged → {USAGE_PATH.name}  ({len(self.usage_log)} beats total)")

    def report(self):
        """Print a summary of what was selected."""
        print(f"\nSample selection for '{self.beat}' ({self.genre}):")
        for role, path in self.chosen.items():
            print(f"  {role:<14} {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Query the instruments index')
    parser.add_argument('--role',    help='Sample role (kick, snare, melody, etc.)')
    parser.add_argument('--genre',   default='reggaeton', help='Genre preset')
    parser.add_argument('--key',     help='Filter by musical key, e.g. C#')
    parser.add_argument('--scale',   help='Filter by scale, e.g. natural_minor')
    parser.add_argument('--n',       type=int, default=10, help='Number of results')
    parser.add_argument('--history', action='store_true', help='Show usage history')
    parser.add_argument('--roles',   action='store_true', help='List roles for a genre')
    args = parser.parse_args()

    if args.history:
        log = load_usage()
        if not log:
            print('No usage history yet.')
            return
        print(f'Usage history ({len(log)} beats):\n')
        for beat in log[-10:]:
            print(f"  {beat['date'][:10]}  {beat['beat']}  ({beat['genre']})")
            for role, path in beat.get('samples', {}).items():
                print(f"    {role:<14} {path}")
        return

    if args.roles:
        preset = GENRE_PRESETS.get(args.genre, {})
        print(f"Roles for genre '{args.genre}': {', '.join(preset.keys())}")
        return

    if not args.role:
        parser.print_help()
        return

    log       = load_usage()
    used      = recently_used_paths(log)
    used_pack = recently_used_packs(log)

    results = query(args.role, genre=args.genre, key=args.key, scale=args.scale,
                    n=args.n, exclude_paths=used, exclude_packs=used_pack)

    if not results:
        print(f"No results for role='{args.role}' genre='{args.genre}'")
        return

    print(f"\nTop {len(results)} candidates — role='{args.role}' genre='{args.genre}'"
          + (f" key={args.key}" if args.key else '')
          + (f" scale={args.scale}" if args.scale else '') + '\n')

    for i, (path, e) in enumerate(results, 1):
        key_str  = f"{e.get('key','')}/{e.get('scale','')}" if e.get('key') else '—'
        dur      = e.get('trimmed_duration_s', e.get('duration_s', 0))
        crest    = e.get('crest_factor', 0)
        rms      = e.get('rms_db', 0)
        sub      = e.get('sub_energy_pct', 0)
        recently = '(used recently)' if path in used else ''
        print(f"  {i:>2}. {path}")
        print(f"      pack={e.get('pack','')}  dur={dur:.2f}s  rms={rms:.1f}dB  "
              f"crest={crest:.1f}  sub={sub:.0f}%  key={key_str}  {recently}")


if __name__ == '__main__':
    main()
