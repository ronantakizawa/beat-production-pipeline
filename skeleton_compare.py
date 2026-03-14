"""
skeleton_compare.py — Beat Similarity Scorer
=============================================
Uses Essentia's MusicExtractor to compute ~400 music-specific features
on two audio files, then produces a weighted similarity score across six
musically meaningful dimensions.

Usage:
    python skeleton_compare.py <our_beat.wav> <reference.wav>

Dimensions scored:
    Tempo / Rhythm   — BPM, beat strength, danceability, pulse clarity
    Key / Harmony    — key match, scale match, tonal strength, tuning
    Dynamics         — loudness, dynamic range, compression feel
    Spectral         — brightness, bass weight, spectral shape (MFCC/GFCC)
    Texture          — zero crossing rate, spectral flux, attack/release feel
    Groove           — beat histogram entropy, beat loudness variation
"""

import sys
import os
import numpy as np
from scipy.spatial.distance import cosine

try:
    import essentia.standard as es
except ImportError:
    sys.exit("essentia not installed — run: pip install essentia")


def extract(path: str) -> dict:
    """Run Essentia MusicExtractor on one file. Returns feature dict."""
    print(f"  Analysing: {os.path.basename(path)}")
    extractor = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev'],
    )
    features, _ = extractor(path)

    def g(key, default=0.0):
        try:
            v = features[key]
            return float(v) if not hasattr(v, '__len__') else list(v)
        except Exception:
            return default

    return {
        'bpm':                  g('rhythm.bpm'),
        'danceability':         g('rhythm.danceability'),
        'beats_loudness_mean':  g('rhythm.beats_loudness.mean'),
        'beats_loudness_std':   g('rhythm.beats_loudness.stdev'),
        'onset_rate':           g('rhythm.onset_rate'),
        'key':                  features['tonal.key_edma.key'] if 'tonal.key_edma.key' in features.descriptorNames() else '?',
        'scale':                features['tonal.key_edma.scale'] if 'tonal.key_edma.scale' in features.descriptorNames() else '?',
        'key_strength':         g('tonal.key_edma.strength'),
        'tuning_freq':          g('tonal.tuning_frequency'),
        'chords_strength_mean': g('tonal.chords_strength.mean'),
        'hpcp_mean':            g('tonal.hpcp.mean', [0]*36),
        'loudness_integrated':  g('lowlevel.loudness_ebu128.integrated'),
        'loudness_range':       g('lowlevel.loudness_ebu128.loudness_range'),
        'dynamic_complexity':   g('lowlevel.dynamic_complexity'),
        'spectral_centroid':    g('lowlevel.spectral_centroid.mean'),
        'spectral_rolloff':     g('lowlevel.spectral_rolloff.mean'),
        'spectral_flux':        g('lowlevel.spectral_flux.mean'),
        'spectral_energy':      g('lowlevel.spectral_energy.mean'),
        'mfcc_mean':            g('lowlevel.mfcc.mean', [0]*13),
        'gfcc_mean':            g('lowlevel.gfcc.mean', [0]*13),
        'zcr_mean':             g('lowlevel.zerocrossingrate.mean'),
        'spectral_complexity':  g('lowlevel.spectral_complexity.mean'),
        'pitch_salience':       g('lowlevel.pitch_salience.mean'),
        'bpm_histogram_first_peak_weight':  g('rhythm.bpm_histogram_first_peak_weight.mean'),
        'bpm_histogram_second_peak_weight': g('rhythm.bpm_histogram_second_peak_weight.mean'),
    }


def scalar_sim(a, b, scale):
    return float(np.exp(-0.5 * ((a - b) / scale) ** 2) * 100)

def vec_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0 or np.linalg.norm(a[:n]) == 0 or np.linalg.norm(b[:n]) == 0:
        return 0.0
    return (1 - cosine(a[:n], b[:n])) * 100

def key_sim(k1, s1, k2, s2):
    CIRCLE = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def idx(k):
        try: return CIRCLE.index(k)
        except ValueError: return -1
    i1, i2 = idx(k1), idx(k2)
    if i1 < 0 or i2 < 0: return 50.0
    semitones = min(abs(i1 - i2), 12 - abs(i1 - i2))
    scale_bonus = 10 if s1 == s2 else 0
    return max(0, 100 - semitones * 16 + scale_bonus)


def compare(path_ours: str, path_ref: str):
    print("\nExtracting features ...")
    f_ours = extract(path_ours)
    f_ref  = extract(path_ref)

    dims = {}

    bpm_s   = scalar_sim(f_ours['bpm'], f_ref['bpm'], scale=5)
    dance_s = scalar_sim(f_ours['danceability'], f_ref['danceability'], scale=0.15)
    bl_s    = scalar_sim(f_ours['beats_loudness_mean'], f_ref['beats_loudness_mean'], scale=0.05)
    onset_s = scalar_sim(f_ours['onset_rate'], f_ref['onset_rate'], scale=2)
    dims['Tempo / Rhythm'] = np.mean([bpm_s, dance_s, bl_s, onset_s])

    key_s   = key_sim(f_ours['key'], f_ours['scale'], f_ref['key'], f_ref['scale'])
    hpcp_s  = vec_sim(f_ours['hpcp_mean'], f_ref['hpcp_mean'])
    chord_s = scalar_sim(f_ours['chords_strength_mean'], f_ref['chords_strength_mean'], scale=0.1)
    dims['Key / Harmony'] = np.mean([key_s, hpcp_s, chord_s])

    loud_s  = scalar_sim(f_ours['loudness_integrated'], f_ref['loudness_integrated'], scale=3)
    range_s = scalar_sim(f_ours['loudness_range'], f_ref['loudness_range'], scale=3)
    dyn_s   = scalar_sim(f_ours['dynamic_complexity'], f_ref['dynamic_complexity'], scale=5)
    dims['Dynamics'] = np.mean([loud_s, range_s, dyn_s])

    mfcc_s    = vec_sim(f_ours['mfcc_mean'], f_ref['mfcc_mean'])
    gfcc_s    = vec_sim(f_ours['gfcc_mean'], f_ref['gfcc_mean'])
    cent_s    = scalar_sim(f_ours['spectral_centroid'], f_ref['spectral_centroid'], scale=500)
    rolloff_s = scalar_sim(f_ours['spectral_rolloff'], f_ref['spectral_rolloff'], scale=1000)
    dims['Spectral'] = np.mean([mfcc_s, gfcc_s, cent_s, rolloff_s])

    zcr_s  = scalar_sim(f_ours['zcr_mean'], f_ref['zcr_mean'], scale=0.01)
    flux_s = scalar_sim(f_ours['spectral_flux'], f_ref['spectral_flux'], scale=0.05)
    comp_s = scalar_sim(f_ours['spectral_complexity'], f_ref['spectral_complexity'], scale=5)
    dims['Texture'] = np.mean([zcr_s, flux_s, comp_s])

    bp1_s = scalar_sim(f_ours['bpm_histogram_first_peak_weight'],
                       f_ref['bpm_histogram_first_peak_weight'], scale=0.1)
    bp2_s = scalar_sim(f_ours['bpm_histogram_second_peak_weight'],
                       f_ref['bpm_histogram_second_peak_weight'], scale=0.1)
    dims['Groove'] = np.mean([bp1_s, bp2_s])

    # Dimension weights — adjust per genre if needed
    weights = {
        'Tempo / Rhythm': 0.25,
        'Key / Harmony':  0.20,
        'Dynamics':       0.15,
        'Spectral':       0.20,
        'Texture':        0.10,
        'Groove':         0.10,
    }
    overall = sum(dims[k] * weights[k] for k in dims)

    W = 58
    print("\n" + "=" * W)
    print(f"  SIMILARITY REPORT")
    print(f"  {os.path.basename(path_ours)}")
    print(f"  vs  {os.path.basename(path_ref)}")
    print("=" * W)
    print(f"\n  {'DIMENSION':<20} {'SCORE':>8}   BAR")
    print(f"  {'---'*7:<20}  {'---'*3:>8}   {'---'*7}")
    for dim, score in dims.items():
        bar = '#' * int(score / 5)
        print(f"  {dim:<20} {score:>7.1f}   {bar}")
    print(f"\n  {'---'*7:<20}  {'---'*3:>8}")
    print(f"  {'OVERALL':<20} {overall:>7.1f}/100")
    print("=" * W)

    print(f"\n-- RAW FEATURE COMPARISON --")
    print(f"  {'Feature':<32} {'Ours':>10}  {'Ref':>10}")
    print(f"  {'---'*11:<32}  {'---'*4:>10}  {'---'*4:>10}")
    for label, key in [
        ('BPM', 'bpm'), ('Danceability', 'danceability'),
        ('Beat loudness', 'beats_loudness_mean'), ('Onset rate', 'onset_rate'),
        ('Key', 'key'), ('Scale', 'scale'), ('Key strength', 'key_strength'),
        ('Tuning (Hz)', 'tuning_freq'), ('Loudness (LUFS)', 'loudness_integrated'),
        ('Loudness range', 'loudness_range'), ('Dynamic complexity', 'dynamic_complexity'),
        ('Spectral centroid', 'spectral_centroid'), ('Spectral rolloff', 'spectral_rolloff'),
        ('Spectral flux', 'spectral_flux'),
    ]:
        o = f_ours.get(key, '?')
        r = f_ref.get(key, '?')
        if isinstance(o, float): o = f'{o:.3f}'
        if isinstance(r, float): r = f'{r:.3f}'
        print(f"  {label:<32} {str(o):>10}  {str(r):>10}")
    print("=" * W)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
