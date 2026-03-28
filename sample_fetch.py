"""
sample_fetch.py -- Niche Sample Fetcher

Three sources for finding unique, untouched samples:

  1. youtube-world   -- Foreign-language music (Turkish psych, Ethiopian jazz,
                        Japanese city pop, Thai funk, Zamrock)
  2. musicradar      -- 223 free retro video game samples (WAV)
  3. forgotify       -- Zero/low-play Spotify tracks → download via YouTube

Usage:
  python sample_fetch.py --source youtube-world [--genre turkish_psych] [--count 3]
  python sample_fetch.py --source musicradar
  python sample_fetch.py --source forgotify --spotify-id <ID> --spotify-secret <SECRET> [--count 5]

All downloads go to: /Users/ronantakizawa/Documents/instruments/niche_samples/<source>/
"""

import argparse
import os
import re
import sys
import subprocess
import json
import zipfile
import random
import urllib.request
import shutil

OUTPUT_BASE = '/Users/ronantakizawa/Documents/instruments/niche_samples'


# ============================================================================
# 1. YOUTUBE WORLD MUSIC
# ============================================================================

# Curated search queries by genre — terms that surface actual music, not tutorials
WORLD_GENRES = {
    'turkish_psych': [
        'Barış Manço full album',
        'Erkin Koray 45lik',
        'Selda Bağcan şarkıları',
        'Anatolian rock 1970s',
        'Cem Karaca psych',
        'Moğollar psych rock',
        '3 Hürel garage rock',
    ],
    'ethiopian_jazz': [
        'Mulatu Astatke full album',
        'Éthiopiques compilation',
        'Hailu Mergia keyboard',
        'Mahmoud Ahmed songs',
        'Alemayehu Eshete 1960s',
        'Getatchew Mekurya saxophone',
        'Emahoy Tsegué-Maryam piano',
    ],
    'japanese_city_pop': [
        'シティポップ 80年代',
        'Tatsuro Yamashita full album',
        '竹内まりや アルバム',
        'Anri Timely album',
        '角松敏生 シティポップ',
        'Taeko Ohnuki sunshower',
        '大貫妙子 アルバム',
    ],
    'thai_funk': [
        'เพลงฟังค์ไทย 70s',
        'Thai funk soul 1970s vinyl',
        'Dao Bandon Thai',
        'Suraphol Sombatcharoen',
        'Thai molam funk',
        'Paradise Bangkok Molam',
        'Thai boogie funk compilation',
    ],
    'zamrock': [
        'WITCH Zamrock full album',
        'Amanaz Africa Zamrock',
        'Ngozi Family Zambia rock',
        'Paul Ngozi guitar',
        'Rikki Ililonga Zamrock',
        'Musi-O-Tunya Zambia',
        'Chrissy Zebby Tembo vultures',
    ],
    'bollywood_retro': [
        'R.D. Burman instrumental',
        'old Hindi film songs 1970s',
        'Laxmikant Pyarelal soundtrack',
        'Asha Bhosle rare songs',
        'Bollywood funk disco 1980s',
        'Kalyanji Anandji film music',
    ],
    'cumbia_psych': [
        'cumbia psicodélica peruana',
        'Los Destellos cumbia',
        'chicha peruana 1970s',
        'Juaneco y su Combo',
        'Los Mirlos selva',
        'cumbia amazónica vinyl',
    ],
    'afrobeat_highlife': [
        'Fela Kuti full album',
        'Tony Allen afrobeat drums',
        'Ebo Taylor highlife',
        'Pat Thomas highlife',
        'William Onyeabor synth',
        'Shina Williams afrobeat',
        'Geraldo Pino Afro soul',
    ],
    'korean_psychedelic': [
        'Shin Jung Hyun 신중현 full album',
        'San Ul Lim 산울림 1970s',
        'Kim Jung Mi 김정미',
        'He6 Korean rock',
        'Korean psychedelic rock 1970s vinyl',
        'Devil Kim Chi rock',
        'Hee Sisters 히 시스터즈',
    ],
    'soviet_synth': [
        'Eduard Artemyev soundtrack',
        'Zodiac Soviet synth pop',
        'Soyuz 77 USSR electronic',
        'Soviet electronic music 1980s',
        'Вячеслав Мещерин ансамбль',
        'Argo Soviet disco',
        'USSR synthesizer music vinyl',
    ],
    'persian_funk': [
        'Googoosh 1970s songs',
        'Iranian funk pop 1970s',
        'Viguen Persian pop vinyl',
        'Kourosh Yaghmaei Persian psych',
        'Dariush Eghbali songs',
        'pre-revolution Iranian disco',
        'Hayedeh Persian songs',
    ],
    'tropicalia': [
        'Os Mutantes full album',
        'Tom Zé Tropicália',
        'Gal Costa 1969',
        'Caetano Veloso Tropicália',
        'Gilberto Gil 1968',
        'Jorge Ben 1970s',
        'Novos Baianos acabou chorare',
    ],
    'desert_blues': [
        'Tinariwen full album',
        'Bombino guitar',
        'Ali Farka Touré songs',
        'Mdou Moctar',
        'Tamikrest desert rock',
        'Tuareg guitar music',
        'Group Inerane Niger',
    ],
    'chinese_jazz': [
        'Shanghai jazz 1930s',
        '上海老歌 jazz age',
        'Li Jinhui Chinese jazz',
        'Zhou Xuan 周璇 songs',
        'Bai Guang 白光',
        'Chinese jazz age vinyl',
        'Shanghai nightclub music 1940s',
    ],
    'balkan_brass': [
        'Boban Marković orkestar',
        'Fanfare Ciocărlia full album',
        'Kočani Orkestar',
        'Balkan brass band music',
        'Serbian brass wedding music',
        'Goran Bregović underground',
        'Romani brass orchestra',
    ],
    'rebetiko': [
        'Markos Vamvakaris songs',
        'rebetiko Greek blues 1930s',
        'Vassilis Tsitsanis bouzouki',
        'Sotiria Bellou rebetiko',
        'Marika Ninou songs',
        'rembetika compilation vinyl',
        'Greek underground music 1920s',
    ],
    'forro_baiao': [
        'Luiz Gonzaga baião',
        'Dominguinhos forró',
        'Jackson do Pandeiro',
        'Sivuca accordion',
        'Alceu Valença frevo',
        'forró pé de serra roots',
        'Trio Nordestino forró',
    ],
    'gamelan': [
        'Javanese gamelan full recording',
        'Balinese gamelan gong kebyar',
        'gamelan degung Sunda',
        'Surakarta court gamelan',
        'gamelan gender wayang',
        'Indonesian gamelan traditional',
        'Yogyakarta gamelan orchestra',
    ],
    'gnawa': [
        'Maalem Mahmoud Gania gnawa',
        'gnawa music Morocco',
        'Nass El Ghiwane Moroccan',
        'Hassan Hakmoun guembri',
        'gnawa lila ceremony',
        'Maâlem Mokhtar Gania',
        'Essaouira gnawa festival',
    ],
    'dub': [
        'King Tubby dub full album',
        'Lee Scratch Perry Black Ark',
        'Augustus Pablo melodica dub',
        'Scientist dub',
        'Prince Jammy dub',
        'Burning Spear dub versions',
        'The Upsetters dub',
    ],
    'kosmische': [
        'Tangerine Dream Phaedra',
        'Klaus Schulze album',
        'Ash Ra Tempel',
        'Manuel Göttsching E2-E4',
        'Popol Vuh soundtrack',
        'Cluster Zuckerzeit',
        'Harmonia Musik Von',
    ],
    'soukous': [
        'Franco TPOK Jazz full album',
        'Tabu Ley Rochereau',
        'Papa Wemba soukous',
        'Mbilia Bel songs',
        'Orchestra Baobab Senegal',
        'Zaiko Langa Langa',
        'Sam Mangwana soukous',
    ],
    'qawwali': [
        'Nusrat Fateh Ali Khan full qawwali',
        'Sabri Brothers qawwali',
        'Aziz Mian qawwali',
        'Abida Parveen sufi',
        'Rahat Fateh Ali Khan qawwali classical',
        'Pakistani qawwali traditional',
        'Amjad Sabri qawwali',
    ],
    'library_music': [
        'KPM library music 1970s',
        'De Wolfe library music',
        'Bruton Music library',
        'Alessandro Alessandroni library',
        'Piero Umiliani library music',
        'Sven Libaek library',
        'Alan Hawkshaw KPM',
    ],
    'chinese_opera': [
        '京剧 Peking opera classic',
        '越剧 Yue opera',
        '粤剧 Cantonese opera',
        'Mei Lanfang 梅兰芳',
        'Chinese opera traditional recording',
        '昆曲 Kunqu opera',
        '豫剧 Yu opera classic',
    ],
    'afro_cuban': [
        'Buena Vista Social Club full album',
        'Celia Cruz Fania years',
        'Arsenio Rodríguez son montuno',
        'Irakere Cuban jazz fusion',
        'Los Van Van timba',
        'Mongo Santamaria Afro Cuban jazz',
        'Machito Afro Cuban orchestra',
    ],
    'mbira_chimurenga': [
        'Thomas Mapfumo chimurenga',
        'Stella Chiweshe mbira',
        'Ephat Mujuru mbira',
        'Shona mbira music Zimbabwe',
        'Oliver Mtukudzi songs',
        'Dumisani Maraire marimba mbira',
        'Zimbabwe traditional mbira nhare',
    ],
}


def fetch_youtube_world(genre=None, count=3):
    """Search YouTube for foreign music and download audio clips."""
    out_dir = os.path.join(OUTPUT_BASE, 'youtube_world')
    os.makedirs(out_dir, exist_ok=True)

    if genre and genre in WORLD_GENRES:
        genres_to_search = {genre: WORLD_GENRES[genre]}
    elif genre:
        print(f'Unknown genre: {genre}')
        print(f'Available: {", ".join(WORLD_GENRES.keys())}')
        return []
    else:
        genres_to_search = WORLD_GENRES

    downloaded = []

    for genre_name, queries in genres_to_search.items():
        genre_dir = os.path.join(out_dir, genre_name)
        os.makedirs(genre_dir, exist_ok=True)

        # Pick random queries from the list
        picks = random.sample(queries, min(count, len(queries)))

        for query in picks:
            print(f'\n  [{genre_name}] Searching: {query}')

            # Use yt-dlp to search and download first result
            cmd = [
                'yt-dlp',
                f'ytsearch1:{query}',
                '-x', '--audio-format', 'wav',
                '--audio-quality', '0',
                '-o', os.path.join(genre_dir, '%(title).80s.%(ext)s'),
                '--no-playlist',
                '--max-filesize', '50M',
                '--socket-timeout', '30',
                '--quiet', '--no-warnings',
                '--print', 'after_move:filepath',
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and result.stdout.strip():
                    fpath = result.stdout.strip().split('\n')[-1]
                    if os.path.exists(fpath):
                        size_mb = os.path.getsize(fpath) / 1e6
                        print(f'    -> {os.path.basename(fpath)} ({size_mb:.1f} MB)')
                        downloaded.append(fpath)
                    else:
                        print(f'    -> File not found after download')
                else:
                    err = result.stderr.strip()[:200] if result.stderr else 'unknown error'
                    print(f'    -> Failed: {err}')
            except subprocess.TimeoutExpired:
                print(f'    -> Timeout')
            except Exception as e:
                print(f'    -> Error: {e}')

    print(f'\n  Total downloaded: {len(downloaded)} files')
    return downloaded


# ============================================================================
# 1b. YOUTUBE QUERY SEARCH (any search string)
# ============================================================================

def fetch_youtube_query(query, count=3, label=None):
    """Search YouTube with any query and download audio as WAV."""
    label = label or re.sub(r'[^\w]+', '_', query)[:40].strip('_')
    out_dir = os.path.join(OUTPUT_BASE, 'youtube_query', label)
    os.makedirs(out_dir, exist_ok=True)

    downloaded = []

    # Use ytsearchN to get N results from one search
    print(f'\n  Searching: "{query}" (count={count})')
    for i in range(count):
        search_term = f'ytsearch{count}:{query}'
        cmd = [
            'yt-dlp',
            search_term,
            '-x', '--audio-format', 'wav',
            '--audio-quality', '0',
            '-o', os.path.join(out_dir, '%(title).80s.%(ext)s'),
            '--no-playlist',
            '--max-filesize', '50M',
            '--socket-timeout', '30',
            '--quiet', '--no-warnings',
            '--print', 'after_move:filepath',
            '--playlist-items', str(i + 1),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and result.stdout.strip():
                fpath = result.stdout.strip().split('\n')[-1]
                if os.path.exists(fpath):
                    size_mb = os.path.getsize(fpath) / 1e6
                    print(f'    -> {os.path.basename(fpath)} ({size_mb:.1f} MB)')
                    downloaded.append(fpath)
                else:
                    print(f'    -> File not found after download')
            else:
                err = result.stderr.strip()[:200] if result.stderr else 'unknown error'
                print(f'    -> Failed: {err}')
        except subprocess.TimeoutExpired:
            print(f'    -> Timeout')
        except Exception as e:
            print(f'    -> Error: {e}')

    print(f'\n  Total downloaded: {len(downloaded)} files')
    return downloaded


# ============================================================================
# 2. MUSICRADAR RETRO VIDEO GAME SAMPLES
# ============================================================================

MUSICRADAR_URL = 'https://cdn.mos.musicradar.com/audio/samples/musicradar-retro-video-game-samples.zip'


def fetch_musicradar():
    """Download the MusicRadar retro video game sample pack."""
    out_dir = os.path.join(OUTPUT_BASE, 'musicradar_retro')
    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(out_dir, 'retro-video-game-samples.zip')

    # Check if already downloaded
    wav_count = len([f for f in os.listdir(out_dir) if f.endswith('.wav')])
    if wav_count > 50:
        print(f'  Already have {wav_count} WAV files in {out_dir}')
        return out_dir

    print(f'  Downloading from MusicRadar ...')
    print(f'  URL: {MUSICRADAR_URL}')

    try:
        req = urllib.request.Request(MUSICRADAR_URL, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            with open(zip_path, 'wb') as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f'\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.0f}%)', end='')
            print()

        size_mb = os.path.getsize(zip_path) / 1e6
        print(f'  Downloaded: {size_mb:.1f} MB')

        # Extract
        print(f'  Extracting ...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)

        os.remove(zip_path)

        # Count extracted files
        wav_files = []
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, f))

        print(f'  Extracted: {len(wav_files)} WAV files')
        # List categories
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        if subdirs:
            for sd in sorted(subdirs):
                n = len([f for f in os.listdir(os.path.join(out_dir, sd)) if f.endswith('.wav')])
                print(f'    {sd}: {n} samples')

        return out_dir

    except Exception as e:
        print(f'  Error: {e}')
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None


# ============================================================================
# 3. FORGOTIFY — Zero/Low-Play Spotify Tracks
# ============================================================================

# Obscure genre seeds — Spotify "micro-genres" that surface forgotten music
OBSCURE_GENRES = [
    'anatolian rock', 'ethiopian jazz', 'zamrock', 'thai funk',
    'japanese jazz fusion', 'afro psych', 'desert blues',
    'cumbia villera', 'bollywood retro', 'turkish psychedelic',
    'space age pop', 'exotica', 'library music', 'kosmische musik',
    'tropicália', 'ye ye', 'thai pop', 'chinese jazz',
    'persian pop', 'highlife', 'soukous', 'juju',
    'dub poetry', 'calypso', 'mento', 'rebetiko',
    'choro', 'forró', 'bossa nova obscure',
]

# Random search terms to find truly obscure tracks
OBSCURE_QUERIES = [
    'year:1965-1979 tag:new',
    'year:1970-1985',
    'year:1960-1975',
    'year:1975-1989',
    'year:1955-1970',
]


def fetch_forgotify(client_id, client_secret, count=5):
    """Find zero/low-play Spotify tracks and download via YouTube."""
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except ImportError:
        print('  Error: pip install spotipy')
        return []

    out_dir = os.path.join(OUTPUT_BASE, 'forgotify')
    os.makedirs(out_dir, exist_ok=True)

    print(f'  Connecting to Spotify ...')
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    ))

    found_tracks = []

    # Strategy: search obscure genres, filter by popularity=0
    genres_to_try = random.sample(OBSCURE_GENRES, min(count * 3, len(OBSCURE_GENRES)))

    for genre in genres_to_try:
        if len(found_tracks) >= count:
            break

        try:
            results = sp.search(q=f'genre:"{genre}"', type='track', limit=50,
                                market='US')
            tracks = results.get('tracks', {}).get('items', [])

            # Filter for zero or near-zero popularity
            zero_pop = [t for t in tracks if t['popularity'] <= 1]
            low_pop = [t for t in tracks if t['popularity'] <= 5]

            pool = zero_pop if zero_pop else low_pop
            if not pool:
                continue

            pick = random.choice(pool)
            artist = pick['artists'][0]['name'] if pick['artists'] else 'Unknown'
            title = pick['name']
            pop = pick['popularity']
            album = pick.get('album', {}).get('name', '')
            year = pick.get('album', {}).get('release_date', '')[:4]

            found_tracks.append({
                'artist': artist,
                'title': title,
                'album': album,
                'year': year,
                'popularity': pop,
                'genre_query': genre,
                'spotify_url': pick['external_urls'].get('spotify', ''),
            })
            print(f'  Found: {artist} - "{title}" ({year}) [pop={pop}, genre={genre}]')

        except Exception as e:
            continue

    if not found_tracks:
        # Fallback: random single-word searches with year filters
        print('  Trying random year-filtered searches ...')
        random_words = ['love', 'night', 'sun', 'rain', 'dance', 'dream',
                        'fire', 'moon', 'river', 'wind', 'soul', 'star']
        for _ in range(count * 5):
            if len(found_tracks) >= count:
                break
            word = random.choice(random_words)
            year_start = random.randint(1955, 1985)
            q = f'{word} year:{year_start}-{year_start + 5}'
            offset = random.randint(0, 950)
            try:
                results = sp.search(q=q, type='track', limit=50, offset=offset, market='US')
                tracks = results.get('tracks', {}).get('items', [])
                zero = [t for t in tracks if t['popularity'] <= 2]
                if zero:
                    pick = random.choice(zero)
                    artist = pick['artists'][0]['name'] if pick['artists'] else 'Unknown'
                    title = pick['name']
                    found_tracks.append({
                        'artist': artist,
                        'title': title,
                        'album': pick.get('album', {}).get('name', ''),
                        'year': pick.get('album', {}).get('release_date', '')[:4],
                        'popularity': pick['popularity'],
                        'genre_query': q,
                        'spotify_url': pick['external_urls'].get('spotify', ''),
                    })
                    print(f'  Found: {artist} - "{title}" ({found_tracks[-1]["year"]}) [pop={pick["popularity"]}]')
            except Exception:
                continue

    # Save manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(found_tracks, f, indent=2)
    print(f'\n  {len(found_tracks)} tracks found. Manifest: {manifest_path}')

    # Download via YouTube
    downloaded = []
    for track in found_tracks:
        query = f'{track["artist"]} {track["title"]}'
        print(f'\n  Downloading: {query}')

        cmd = [
            'yt-dlp',
            f'ytsearch1:{query}',
            '-x', '--audio-format', 'wav',
            '--audio-quality', '0',
            '-o', os.path.join(out_dir, '%(title).80s.%(ext)s'),
            '--no-playlist',
            '--max-filesize', '50M',
            '--socket-timeout', '30',
            '--quiet', '--no-warnings',
            '--print', 'after_move:filepath',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and result.stdout.strip():
                fpath = result.stdout.strip().split('\n')[-1]
                if os.path.exists(fpath):
                    size_mb = os.path.getsize(fpath) / 1e6
                    print(f'    -> {os.path.basename(fpath)} ({size_mb:.1f} MB)')
                    downloaded.append(fpath)
                else:
                    print(f'    -> File not found')
            else:
                print(f'    -> Not found on YouTube')
        except subprocess.TimeoutExpired:
            print(f'    -> Timeout')
        except Exception as e:
            print(f'    -> Error: {e}')

    print(f'\n  Downloaded: {len(downloaded)} / {len(found_tracks)} tracks')
    return downloaded


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample fetcher')
    parser.add_argument('--source',
                        choices=['youtube-world', 'musicradar', 'forgotify'],
                        help='Sample source (not needed with --query)')
    parser.add_argument('--query', type=str, default=None,
                        help='YouTube search query (e.g. "dark trap melody loop")')
    parser.add_argument('--label', type=str, default=None,
                        help='Folder name for query results (default: derived from query)')
    parser.add_argument('--genre', default=None,
                        help='Genre for youtube-world (e.g. turkish_psych, ethiopian_jazz)')
    parser.add_argument('--count', type=int, default=3,
                        help='Number of samples (default: 3)')
    parser.add_argument('--spotify-id', default=os.environ.get('SPOTIPY_CLIENT_ID'),
                        help='Spotify Client ID')
    parser.add_argument('--spotify-secret', default=os.environ.get('SPOTIPY_CLIENT_SECRET'),
                        help='Spotify Client Secret')
    parser.add_argument('--list-genres', action='store_true',
                        help='List available world music genres')
    args = parser.parse_args()

    if args.list_genres:
        print('Available world music genres:')
        for g in sorted(WORLD_GENRES.keys()):
            print(f'  {g} ({len(WORLD_GENRES[g])} search queries)')
        sys.exit(0)

    # --query mode: search YouTube with any query
    if args.query:
        print(f'\n=== sample_fetch.py — youtube query ===')
        print(f'Query: "{args.query}"')
        fetch_youtube_query(args.query, count=args.count, label=args.label)
        print('\nDone!')
        sys.exit(0)

    if not args.source:
        parser.error('--source or --query is required (unless using --list-genres)')

    print(f'\n=== sample_fetch.py — {args.source} ===')
    print(f'Output: {OUTPUT_BASE}/{args.source}/')

    if args.source == 'youtube-world':
        fetch_youtube_world(genre=args.genre, count=args.count)

    elif args.source == 'musicradar':
        fetch_musicradar()

    elif args.source == 'forgotify':
        if not args.spotify_id or not args.spotify_secret:
            print('\n  Spotify credentials required.')
            print('  Either set SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET env vars')
            print('  or pass --spotify-id and --spotify-secret')
            print('\n  Get free credentials at: https://developer.spotify.com/dashboard')
            sys.exit(1)
        fetch_forgotify(args.spotify_id, args.spotify_secret, count=args.count)

    print('\nDone!')
