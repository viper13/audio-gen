"""Curated style chips and ready-to-use song presets for YuE2 Studio."""
from __future__ import annotations

STYLE_TAG_GROUPS = [
    {
        "category": "Genre",
        "tags": [
            "Indie Pop",
            "Synthwave",
            "City Pop",
            "Acoustic Folk",
            "Jazz Ballad",
            "Lo-Fi Hip Hop",
            "Alternative Rock",
            "Heavy Metal",
            "R&B Soul",
            "Electronic Dance",
            "Cinematic Orchestral",
            "Ukrainian Folk Rock",
        ],
    },
    {
        "category": "Vocals",
        "tags": [
            "expressive female vocal",
            "warm male vocal",
            "intimate breathy vocal",
            "soulful powerful vocal",
            "duet vocals",
            "choir harmonies",
            "raspy rock vocal",
        ],
    },
    {
        "category": "Mood",
        "tags": [
            "melancholic",
            "uplifting",
            "dreamy atmospheric",
            "energetic driving",
            "nostalgic",
            "romantic warm",
            "epic dramatic",
            "chill relaxing",
        ],
    },
    {
        "category": "Instruments & Tempo",
        "tags": [
            "acoustic piano",
            "fingerpicked acoustic guitar",
            "analog synths",
            "warm bassline",
            "punchy drums",
            "lush strings",
            "muted trumpet",
            "distorted electric guitars",
            "85 BPM",
            "110 BPM",
            "128 BPM",
        ],
    },
]

SONG_PRESETS = [
    {
        "id": "kyiv-midnight-synth",
        "name": "Київська Ніч (Ukrainian Indie Synthpop)",
        "badge": "UA · Synthpop",
        "title": "Вогні Над Дніпром",
        "style": "Ukrainian, atmospheric indie synthpop, expressive female vocal, warm analog synthesizers, shimmering electric guitar, pulsing bassline, crisp electronic drums, nostalgic dreamy mood, 108 BPM",
        "lyrics": (
            "[Verse 1]\n"
            "Вечірні вогні відбиває Дніпро\n"
            "Холодний вітер торкається вікон\n"
            "Ми йдемо крізь тишу нічного метро\n"
            "І місто співає нам тихим потоком\n\n"
            "[Chorus]\n"
            "Тримай мене поруч у сяйві вогнів\n"
            "Де небо торкається дахів і снів\n"
            "Хай музика лине крізь ніч до зорі\n"
            "Ми вільні сьогодні, ми вічно живі"
        ),
        "cot": "melody",
        "ode_steps": 8,
        "precision": "8bit",
        "max_tokens": 1200,
    },
    {
        "id": "neon-rain-piano-pop",
        "name": "Neon Rain (Warm Piano Pop · Quickstart)",
        "badge": "EN · Piano Pop",
        "title": "Neon Fades Along The Lane",
        "style": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
        "lyrics": (
            "[Verse]\n"
            "Neon fades along the lane\n"
            "Footsteps keep the time of rain\n"
            "Fold the night and leave it here\n"
            "Morning has a sky to clear\n\n"
            "[Chorus]\n"
            "Let the day come into view\n"
            "Every road begins with you\n"
            "Hold a little room for light\n"
            "We will sing beyond the night"
        ),
        "cot": "full",
        "ode_steps": 8,
        "precision": "8bit",
        "max_tokens": 900,
    },
    {
        "id": "midnight-city-pop",
        "name": "Midnight Avenue (80s City Pop)",
        "badge": "EN · City Pop",
        "title": "Under The City Glow",
        "style": "city pop, 80s groovy slap bass, bright brass stabs, Rhodes electric piano, crisp snare drums, smooth female vocal, upbeat nostalgic mood, 118 BPM",
        "lyrics": (
            "[Verse]\n"
            "Neon lights reflect in the rainy avenue\n"
            "Whispering shadows dancing through the midnight blue\n"
            "Headlights gliding softly on the wet asphalt\n"
            "Time slows down and never comes to a halt\n\n"
            "[Chorus]\n"
            "Stay with me tonight under the city glow\n"
            "Let the rhythm guide us where the night winds blow\n"
            "Turn the radio up till the morning comes\n"
            "Hearts beating steady to the midnight drums"
        ),
        "cot": "melody",
        "ode_steps": 8,
        "precision": "8bit",
        "max_tokens": 1200,
    },
    {
        "id": "velvet-jazz-ballad",
        "name": "Velvet Coffeehouse (Intimate Jazz)",
        "badge": "EN · Jazz",
        "title": "Late Night Espresso",
        "style": "Jazz ballad, brushed drums, walking upright bass, warm grand piano, muted trumpet, intimate male vocal, smoky lounge atmosphere, 68 BPM",
        "lyrics": (
            "[Verse]\n"
            "Steam rises slowly from a porcelain cup\n"
            "The band plays soft while the street lights flicker up\n"
            "A silver trumpet sighs across the wooden floor\n"
            "Every note feels like we've met here once before\n\n"
            "[Chorus]\n"
            "Hold the chord and let the quiet linger on\n"
            "In this velvet corner till the dark is gone"
        ),
        "cot": "full",
        "ode_steps": 8,
        "precision": "8bit",
        "max_tokens": 1000,
    },
    {
        "id": "lofi-study-instrumental",
        "name": "Rainy Tape (Lo-Fi Instrumental)",
        "badge": "Instrumental · Lo-Fi",
        "title": "Cassette Rain",
        "style": "Instrumental, lo-fi hip hop, warm dusty Rhodes piano, soft vinyl crackle, gentle jazz guitar chords, boom bap mellow drums, deep sub bass, relaxing nocturnal atmosphere, 76 BPM, no vocals, no singing.",
        "lyrics": "[Intro]\n\n[Instrumental Verse]\n\n[Instrumental Chorus]\n\n[Outro]",
        "cot": "melody",
        "instrumental": True,
        "ode_steps": 8,
        "precision": "8bit",
        "max_tokens": 900,
    },
]
