# emoji_backend.py
# Simple emoji -> playlist CSV mapping and helper to return dataframe
import pandas as pd
from pathlib import Path

# Map simple emoji (or names) to the CSV files you already have under songs/
EMOJI_MAP = {
    "😡": "songs/angry.csv",
    "🤢": "songs/disgusted.csv",
    "😨": "songs/fearful.csv",
    "😄": "songs/happy.csv",
    "😐": "songs/neutral.csv",
    "😢": "songs/sad.csv",
    "😲": "songs/surprised.csv",
    # alternative text keys (in case UI sends words)
    "angry": "songs/angry.csv",
    "disgusted": "songs/disgusted.csv",
    "fearful": "songs/fearful.csv",
    "happy": "songs/happy.csv",
    "neutral": "songs/neutral.csv",
    "sad": "songs/sad.csv",
    "surprised": "songs/surprised.csv",
}

DEFAULT_CSV = "songs/neutral.csv"

def emoji_to_df(emoji_or_name, top_n=15):
    """
    Return a pandas DataFrame of top_n songs for the provided emoji_or_name.
    Falls back to DEFAULT_CSV if mapping missing or file unavailable.
    """
    path = EMOJI_MAP.get(emoji_or_name, EMOJI_MAP.get(emoji_or_name.strip().lower(), DEFAULT_CSV))
    p = Path(path)
    if not p.exists():
        # fallback to default
        p = Path(DEFAULT_CSV)
    try:
        df = pd.read_csv(p)
    except Exception:
        # if CSV malformed or missing, return empty dataframe with columns used by app
        df = pd.DataFrame(columns=["Name", "Album", "Artist"])
    return df.head(top_n)
