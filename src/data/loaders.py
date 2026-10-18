"""Loaders for the raw Steam datasets.

The Fronkon Games ``games.csv`` header declares 39 fields but every data row has
40: the header is missing a comma between ``Discount`` and ``DLC count``. Reading
it with a plain ``pd.read_csv`` silently shifts every column from index 7 onward
(``Tags`` ends up holding genre data). Always go through ``load_store_games``,
which skips the broken header and pins an explicit 40-column schema.
"""

from __future__ import annotations

import json
from typing import IO, Union

import pandas as pd

PathOrBuffer = Union[str, IO]

STORE_COLUMNS = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Required age", "Price", "Discount", "DLC count", "About the game",
    "Supported languages", "Full audio languages", "Reviews", "Header image",
    "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative",
    "Score rank", "Achievements", "Recommendations", "Notes",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks", "Developers",
    "Publishers", "Categories", "Genres", "Tags", "Screenshots", "Movies",
]
assert len(STORE_COLUMNS) == 40

# Only the columns the system actually uses are kept, with snake_case names.
STORE_RENAME = {
    "AppID": "app_id",
    "Name": "name",
    "Release date": "release_date",
    "Estimated owners": "estimated_owners",
    "Peak CCU": "peak_ccu",
    "Required age": "required_age",
    "Price": "price",
    "Discount": "discount",
    "DLC count": "dlc_count",
    "About the game": "description",
    "Supported languages": "supported_languages",
    "Windows": "windows",
    "Mac": "mac",
    "Linux": "linux",
    "Positive": "positive",
    "Negative": "negative",
    "Recommendations": "recommendations",
    "Average playtime forever": "avg_playtime_forever",
    "Median playtime forever": "median_playtime_forever",
    "Developers": "developers",
    "Publishers": "publishers",
    "Categories": "categories",
    "Genres": "genres",
    "Tags": "tags",
}


def _split_list(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def load_store_games(path: PathOrBuffer) -> pd.DataFrame:
    """Load Fronkon ``games.csv`` with the pinned 40-column schema."""
    df = pd.read_csv(
        path,
        # header=None,
        # skiprows=1,  # the malformed 39-field header
        # names=STORE_COLUMNS,
        usecols=list(STORE_RENAME),
        dtype={"About the game": "string", "Tags": "string", "Genres": "string"},
        keep_default_na=False,
        na_values={"Price": [""], "Required age": [""]},
    )
    df = df.rename(columns=STORE_RENAME)
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["app_id"])
    df["app_id"] = df["app_id"].astype("int64")
    for col in ("price", "discount"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ("required_age", "dlc_count", "peak_ccu", "positive", "negative",
                "recommendations", "avg_playtime_forever", "median_playtime_forever"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    for col in ("windows", "mac", "linux"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    for col in ("tags", "genres", "categories"):
        df[col] = df[col].map(_split_list)
    df["description"] = df["description"].fillna("").astype(str)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    return df.drop_duplicates("app_id").reset_index(drop=True)


def load_recs_games(path: PathOrBuffer) -> pd.DataFrame:
    """Load ``games.csv`` from the Game Recommendations on Steam dataset."""
    df = pd.read_csv(path, parse_dates=["date_release"])
    for col in ("win", "mac", "linux", "steam_deck"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def load_games_metadata(path: str) -> pd.DataFrame:
    """Load ``games_metadata.json`` (JSON lines: app_id, description, tags)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows, columns=["app_id", "description", "tags"])


def load_users(path: PathOrBuffer) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": "int64", "products": "int32", "reviews": "int32"})


RECS_DTYPES = {
    "app_id": "int32",
    "helpful": "int32",
    "funny": "int32",
    "hours": "float32",
    "user_id": "int64",
    "review_id": "int64",
}


def iter_recommendations(path: PathOrBuffer, chunksize: int = 2_000_000):
    """Stream ``recommendations.csv`` (41M rows) in chunks to bound memory."""
    for chunk in pd.read_csv(path, dtype=RECS_DTYPES, chunksize=chunksize):
        chunk["is_recommended"] = chunk["is_recommended"].astype(str).str.lower().eq("true")
        chunk["date"] = pd.to_datetime(chunk["date"])
        yield chunk
