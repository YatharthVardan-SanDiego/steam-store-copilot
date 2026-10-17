"""Cleaning, filtering, labeling and splitting for the Tag QA Gate.

Every rule here is documented in the README "Preparation rules" section and is
covered by ``tests/test_preprocess.py``.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

MIN_DESCRIPTION_CHARS = 50
MIN_LATIN_SHARE = 0.5
MIN_GAMES_PER_TAG = 10_000

# Games released on/after this date are held back as the "incoming drafts"
# stream used for batch inference and drift monitoring; they are never trained on.
INCOMING_CUTOFF = pd.Timestamp("2025-01-01")

# The 22 storefront tags with >= 10,000 games each (computed in
# notebooks/01_data_exploration.ipynb; fixed for v1 so the label space is stable).
TAGS = [
    "Singleplayer", "Indie", "Action", "Casual", "Adventure", "2D", "3D",
    "Simulation", "Strategy", "Puzzle", "Atmospheric", "RPG", "Colorful",
    "Pixel Graphics", "Story Rich", "Exploration", "Cute", "First-Person",
    "Arcade", "Fantasy", "Early Access", "Multiplayer",
]

# Columns available when a studio submits a draft (no post-release signal).
DRAFT_NUMERIC = ["price", "required_age", "desc_chars", "desc_words"]
DRAFT_BOOL = ["windows", "mac", "linux"]

_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]+(?:=[^\]]*)?\]")
_WS = re.compile(r"\s+")


def strip_bbcode(text: str) -> str:
    return _WS.sub(" ", _BBCODE.sub(" ", text)).strip()


def latin_share(text: str) -> float:
    """Share of alphabetic characters that belong to the Latin script blocks."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(ord(c) < 0x250 for c in letters) / len(letters)


def tag_label_vector(tags: list[str], tag_space: list[str] = TAGS) -> list[int]:
    present = set(tags)
    return [int(t in present) for t in tag_space]


def split_for(app_id: int, release_date) -> str:
    """Deterministic split: incoming (by date) else 80/10/10 by app_id hash."""
    if pd.notna(release_date) and release_date >= INCOMING_CUTOFF:
        return "incoming"
    bucket = int(hashlib.md5(str(app_id).encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def build_tag_dataset(games: pd.DataFrame) -> pd.DataFrame:
    """Raw store games -> one cleaned row per game with draft features + labels."""
    df = games.copy()
    df["description"] = df["description"].map(strip_bbcode)
    df = df[df["description"].str.len() >= MIN_DESCRIPTION_CHARS]
    df = df[df["tags"].map(len) > 0]
    df = df[df["description"].map(latin_share) >= MIN_LATIN_SHARE]

    df["desc_chars"] = df["description"].str.len()
    df["desc_words"] = df["description"].str.split().map(len)
    df["tags_22"] = df["tags"].map(lambda ts: [t for t in TAGS if t in set(ts)])
    df["n_tags_22"] = df["tags_22"].map(len)
    df["developers"] = df["developers"].fillna("").astype(str)
    df["publishers"] = df["publishers"].fillna("").astype(str)
    df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)
    df["split"] = [split_for(a, d) for a, d in zip(df["app_id"], df["release_date"])]

    cols = (["app_id", "name", "description"] + DRAFT_NUMERIC + DRAFT_BOOL
            + ["developers", "publishers", "release_date", "release_year",
               "tags_22", "n_tags_22", "split"])
    return df[cols].reset_index(drop=True)


def labels_matrix(tag_lists, tag_space: list[str] = TAGS) -> np.ndarray:
    return np.array([tag_label_vector(t, tag_space) for t in tag_lists], dtype=np.int8)


def encode_tags(tags: list[str]) -> str:
    """Pipe-joined tag list (Feature Store and CSV friendly)."""
    return "|".join(tags)


def decode_tags(value) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return value.split("|")


def make_mismatched_drafts(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Pair each description with *another* game's tags.

    These simulate drafts that do not describe the game they were written for;
    the gate must reject them. Used to measure the false-approval rate.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    # fix any fixed points so no row keeps its own tags
    fixed = np.where(perm == np.arange(n))[0]
    if len(fixed):
        perm[fixed] = np.roll(perm[fixed], 1) if len(fixed) > 1 else (fixed + 1) % n
    out = df.copy().reset_index(drop=True)
    out["tags_22"] = df["tags_22"].reset_index(drop=True).iloc[perm].values
    out["is_mismatch"] = 1
    return out
