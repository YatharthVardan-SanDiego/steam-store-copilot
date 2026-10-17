"""Reception Estimator: review-level features and chronological split.

Label: ``negative`` = 1 when the review does *not* recommend the game (14% of
reviews; the class the content team cares about). Game-level aggregates that
encode reception itself (positive ratio, rating, positive/negative counts) are
deliberately excluded; they would leak the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import iter_recommendations
from src.data.preprocess import TAGS

TAG_FEATURES = [f"tag_{t.lower().replace(' ', '_').replace('-', '_')}" for t in TAGS]
FEATURES = (["hours", "user_products", "user_reviews", "price_final", "discount", "win", "mac",
             "linux", "steam_deck", "days_since_release", "review_year", "desc_chars",
             "required_age"] + TAG_FEATURES)
LABEL = "negative"

VAL_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2022-07-01")


def sample_reviews(path, frac: float = 0.05, seed: int = 0, chunksize: int = 4_000_000):
    """Uniform sample of the 41M reviews, read in chunks to bound memory."""
    parts = [chunk.sample(frac=frac, random_state=seed)
             for chunk in iter_recommendations(path, chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True)


def build_reception_frame(reviews: pd.DataFrame, users: pd.DataFrame,
                          recs_games: pd.DataFrame, tag_dataset: pd.DataFrame) -> pd.DataFrame:
    df = reviews.merge(users.rename(columns={"products": "user_products",
                                             "reviews": "user_reviews"}),
                       on="user_id", how="left")
    games = recs_games[["app_id", "price_final", "discount", "win", "mac", "linux",
                        "steam_deck", "date_release"]]
    df = df.merge(games, on="app_id", how="left")

    draft = tag_dataset[["app_id", "desc_chars", "required_age", "tags_22"]].copy()
    tag_sets = draft["tags_22"].map(set)
    for tag, col in zip(TAGS, TAG_FEATURES):
        draft[col] = tag_sets.map(lambda s, t=tag: int(t in s))
    df = df.merge(draft.drop(columns="tags_22"), on="app_id", how="left")

    df["days_since_release"] = (df["date"] - df["date_release"]).dt.days.clip(lower=0)
    df["review_year"] = df["date"].dt.year
    for col in ("win", "mac", "linux", "steam_deck"):
        df[col] = df[col].astype(float)
    df[LABEL] = (~df["is_recommended"]).astype(int)
    df[FEATURES] = df[FEATURES].astype(float).fillna(0)
    return df[["review_id", "app_id", "date", LABEL] + FEATURES]


def chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["date"] < VAL_START],
        "val": df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)],
        "test": df[df["date"] >= TEST_START],
    }


def to_xgboost_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Built-in XGBoost CSV format: label first, no header."""
    return df[[LABEL] + FEATURES]


def scale_pos_weight(y: np.ndarray) -> float:
    pos = max(int(np.sum(y)), 1)
    return float((len(y) - pos) / pos)
