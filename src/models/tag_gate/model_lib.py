"""Tag QA Gate model: featurization, per-tag thresholds and the gate decision.

This module ships inside the SageMaker model artifact (``source_dir``), so it
only depends on numpy/pandas/scikit-learn and is importable as ``model_lib``.
Only built-in scikit-learn estimators are used so the pickled pipeline loads in
any container with the same scikit-learn version (1.2.x).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TAGS = [
    "Singleplayer", "Indie", "Action", "Casual", "Adventure", "2D", "3D",
    "Simulation", "Strategy", "Puzzle", "Atmospheric", "RPG", "Colorful",
    "Pixel Graphics", "Story Rich", "Exploration", "Cute", "First-Person",
    "Arcade", "Fantasy", "Early Access", "Multiplayer",
]
NUMERIC = ["price", "required_age", "desc_chars", "desc_words"]
BOOL = ["windows", "mac", "linux"]
FEATURE_COLUMNS = ["description"] + NUMERIC + BOOL


def build_pipeline(kind: str, C: float = 2.0, max_features: int = 200_000) -> Pipeline:
    """``baseline``: word TF-IDF + logistic regression at the default 0.5 cut-off.

    ``improved``: word TF-IDF plus draft-time tabular features and class-balanced
    logistic regression; per-tag thresholds are tuned on validation afterwards.
    (Character n-grams were tried and dropped: ~20x slower to train for no lift.)
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=max_features,
        sublinear_tf=True, strip_accents="unicode", stop_words="english",
        dtype=np.float32,
    )
    if kind == "baseline":
        features = ColumnTransformer([("word", word, "description")])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
    elif kind == "improved":
        numeric = Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
        ])
        features = ColumnTransformer([
            ("word", word, "description"),
            ("num", numeric, NUMERIC),
            ("bool", "passthrough", BOOL),
        ])
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear",
                                 class_weight="balanced")
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("features", features), ("clf", OneVsRestClassifier(clf, n_jobs=-1))])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame (training CSV or inference JSON) into model input."""
    out = pd.DataFrame(index=df.index)
    out["description"] = df["description"].fillna("").astype(str)
    if "desc_chars" not in df:
        df = df.assign(desc_chars=out["description"].str.len(),
                       desc_words=out["description"].str.split().map(len))
    for col in NUMERIC:
        out[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).clip(lower=0)
    for col in BOOL:
        out[col] = df.get(col, False)
        out[col] = out[col].astype(str).str.lower().isin(["true", "1"]).astype(float)
    return out


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray,
                    grid=np.arange(0.05, 0.96, 0.01)) -> np.ndarray:
    """Per-tag decision threshold that maximizes that tag's F1 on validation."""
    best = np.full(y_true.shape[1], 0.5)
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], proba[:, j] >= t, zero_division=0) for t in grid]
        best[j] = float(grid[int(np.argmax(scores))])
    return best


def tag_metrics(y_true: np.ndarray, y_pred: np.ndarray, tags=TAGS) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_tag": {
            t: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                "support": int(s[i])}
            for i, t in enumerate(tags)
        },
    }


# --- Gate ---------------------------------------------------------------------

def agreement(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Set-F1 between predicted and ground-truth tag sets, row-wise.

    Rows whose ground truth has none of the 22 tags get NaN (cannot be gated).
    """
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    inter = (pred & truth).sum(1)
    denom = pred.sum(1) + truth.sum(1)
    score = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), np.nan)
    score[truth.sum(1) == 0] = np.nan
    return score


def gate_decisions(score: np.ndarray, gate_threshold: float) -> np.ndarray:
    return np.where(np.isnan(score), "human_review",
                    np.where(score >= gate_threshold, "approve", "reject"))


def gate_rates(score_genuine: np.ndarray, score_mismatch: np.ndarray, t: float) -> dict:
    g = score_genuine[~np.isnan(score_genuine)]
    m = score_mismatch[~np.isnan(score_mismatch)]
    return {
        "auto_approval_rate": float((g >= t).mean()) if len(g) else 0.0,
        "false_approval_rate": float((m >= t).mean()) if len(m) else 0.0,
    }


def tune_gate_threshold(score_genuine, score_mismatch, max_false_approval=0.05,
                        grid=np.arange(0.0, 1.001, 0.01)) -> tuple[float, dict]:
    """Highest auto-approval subject to false-approval <= the ceiling."""
    best_t, best = 1.0, gate_rates(score_genuine, score_mismatch, 1.0)
    for t in grid:
        rates = gate_rates(score_genuine, score_mismatch, float(t))
        if (rates["false_approval_rate"] <= max_false_approval
                and rates["auto_approval_rate"] > best["auto_approval_rate"]):
            best_t, best = float(t), rates
    return round(best_t, 2), best
