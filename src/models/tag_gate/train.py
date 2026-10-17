"""Train the Tag QA Gate (SageMaker SKLearn entry point; also runs locally).

Channels (CSV produced by src/pipelines/scripts/prepare.py):
  train/  train.csv
  val/    val.csv

Writes ``model.joblib`` (pipeline + per-tag thresholds + gate threshold) and
``val_metrics.json`` to the model directory.

Local run:
  python src/models/tag_gate/train.py --kind improved \
      --train data/train --val data/val --model-dir /tmp/model
"""

from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

import model_lib as ml


def read_channel(path: str) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
    return pd.concat([pd.read_csv(os.path.join(path, f), keep_default_na=False) for f in files],
                     ignore_index=True)


def labels(df: pd.DataFrame) -> np.ndarray:
    tag_lists = df["tags_22"].map(lambda v: v.split("|") if v else [])
    return np.array([[int(t in set(ts)) for t in ml.TAGS] for ts in tag_lists], dtype=np.int8)


def mismatched_labels(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Labels of *other* games, simulating drafts written for the wrong game."""
    perm = np.random.default_rng(seed).permutation(len(y))
    perm = np.where(perm == np.arange(len(y)), (perm + 1) % len(y), perm)
    return y[perm]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="improved", choices=["baseline", "improved"])
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--max-false-approval", type=float, default=0.05)
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--val", default=os.environ.get("SM_CHANNEL_VAL", "data/val"))
    args = p.parse_args()

    train, val = read_channel(args.train), read_channel(args.val)
    y_train, y_val = labels(train), labels(val)
    print(f"train={len(train)} val={len(val)} kind={args.kind}")

    start = time.time()
    model = ml.build_pipeline(args.kind, C=args.C, max_features=args.max_features)
    model.fit(ml.prepare_features(train), y_train)
    print(f"fit seconds={time.time() - start:.0f}")

    proba_val = model.predict_proba(ml.prepare_features(val))
    # The baseline keeps the default 0.5 cut-off; the improved model tunes one per tag.
    thresholds = (np.full(len(ml.TAGS), 0.5) if args.kind == "baseline"
                  else ml.tune_thresholds(y_val, proba_val))
    pred_val = (proba_val >= thresholds).astype(int)

    # Gate threshold: maximize auto-approval with false-approval <= ceiling.
    score_genuine = ml.agreement(pred_val, y_val)
    score_mismatch = ml.agreement(pred_val, mismatched_labels(y_val))
    gate_t, rates = ml.tune_gate_threshold(score_genuine, score_mismatch,
                                           args.max_false_approval)

    metrics = ml.tag_metrics(y_val, pred_val)
    metrics.update({"kind": args.kind, "gate_threshold": gate_t, **rates})
    # SageMaker scrapes these lines into training-job metrics.
    print(f"val_macro_f1={metrics['macro_f1']:.4f};")
    print(f"val_micro_f1={metrics['micro_f1']:.4f};")
    print(f"val_auto_approval={rates['auto_approval_rate']:.4f};")
    print(f"val_false_approval={rates['false_approval_rate']:.4f};")

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump({"pipeline": model, "tags": ml.TAGS, "thresholds": thresholds.tolist(),
                 "gate_threshold": gate_t, "kind": args.kind},
                os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "val_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
