"""Pipeline step 1 — validate the feature snapshot and write train/val/test.

Input  (/opt/ml/processing/input/snapshot): CSV export of the draft-time feature
group (see notebooks/02_feature_store.ipynb).
Output (/opt/ml/processing/output/{train,val,test,incoming,monitor}).

Fails the pipeline (non-zero exit) if the snapshot violates the data contract:
missing columns, too few rows, or a tag with too few positive examples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import TAGS, split_for  # noqa: E402

REQUIRED = ["app_id", "description", "price", "required_age", "desc_chars", "desc_words",
            "windows", "mac", "linux", "release_date", "tags_22"]
MONITOR_COLUMNS = ["price", "required_age", "desc_chars", "desc_words",
                   "windows", "mac", "linux", "n_tags_22"]


def load_snapshot(path: str) -> pd.DataFrame:
    files = [os.path.join(root, f) for root, _, fs in os.walk(path) for f in fs
             if f.endswith(".csv")]
    if not files:
        raise SystemExit(f"DATA CONTRACT: no CSV files under {path}")
    return pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)


def validate(df: pd.DataFrame, min_rows: int, min_positives: int) -> dict:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"DATA CONTRACT: missing columns {missing}")
    if len(df) < min_rows:
        raise SystemExit(f"DATA CONTRACT: {len(df)} rows < minimum {min_rows}")
    counts = {t: int(df["tags_22"].str.contains(rf"(?:^|\|){t}(?:\||$)", regex=True).sum())
              for t in TAGS}
    weak = {t: c for t, c in counts.items() if c < min_positives}
    if weak:
        raise SystemExit(f"DATA CONTRACT: tags below {min_positives} positives: {weak}")
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/opt/ml/processing/input/snapshot")
    p.add_argument("--output", default="/opt/ml/processing/output")
    p.add_argument("--min-rows", type=int, default=50_000)
    p.add_argument("--min-positives", type=int, default=2_000)
    args = p.parse_args()

    df = load_snapshot(args.input)
    df = df.drop_duplicates("app_id")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["tags_22"] = df["tags_22"].astype(str).replace("nan", "")
    tag_counts = validate(df, args.min_rows, args.min_positives)
    df["split"] = [split_for(a, d) for a, d in zip(df["app_id"], df["release_date"])]
    df["n_tags_22"] = df["tags_22"].map(lambda v: len(v.split("|")) if v else 0)

    for split in ("train", "val", "test", "incoming"):
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)
        df[df["split"] == split].to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)

    # Tabular view of the training data: baseline for SageMaker Model Monitor.
    mon_dir = os.path.join(args.output, "monitor")
    os.makedirs(mon_dir, exist_ok=True)
    mon = df[df["split"] == "train"][MONITOR_COLUMNS].copy()
    for col in ("windows", "mac", "linux"):
        mon[col] = mon[col].astype(str).str.lower().isin(["true", "1"]).astype(int)
    mon.to_csv(os.path.join(mon_dir, "train_features.csv"), index=False)

    profile = {"rows": int(len(df)), "splits": df["split"].value_counts().to_dict(),
               "tag_counts": tag_counts}
    with open(os.path.join(args.output, "monitor", "data_profile.json"), "w") as fh:
        json.dump(profile, fh, indent=2)
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
