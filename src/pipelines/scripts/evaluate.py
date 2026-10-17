"""Pipeline step — evaluate baseline and candidate on the held-out test split.

Writes ``evaluation.json`` (read by the pipeline's quality-gate ConditionStep and
attached to the Model Registry entry) and ``per_tag.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_lib as ml  # noqa: E402


def load_model(model_dir: str):
    tar = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar):
        with tarfile.open(tar) as fh:
            fh.extractall(model_dir)
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def labels(series: pd.Series) -> np.ndarray:
    lists = series.map(lambda v: v.split("|") if isinstance(v, str) and v else [])
    return np.array([[int(t in set(ts)) for t in ml.TAGS] for ts in lists], dtype=np.int8)


def score(artifact, test: pd.DataFrame, y: np.ndarray, y_mismatch: np.ndarray) -> dict:
    proba = artifact["pipeline"].predict_proba(ml.prepare_features(test))
    pred = (proba >= np.asarray(artifact["thresholds"])).astype(int)
    out = ml.tag_metrics(y, pred)
    out.update(ml.gate_rates(ml.agreement(pred, y), ml.agreement(pred, y_mismatch),
                             artifact["gate_threshold"]))
    out["gate_threshold"] = artifact["gate_threshold"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="/opt/ml/processing/model/baseline")
    p.add_argument("--candidate", default="/opt/ml/processing/model/candidate")
    p.add_argument("--test", default="/opt/ml/processing/test")
    p.add_argument("--output", default="/opt/ml/processing/evaluation")
    args = p.parse_args()

    test = pd.concat([pd.read_csv(os.path.join(args.test, f), keep_default_na=False)
                      for f in os.listdir(args.test) if f.endswith(".csv")], ignore_index=True)
    y = labels(test["tags_22"])
    perm = np.random.default_rng(7).permutation(len(y))
    perm = np.where(perm == np.arange(len(y)), (perm + 1) % len(y), perm)
    y_mismatch = y[perm]

    base = score(load_model(args.baseline), test, y, y_mismatch)
    cand = score(load_model(args.candidate), test, y, y_mismatch)

    report = {
        "tag_gate": {
            "macro_f1": {"value": cand["macro_f1"]},
            "micro_f1": {"value": cand["micro_f1"]},
            "baseline_macro_f1": {"value": base["macro_f1"]},
            "lift_over_baseline": {"value": cand["macro_f1"] - base["macro_f1"]},
            "auto_approval_rate": {"value": cand["auto_approval_rate"]},
            "false_approval_rate": {"value": cand["false_approval_rate"]},
            "gate_threshold": {"value": cand["gate_threshold"]},
            "test_rows": {"value": int(len(test))},
        },
        "per_tag": cand["per_tag"],
        "baseline_per_tag": base["per_tag"],
    }
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "evaluation.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    rows = [{"tag": t, "model": "candidate", **m} for t, m in cand["per_tag"].items()]
    rows += [{"tag": t, "model": "baseline", **m} for t, m in base["per_tag"].items()]
    pd.DataFrame(rows).to_csv(os.path.join(args.output, "per_tag.csv"), index=False)
    print(json.dumps(report["tag_gate"], indent=2))


if __name__ == "__main__":
    main()
