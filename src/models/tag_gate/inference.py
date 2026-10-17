"""SageMaker inference handlers for the Tag QA Gate (batch transform / endpoint).

Input: JSON lines, one draft per line::

    {"app_id": 123, "description": "...", "price": 9.99, "required_age": 0,
     "windows": true, "mac": false, "linux": false, "tags": ["Indie", "2D"]}

``tags`` is the game's ground-truth tag list. Output: one JSON line per draft
with the predicted tags, the agreement score and the gate decision
(approve / reject / human_review).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

import model_lib as ml

JSONLINES = "application/jsonlines"


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(body, content_type: str = JSONLINES) -> pd.DataFrame:
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if content_type in (JSONLINES, "application/x-jsonlines", "application/json"):
        text = body.strip()
        if content_type == "application/json" and text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame.from_records(records)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(df: pd.DataFrame, artifact) -> list[dict]:
    tags = artifact["tags"]
    thresholds = np.asarray(artifact["thresholds"])
    proba = artifact["pipeline"].predict_proba(ml.prepare_features(df))
    pred = proba >= thresholds

    truth_lists = df["tags"] if "tags" in df else pd.Series([[]] * len(df))
    truth = np.array([[int(t in set(ts or [])) for t in tags] for ts in truth_lists])
    score = ml.agreement(pred, truth)
    decision = ml.gate_decisions(score, artifact["gate_threshold"])

    out = []
    for i in range(len(df)):
        out.append({
            "app_id": int(df["app_id"].iloc[i]) if "app_id" in df else i,
            "predicted_tags": [t for t, p in zip(tags, pred[i]) if p],
            "true_tags": [t for t, v in zip(tags, truth[i]) if v],
            "agreement": None if np.isnan(score[i]) else round(float(score[i]), 4),
            "decision": str(decision[i]),
            "top_probabilities": {
                tags[j]: round(float(proba[i, j]), 4) for j in np.argsort(-proba[i])[:5]
            },
        })
    return out


def output_fn(prediction: list[dict], accept: str = JSONLINES):
    body = "\n".join(json.dumps(r) for r in prediction) + "\n"
    try:  # SageMaker scikit-learn serving stack
        from sagemaker_containers.beta.framework import worker
        return worker.Response(body, mimetype=JSONLINES)
    except ImportError:
        return body, JSONLINES
