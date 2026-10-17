import json

import numpy as np
import pandas as pd
import pytest

import inference
import model_lib as ml

VOCAB = {
    "Action": "shoot fight combat explosive guns",
    "Puzzle": "puzzle logic riddle brain solve",
    "2D": "pixel sprite flat retro side",
    "Singleplayer": "story campaign solo journey hero",
}


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tags = [t for t in VOCAB if rng.random() < 0.45] or ["Singleplayer"]
        words = " ".join(VOCAB[t] for t in tags) + " game fun play"
        rows.append({"app_id": i, "description": words, "price": float(rng.integers(0, 30)),
                     "required_age": 0, "windows": True, "mac": False, "linux": bool(i % 2),
                     "tags": tags})
    return pd.DataFrame(rows)


def labels(df):
    return np.array([[int(t in set(ts)) for t in ml.TAGS] for ts in df["tags"]])


@pytest.fixture(scope="module")
def artifact():
    df = synthetic()
    y = labels(df)
    # Tags absent from the synthetic vocabulary get one positive so every
    # one-vs-rest column has two classes.
    for j in np.where(y.sum(0) == 0)[0]:
        y[j, j] = 1
    model = ml.build_pipeline("improved", max_features=5000)
    model.fit(ml.prepare_features(df), y)
    return {"pipeline": model, "tags": ml.TAGS, "thresholds": [0.5] * 22,
            "gate_threshold": 0.5, "kind": "improved"}


def test_agreement_and_decisions():
    pred = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    truth = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    score = ml.agreement(pred, truth)
    assert score[0] == 1.0 and score[1] == 0.0 and np.isnan(score[2])
    assert list(ml.gate_decisions(score, 0.5)) == ["approve", "reject", "human_review"]


def test_gate_threshold_respects_false_approval_ceiling():
    genuine = np.linspace(0.3, 1.0, 100)
    mismatch = np.linspace(0.0, 0.7, 100)
    t, rates = ml.tune_gate_threshold(genuine, mismatch, max_false_approval=0.05)
    assert rates["false_approval_rate"] <= 0.05
    assert rates["auto_approval_rate"] > 0.3


def test_tune_thresholds_shape():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(200, 3))
    proba = np.clip(y * 0.6 + rng.random((200, 3)) * 0.4, 0, 1)
    th = ml.tune_thresholds(y, proba)
    assert th.shape == (3,) and ((th > 0) & (th < 1)).all()


def test_inference_round_trip(artifact):
    df = synthetic(5, seed=1)
    body = "\n".join(json.dumps(r) for r in df.to_dict("records"))
    frame = inference.input_fn(body.encode(), "application/jsonlines")
    preds = inference.predict_fn(frame, artifact)
    payload, content_type = inference.output_fn(preds, "application/jsonlines")
    lines = [json.loads(x) for x in payload.strip().splitlines()]
    assert content_type == "application/jsonlines"
    assert len(lines) == 5
    assert {"app_id", "predicted_tags", "agreement", "decision"} <= set(lines[0])
    assert all(r["decision"] in {"approve", "reject", "human_review"} for r in lines)


def test_unsupported_content_type():
    with pytest.raises(ValueError):
        inference.input_fn(b"a,b", "text/csv")
