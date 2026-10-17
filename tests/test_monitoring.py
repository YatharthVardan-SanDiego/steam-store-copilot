import json

import numpy as np
import pandas as pd

from src.monitoring import dashboard, retrain
from src.monitoring import data_quality as dq


def frame(n=2000, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": rng.gamma(2, 5, n) * scale, "required_age": rng.choice([0, 0, 0, 18], n),
        "desc_chars": rng.normal(900, 200, n) * scale, "desc_words": rng.normal(150, 30, n) * scale,
        "windows": np.ones(n), "mac": rng.integers(0, 2, n), "linux": rng.integers(0, 2, n),
        "n_tags_22": rng.integers(1, 8, n),
    })


def test_psi_no_drift_on_same_distribution():
    report = dq.psi_report(frame(seed=0), frame(seed=1))
    assert not report["drift"].any()


def test_psi_flags_shifted_features():
    report = dq.psi_report(frame(seed=0), frame(seed=1, scale=0.2)).set_index("feature")
    assert report.loc["desc_chars", "drift"] and report.loc["price", "drift"]
    assert not report.loc["n_tags_22", "drift"]


def test_should_retrain():
    calm = retrain.should_retrain([], 0, {"macro_f1": 0.60, "false_approval_rate": 0.04}, 0.61)
    assert not calm.retrain
    drift = retrain.should_retrain(["price"], 2, {"macro_f1": 0.50}, 0.61)
    assert drift.retrain and len(drift.reasons) == 3


def test_dashboard_body_is_valid_json():
    body = dashboard.dashboard_body("us-east-1")
    assert len(body["widgets"]) >= 8
    json.dumps(body)
