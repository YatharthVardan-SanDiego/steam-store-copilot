"""Model-quality monitoring for batch outputs + CloudWatch publishing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import CLOUDWATCH_NAMESPACE
from src.data.preprocess import TAGS


def read_jsonlines(text: str) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in text.splitlines() if line.strip()])


def batch_quality(outputs: pd.DataFrame, is_mismatch: pd.Series) -> dict:
    """Business + model metrics for one batch of gate decisions.

    ``outputs`` are batch-transform rows; ``is_mismatch`` marks the drafts that
    were deliberately paired with the wrong game (ground truth for false approvals).
    """
    is_mismatch = is_mismatch.astype(bool).values
    decision = outputs["decision"].values
    genuine = ~is_mismatch
    gated = decision != "human_review"

    y_true = np.array([[int(t in set(ts)) for t in TAGS] for ts in outputs["true_tags"]])
    y_pred = np.array([[int(t in set(ts)) for t in TAGS] for ts in outputs["predicted_tags"]])
    rows = genuine & gated
    return {
        "records": int(len(outputs)),
        "macro_f1": float(f1_score(y_true[rows], y_pred[rows], average="macro",
                                   zero_division=0)) if rows.any() else 0.0,
        "auto_approval_rate": float((decision[genuine & gated] == "approve").mean())
        if (genuine & gated).any() else 0.0,
        "false_approval_rate": float((decision[is_mismatch & gated] == "approve").mean())
        if (is_mismatch & gated).any() else 0.0,
        "human_review_rate": float((~gated).mean()),
    }


def publish_metrics(metrics: dict, dimensions: dict, cloudwatch_client) -> None:
    dims = [{"Name": k, "Value": str(v)} for k, v in dimensions.items()]
    data = [{"MetricName": k, "Value": float(v), "Dimensions": dims, "Unit": "None"}
            for k, v in metrics.items() if isinstance(v, (int, float))]
    cloudwatch_client.put_metric_data(Namespace=CLOUDWATCH_NAMESPACE, MetricData=data)
