"""Batch inference helpers: draft batches, latest approved model, transform job."""

from __future__ import annotations

import json

import pandas as pd

from src.config import TAG_GATE_MODEL_GROUP
from src.data.preprocess import decode_tags


def make_draft_batch(df: pd.DataFrame, mismatch_share: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Incoming drafts: genuine descriptions plus a share paired with the wrong game.

    The mismatched share is labelled ``is_mismatch=1`` so the batch output can be
    scored for false approvals (a draft that should have been rejected).
    """
    df = df.reset_index(drop=True).copy()
    df["tags"] = df["tags_22"].map(decode_tags)
    df["is_mismatch"] = 0
    mism = df.sample(frac=mismatch_share, random_state=seed).index
    donors = df["tags"].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    for k, i in enumerate(mism):
        j = (k * 7919 + 1) % len(df)
        if donors[j] == df.at[i, "tags"]:
            j = (j + 1) % len(df)
        df.at[i, "tags"] = donors[j]
        df.at[i, "is_mismatch"] = 1
    return df


def to_jsonlines(df: pd.DataFrame) -> str:
    cols = ["app_id", "description", "price", "required_age", "windows", "mac", "linux", "tags"]
    records = []
    for r in df[cols].itertuples(index=False):
        rec = r._asdict()
        rec["app_id"] = int(rec["app_id"])
        rec["required_age"] = int(rec["required_age"])
        rec["price"] = float(rec["price"])
        for c in ("windows", "mac", "linux"):
            rec[c] = str(rec[c]).lower() in ("true", "1")
        records.append(json.dumps(rec))
    return "\n".join(records) + "\n"


def latest_approved_package(sagemaker_client, group: str = TAG_GATE_MODEL_GROUP) -> str:
    resp = sagemaker_client.list_model_packages(
        ModelPackageGroupName=group, ModelApprovalStatus="Approved",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    packages = resp["ModelPackageSummaryList"]
    if not packages:
        raise RuntimeError(f"No approved model package in {group}; approve one first.")
    return packages[0]["ModelPackageArn"]
