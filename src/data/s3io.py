"""Small S3 helpers so notebooks read raw data from S3 (the system of record)."""

from __future__ import annotations

import io
import os
from urllib.parse import urlparse

import boto3
import pandas as pd

RAW_FILES = {
    "store/games.csv": "Fronkon Games Steam Games Dataset (MIT) - descriptions, tags",
    "recs/games.csv": "Game Recommendations on Steam (CC0) - game ratings, prices",
    "recs/users.csv": "Game Recommendations on Steam (CC0) - reviewer history",
    "recs/recommendations.csv": "Game Recommendations on Steam (CC0) - 41M user reviews",
    "recs/games_metadata.json": "Game Recommendations on Steam (CC0) - descriptions, tags",
}

CACHE_DIR = os.path.expanduser("~/data-cache")


def split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def fetch(uri: str, cache_dir: str = CACHE_DIR) -> str:
    """Download an S3 object once into a local cache; returns the local path."""
    bucket, key = split_uri(uri)
    local = os.path.join(cache_dir, key)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        boto3.client("s3").download_file(bucket, key, local)
    return local


def stream(uri: str):
    """Streaming body for large objects (e.g. 2 GB recommendations.csv)."""
    bucket, key = split_uri(uri)
    return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]


def read_csv(uri: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(fetch(uri), **kwargs)


def write_text(uri: str, text: str) -> str:
    bucket, key = split_uri(uri)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    return uri


def write_csv(df: pd.DataFrame, uri: str, **kwargs) -> str:
    buf = io.StringIO()
    df.to_csv(buf, **kwargs)
    return write_text(uri, buf.getvalue())


def read_text(uri: str) -> str:
    bucket, key = split_uri(uri)
    return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def list_objects(uri: str) -> pd.DataFrame:
    bucket, prefix = split_uri(uri)
    rows = []
    for page in boto3.client("s3").get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rows.append({"key": obj["Key"], "size_mb": round(obj["Size"] / 1e6, 1),
                         "last_modified": obj["LastModified"]})
    return pd.DataFrame(rows)
