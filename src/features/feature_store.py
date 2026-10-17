"""SageMaker Feature Store: draft-time vs. post-release feature groups.

The split into two groups is the leakage guard: the Tag QA Gate's training
snapshot is built only from ``steam-draft-features``. Post-release signals
(reviews, playtime, CCU, owners) live in ``steam-post-release-features`` and can
never reach the tag model by accident.
"""

from __future__ import annotations

import time

import pandas as pd

from src.data.preprocess import encode_tags

DRAFT_COLUMNS = ["app_id", "description", "price", "required_age", "desc_chars", "desc_words",
                 "windows", "mac", "linux", "developers", "publishers", "release_date",
                 "release_year", "tags_22", "n_tags_22"]

POST_RELEASE_COLUMNS = ["app_id", "positive", "negative", "peak_ccu", "recommendations",
                        "avg_playtime_forever", "median_playtime_forever", "estimated_owners",
                        "positive_ratio", "user_reviews", "rating"]


def _finalize(df: pd.DataFrame, event_time: float) -> pd.DataFrame:
    """Feature Store needs explicit string dtypes and an event-time column."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype("int64")
        elif df[col].dtype == object:
            df[col] = df[col].fillna("").astype("string")
    df["event_time"] = float(event_time)
    return df.reset_index(drop=True)


def draft_feature_frame(tag_dataset: pd.DataFrame, event_time: float | None = None) -> pd.DataFrame:
    """Rows of ``build_tag_dataset`` -> records for the draft-time feature group."""
    df = tag_dataset.copy()
    df["tags_22"] = df["tags_22"].map(encode_tags)
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.strftime("%Y-%m-%d").fillna("")
    df["developers"] = df["developers"].astype(str).str.slice(0, 500)
    df["publishers"] = df["publishers"].astype(str).str.slice(0, 500)
    return _finalize(df[DRAFT_COLUMNS], event_time or time.time())


def post_release_feature_frame(store_games: pd.DataFrame, recs_games: pd.DataFrame,
                               app_ids, event_time: float | None = None) -> pd.DataFrame:
    """Post-release signals for the same games (never used by the tag model)."""
    df = store_games[store_games["app_id"].isin(set(app_ids))]
    df = df.merge(recs_games[["app_id", "positive_ratio", "user_reviews", "rating"]],
                  on="app_id", how="left")
    df["positive_ratio"] = df["positive_ratio"].fillna(-1).astype("int64")
    df["user_reviews"] = df["user_reviews"].fillna(0).astype("int64")
    df["rating"] = df["rating"].fillna("Unknown")
    df["estimated_owners"] = df["estimated_owners"].astype(str)
    return _finalize(df[POST_RELEASE_COLUMNS], event_time or time.time())


def create_feature_group(name: str, df: pd.DataFrame, offline_uri: str, role: str,
                         sagemaker_session, description: str):
    """Create (or reuse) a feature group whose schema is inferred from ``df``."""
    from sagemaker.feature_store.feature_group import FeatureGroup

    fg = FeatureGroup(name=name, sagemaker_session=sagemaker_session)
    try:
        status = fg.describe()["FeatureGroupStatus"]
        print(f"{name}: already exists ({status})")
        return fg
    except sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
        pass
    fg.load_feature_definitions(data_frame=df)
    fg.create(s3_uri=offline_uri, record_identifier_name="app_id",
              event_time_feature_name="event_time", role_arn=role,
              enable_online_store=True, description=description)
    while (status := fg.describe()["FeatureGroupStatus"]) == "Creating":
        print(f"{name}: {status} ...")
        time.sleep(10)
    print(f"{name}: {status}")
    return fg


def _to_record(row: dict) -> list[dict]:
    return [{"FeatureName": k, "ValueAsString": str(v)} for k, v in row.items()
            if v is not None and not (isinstance(v, float) and pd.isna(v))]


def ingest(fg, df: pd.DataFrame, max_workers: int = 16, report_every: int = 10_000) -> None:
    """PutRecord every row with a thread pool.

    Used instead of ``FeatureGroup.ingest``, whose multiprocessing pool can
    deadlock on macOS; threads behave the same on a laptop and in Studio.
    Throttled calls are retried by botocore's adaptive retry mode.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from botocore.config import Config

    runtime = fg.sagemaker_session.boto_session.client(
        "sagemaker-featurestore-runtime",
        config=Config(max_pool_connections=max_workers * 2,
                      retries={"max_attempts": 10, "mode": "adaptive"}))
    records = df.to_dict("records")

    def put(rows: list[dict]) -> int:
        for row in rows:
            runtime.put_record(FeatureGroupName=fg.name, Record=_to_record(row))
        return len(rows)

    start, done = time.time(), 0
    chunks = [records[i:i + 100] for i in range(0, len(records), 100)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in as_completed([pool.submit(put, c) for c in chunks]):
            done += future.result()
            if done % report_every < 100 or done == len(records):
                print(f"  {fg.name}: {done:,}/{len(records):,} records "
                      f"({time.time() - start:.0f}s)", flush=True)
    print(f"Ingested {len(records):,} records into {fg.name} in {time.time() - start:.0f}s")


def snapshot_query(table: str) -> str:
    """Latest record per game from the offline store, excluding deleted records."""
    cols = ", ".join(c for c in DRAFT_COLUMNS)
    return f"""
        SELECT {cols} FROM (
            SELECT *, row_number() OVER (PARTITION BY app_id ORDER BY event_time DESC,
                                         api_invocation_time DESC) AS rn
            FROM "{table}"
            WHERE NOT is_deleted
        ) WHERE rn = 1
    """


def offline_store_ready(fg, min_rows: int) -> bool:
    query = fg.athena_query()
    query.run(query_string=f'SELECT count(*) AS n FROM "{query.table_name}"',
              output_location=f"s3://{fg.sagemaker_session.default_bucket()}/athena-results/")
    query.wait()
    return int(query.as_dataframe()["n"].iloc[0]) >= min_rows
