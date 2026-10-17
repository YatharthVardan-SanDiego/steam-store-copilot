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


def ingest(fg, df: pd.DataFrame, max_workers: int = 4) -> None:
    start = time.time()
    fg.ingest(data_frame=df, max_workers=max_workers, wait=True)
    print(f"Ingested {len(df):,} records into {fg.name} in {time.time() - start:.0f}s")


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
