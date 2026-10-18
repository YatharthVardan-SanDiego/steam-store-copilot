import numpy as np
import pandas as pd

from src.models.reception import features as rf


def test_chronological_split_has_no_time_overlap():
    dates = pd.to_datetime(
        ["2021-06-01", "2022-03-01", "2022-08-01", "2021-12-31", "2022-07-01"]
    )
    df = pd.DataFrame({"date": dates, rf.LABEL: [0, 1, 0, 1, 0]})
    s = rf.chronological_split(df)

    assert s["train"]["date"].max() < s["val"]["date"].min()
    assert s["val"]["date"].max() < s["test"]["date"].min()
    assert sum(len(v) for v in s.values()) == len(df)


def test_features_exclude_reception_aggregates():
    leaky = {
        "positive_ratio",
        "rating",
        "user_reviews_game",
        "positive",
        "negative",
        "is_recommended",
    }
    assert not leaky & set(rf.FEATURES)
    assert rf.LABEL not in rf.FEATURES


def test_xgboost_csv_puts_label_first():
    df = pd.DataFrame({c: [0.0] for c in rf.FEATURES} | {rf.LABEL: [1]})
    out = rf.to_xgboost_csv(df)

    assert out.columns[0] == rf.LABEL
    assert len(out.columns) == len(rf.FEATURES) + 1


def test_scale_pos_weight():
    assert rf.scale_pos_weight(np.array([1, 0, 0, 0])) == 3.0
