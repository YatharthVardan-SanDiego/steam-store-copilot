import numpy as np
import pandas as pd

from src.data import preprocess as pp


def test_strip_bbcode():
    assert pp.strip_bbcode("[b]Bold[/b] and [url=http://x]link[/url]  text") == "Bold and link text"


def test_latin_share():
    assert pp.latin_share("A cozy puzzle game") == 1.0
    assert pp.latin_share("这是一个游戏") == 0.0


def test_split_is_deterministic_and_respects_incoming_cutoff():
    assert pp.split_for(123, pd.Timestamp("2020-01-01")) == pp.split_for(123, pd.Timestamp("2020-01-01"))
    assert pp.split_for(123, pd.Timestamp("2025-03-01")) == "incoming"
    splits = [pp.split_for(i, pd.Timestamp("2020-01-01")) for i in range(5000)]
    share = pd.Series(splits).value_counts(normalize=True)
    assert 0.77 < share["train"] < 0.83


def test_label_space_is_fixed_at_22():
    assert len(pp.TAGS) == 22 and len(set(pp.TAGS)) == 22
    assert pp.tag_label_vector(["Indie", "Horror"]).count(1) == 1  # Horror is out of scope


def test_build_tag_dataset_filters():
    games = pd.DataFrame({
        "app_id": [1, 2, 3, 4],
        "name": list("abcd"),
        "description": ["A long enough English description of a fun game." * 2, "short",
                        "这是一个很长的游戏描述" * 10, "Another long English description for game four!!"],
        "tags": [["Indie", "2D"], ["Indie"], ["Indie"], []],
        "price": [1.0] * 4, "required_age": [0] * 4,
        "windows": [True] * 4, "mac": [False] * 4, "linux": [False] * 4,
        "developers": ["d"] * 4, "publishers": ["p"] * 4,
        "release_date": pd.to_datetime(["2020-01-01"] * 4),
    })
    out = pp.build_tag_dataset(games)
    assert out["app_id"].tolist() == [1]
    assert out.loc[0, "tags_22"] == ["Indie", "2D"]


def test_mismatched_drafts_never_keep_their_own_tags():
    df = pd.DataFrame({"app_id": range(50), "tags_22": [[f"t{i}"] for i in range(50)]})
    mm = pp.make_mismatched_drafts(df)
    assert all(mm["tags_22"].iloc[i] != df["tags_22"].iloc[i] for i in range(50))
    assert mm["is_mismatch"].eq(1).all()
    assert np.array_equal(np.sort(mm["app_id"]), np.arange(50))
