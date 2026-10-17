"""The Fronkon games.csv header has 39 names for 40 values per row."""
import io

import pandas as pd

from src.data.loaders import STORE_COLUMNS, load_store_games

HEADER = ",".join(STORE_COLUMNS).replace("Discount,DLC count", "DiscountDLC count")
ROW = ('10,"Pixel Quest","Jan 5, 2020","0 - 20000",3,0,4.99,0,2,"A cozy pixel-art puzzle '
       'adventure about a lighthouse keeper.","[\'English\']","[]","","h.jpg","","","",True,False,'
       'True,0,"",0,120,5,"",10,0,"",30,0,25,0,"Dev Studio","Pub House","Single-player",'
       '"Indie,Puzzle","Puzzle,Pixel Graphics,2D","",""')


def make_csv() -> io.StringIO:
    return io.StringIO(HEADER + "\n" + ROW + "\n")


def test_header_really_is_malformed():
    assert len(HEADER.split(",")) == 39
    naive = pd.read_csv(make_csv(), index_col=False)
    assert naive.loc[0, "Tags"] != "Puzzle,Pixel Graphics,2D"  # silently shifted


def test_loader_pins_40_columns():
    df = load_store_games(make_csv())
    row = df.iloc[0]
    assert row["app_id"] == 10
    assert row["tags"] == ["Puzzle", "Pixel Graphics", "2D"]
    assert row["genres"] == ["Indie", "Puzzle"]
    assert row["price"] == 4.99
    assert row["dlc_count"] == 2
    assert bool(row["windows"]) and not bool(row["mac"]) and bool(row["linux"])
    assert row["developers"] == "Dev Studio"
