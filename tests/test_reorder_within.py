import pandas as pd

from tidytext import reorder_within

def test_reorder_within():
    df = pd.DataFrame({
        "x":     ['v0', 'v1', 'v1', 'v0'],
        "group": ['a', 'a', 'b', 'b'],
        "by":    [1, 2, 3, 0]
        })

    res = reorder_within(df["x"], df["by"], df["group"])

    dst_cats = ["v0___b", "v0___a", "v1___a", "v1___b"]

    assert isinstance(res, pd.Categorical)
    assert all(res.categories == dst_cats)
    assert all(res == df["x"].str.cat(df["group"], sep = "___"))
    
