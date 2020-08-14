# corresponds to: 
# https://github.com/juliasilge/tidytext/blob/master/tests/testthat/test-tf-idf.R


import pandas as pd
import numpy as np
import pytest

from tidytext import bind_tf_idf
from siuba.siu import _

from pandas.testing import assert_frame_equal
from pandas.api.types import is_numeric_dtype
from pandas.core.groupby import DataFrameGroupBy

@pytest.fixture
def data():
    return pd.DataFrame({
        "document": [1]*5 + [2]*5,
        "word": ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "brown", "dog"],
        "frequency": [1, 1, 1, 1, 2]*2
        })


def test_bind_tf_idf_simple(data):
    res = bind_tf_idf(data, _.word, _.document, _.frequency)
    res2 = bind_tf_idf(data, "word", "document", "frequency")

    new_cols = ["document", "word", "frequency"]

    assert_frame_equal(res, res2)
    assert_frame_equal(data[new_cols], res[new_cols])

    #assert isinstance(res, pd.DataFrame)
    assert is_numeric_dtype(res["tf"])
    assert is_numeric_dtype(res["idf"])
    assert is_numeric_dtype(res["tf_idf"])

    assert res["tf"].equals(pd.Series([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3] * 2))
    assert res["idf"].iloc[:4].equals(pd.Series([0, np.log(2), 0, np.log(2)]))
    assert res["tf_idf"].equals(res["tf"] * res["idf"])

def test_bind_tf_idf_grouped(data):
    # preserves but ignores groups
    res = bind_tf_idf(data.groupby("document"), "word", "document", "frequency")

    assert isinstance(res, DataFrameGroupBy) 
    assert len(res.grouper.groupings) == 1
    assert res.grouper.groupings[0].name == "document"


#test_that("TF-IDF works when the document ID is a number", {
#  # example thanks to https://github.com/juliasilge/tidytext/issues/31
#  my_corpus <- dplyr::tibble(
#    id = rep(c(2, 3), each = 3),
#    word = c("an", "interesting", "text", "a", "boring", "text"),
#    n = c(1, 1, 3, 1, 2, 1)
#  )
#
#  tf_idf <- bind_tf_idf(my_corpus, word, id, n)
#  expect_false(any(is.na(tf_idf)))
#  expect_equal(tf_idf$tf_idf[c(3, 6)], c(0, 0))
#})
