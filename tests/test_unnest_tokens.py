import pytest

from tidytext import unnest_tokens
import pandas as pd

from siuba.siu import _

@pytest.mark.skip
def test_unnest_tokens_character():
    df = pd.DataFrame({"txt": "Emily Dickinson"})
    res = unnest_tokens(df, _.char, _.txt, token = "characters")

    nrow, ncol = res.shape
    assert nrow == 14
    assert ncol == 1
    assert res["char"].iloc[0] == "e"


@pytest.mark.skip
def test_unnest_tokens_char_shingles():
    df = pd.DataFrame({"txt":  "tidytext is the best"})
    res = unnest_tokens(df, _.char_ngram, _.txt, token = "character_shingles", n = 4)

    nrow, ncol = res.shape
    assert nrow == 14
    assert ncol == 1
    assert df["char_ngram"].iloc[0] == "tidy"


@pytest.mark.skip
def test_unnest_tokens_char_shingles_whitespace():
    df = pd.DataFrame({"txt": "tidytext is the best!"})
    res = unnest_tokens(df, _.char_ngram, _.txt, token = "character_shingles", strip_non_alphanum = False)

    nrow, ncol = res.shape
    assert nrow == 19
    assert ncol == 1
    assert res["char_ngram"].iloc[0] == "tid"


def test_unnest_tokens_words():
    df = pd.DataFrame({"txt": [
        "Because I could not stop for Death -", 
        "He kindly stopped for me -"]
    })
    res = unnest_tokens(df, _.word, _.txt)
    nrow, ncol = res.shape
    assert nrow == 12
    assert ncol == 1
    assert res["word"].iloc[0] == "because"


def test_unnest_tokens_token_arg_wrong():
    df = pd.DataFrame({"txt": ["tidytext is the best!"]})

    with pytest.raises(ValueError):
        # TODO: test part of error message
        unnest_tokens(df, _.word, _.txt, token = "word")


# To continue tests, see https://github.com/juliasilge/tidytext/blob/master/tests/testthat/test-unnest-tokens.R#L56
