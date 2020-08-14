__version__ = "0.0.1"

import string
import numpy as np

from siuba.dply.verbs import singledispatch2
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy

from siuba.dply.verbs import simple_varname
from nltk import tokenize


import warnings

TOKEN_OPTIONS = {
      "words": tokenize.word_tokenize,
      "ngrams": None,
      "skip_ngrams": None,
      "sentences": tokenize.sent_tokenize,
      "lines": None,
      "paragraphs": None,
      "regex": None,
      "character_shingles": None,
      }

PUNCTUATION_MAP = str.maketrans(dict.fromkeys(string.punctuation))

def _get_tokenizer(token, remove_punc = True, mapping = TOKEN_OPTIONS):
    # see https://stackoverflow.com/q/15547409/1144523 
    f_tokenize = TOKEN_OPTIONS[token]
    if f_tokenize is None:
        raise NotImplementedError("token option %s not implemented" %token)

    punc_dict = dict.fromkeys(string.punctuation)
    if remove_punc:
        return lambda x: f_tokenize(x.translate(PUNCTUATION_MAP))

    return f_tokenize


@singledispatch2(DataFrame)
def unnest_tokens(
        __data, output, input,
        token = "words",
        format = "text",
        to_lower = True,
        drop = True,
        collapse = None,
        *args, **kwargs
        ):

    if token not in TOKEN_OPTIONS:
        raise ValueError("token must be one of %s" % TOKEN_OPTIONS)
    elif token != "words":
        raise NotImplementedError("Currently, token argument must be 'words'")

    # convert bare columns, like _.char to "char"
    output_col, input_col = map(simple_varname, [output, input])

    nested_df = __data.drop(columns = input_col) if drop else __data.copy()
    
    # make each row hold a list of tokens
    # note that the tidytext R package drops punctuation.
    #   E.g. "a b -" -> ["a", "b"]
    #   E.g. "a b-"  -> ["a", "b-"]
    f_tokenize = _get_tokenizer(token, remove_punc = True)
    tokens = __data[input_col].transform(f_tokenize)

    # assign to final DataFrame and unnest (explode)
    out_df = nested_df.assign(**{output_col: tokens}).explode(output_col)

    if to_lower:
        out_df[output_col] = out_df[output_col].str.lower()

    return out_df


@singledispatch2(DataFrame)
def bind_tf_idf(__data, term, document, n):
    term, document, n_col = map(simple_varname, [term, document, n])

    # TODO: coerce all to be strings?
    # term is on index, its count the value
    ser_terms = __data[term]
    ser_n     = __data[n_col]
    term_counts = __data[term].value_counts()

    # document on index
    g_doc = __data.groupby(document)
    doc_totals = g_doc[n_col].transform("sum")

    # index is still term
    idf = (g_doc.ngroups / term_counts).transform(np.log)

    out_df = __data.copy()
    out_df["tf"] = ser_n / doc_totals
    out_df["idf"] = idf.loc[ser_terms].reset_index(drop = True)
    out_df["tf_idf"] = out_df["tf"] * out_df["idf"]

    if (out_df["idf"] < 0).any():
        warnings.warn(
                "A value for tf_idf is negative: \n"
                "Input should have exactly one row per document-term combination."
                )

    return out_df

@bind_tf_idf.register(DataFrameGroupBy)
def _bind_tf_idf_gdf(__data, term, document, n):
    res = bind_tf_idf(__data.obj, term, document, n)
    return res.groupby(__data.grouper)

