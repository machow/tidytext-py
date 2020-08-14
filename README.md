tidytext-py
===========

Handy text processing in python, using pandas DataFrames.

This library is a python port of the [R package tidytext](https://github.com/juliasilge/tidytext). 

Install
-------

```
pip install tidytext
```

This will also install the nltk package.
However, you will need to download additional resources to use tidytext, using the code below.

```python
nltk.download("punkt")
```

Functions implemented
---------------------

* bind_tfidf
* unnest_tokens

Examples
--------


```python
import pandas as pd

pd.set_option("display.max_rows", 6)

zen = """
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
"""

zen_split = zen.splitlines()


df = pd.DataFrame({
    "zen": zen_split,
    "line": list(range(len(zen_split)))
})

df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zen</th>
      <th>line</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Zen of Python, by Tim Peters</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>If the implementation is hard to explain, it's a bad idea.</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>If the implementation is easy to explain, it may be a good idea.</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Namespaces are one honking great idea -- let's do more of those!</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 2 columns</p>




```python
from tidytext import unnest_tokens

unnest_tokens(df, "word", "zen")
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>line</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>the</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>zen</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>more</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>of</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>those</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 2 columns</p>


