---
layout: post
title:  "A Single Line Pandas Speed-up: You just need to..."
class: post-template
subclass: 'post'
comments: true
---

*If you deal with Python and Data Science, you definitely know [Pandas](https://pandas.pydata.org/a) --
an open source library, which provides an abstraction for tabular data structures like
SQL tables or CSV files. Python sometimes extremely slow when processing large amounts of data
(don't even try going through 10M of records using a loop). This is where Pandas (and [NumPy](http://www.numpy.org/))
comes in handy providing you SQL-like API for relational operations over your tables.
All the heavy stuff implemented in C and this is why Pandas is much faster than naive row-by-row loops.
But still, its performance can be significantly improved with a single line statement.*

```python
import pandas as pd

pd.set_option('mode.chained_assignment', None)
```

Despite the fact, that article title looks like a click-bait, there is no trick.
I have presented a solution at the first line, it is as simple as is. You can use it in your code right now. But please follow
my explanations to understand, why it has such influence to performance.

# Explanation

From [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html):

|mode.chained_assignment |warn |Controls `SettingWithCopyWarning`: `raise`, `warn`, or None. Raise an exception, warn, or no action if trying to use chained assignment.|

Two types of indexing.
Double indexing creates a copy.
Assigning to copy doesn't change an original.
Should be an exception or None because writing a WARNING cost too much.

# Experiment

# Solution

Tests with `raise`, production with `None`. 
