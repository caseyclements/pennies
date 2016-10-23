"""Daycount calculations, Schedule creation, and so on."""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from pandas import Series

from pandas.tseries.tools import normalize_date, to_datetime


def act365_fixed(start, end):
    if isinstance(start, pd.DatetimeIndex):
        start = Series(start)
    if isinstance(end, pd.DatetimeIndex):
        end = Series(end)
    if isinstance(start, Series) or isinstance(end, Series):
        return (end - start) / np.timedelta64(365, 'D')
    else:
        return (end - start).days / 365.0


_map_daycounts = {'ACT365FIXED': act365_fixed}
"""Map of Standard day count conventions for computing year fractions."""
# TODO - Is this to remain? should it be a dictionary to functions?


def daycount(name):  # TODO This is horrible. Change after daycounts added
    return _map_daycounts.get(name, act365_fixed)
