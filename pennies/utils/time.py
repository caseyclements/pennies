from __future__ import absolute_import, division, print_function

import datetime as dt
import pandas as pd


def calendar_date(date_time):
    if isinstance(date_time, dt.datetime):
        return date_time.date()
    elif not isinstance(date_time, dt.date):
        raise ValueError('Expecting date or datetime. Found {} of type {}'
                         .format(date_time, type(date_time)))
    return date_time


def to_datetime(date):
    if isinstance(date, dt.date):
        return dt.datetime(date.year, date.month, date.day)
    elif not isinstance(date, dt.datetime):
        raise ValueError('Expecting date or datetime. Found {} of type {}'
                         .format(date, type(date)))
    else:
        return date


def act365_fixed(dt_start, dt_end):
    return (calendar_date(dt_end) - calendar_date(dt_start)).days / 365.0

_map_daycounts = {'Act/365 Fixed': act365_fixed}
"""Standard day count conventions for computing year fractions."""
# TODO - Is this to remain? should it be a dictionary to functions?


def daycount(name):  # TODO This is horrible. Change after daycounts added
    return _map_daycounts.get(name, act365_fixed)


FREQ_TO_YEAR_FRAC = {
    'D': 1/365,
    'W': 1/52,
    'WS': 1/52,
    'M': 1/12,
    'MS': 1/12,
    'Q': 1/4,
    'QS': 1/4,
    'A': 1,
    'AS': 1
}

def freq_to_frac(freq):
    return FREQ_TO_YEAR_FRAC[freq]


def date_range(start, end, freq):
    """
    Generate range of dates.

    Parameters
    ----------

    start: str, date, datetime
        start date of range (exclusive)
    end: str, date, datetime
        end date of range(inclusive)
    freq: str
        D, W, M, Q, A for end (WS, MS, QS, AS for start)
        If None, return DatetimeIndex with end.

    Returns
    -------

    `DatetimeIndex`
    """
    if freq is None:
        return pd.DatetimeIndex([end])

    if isinstance(end, str):
        end = dt.datetime.strptime(end, '%Y-%m-%d')

    return pd.date_range(start, end, freq=freq)


def to_offset(s):
    """Pass in a string to get a date offset.

    Arguments
    ---------

    s: str
        offset string which has format "<int> <freq>" where
        frequency can be one of "days", "months", or "years".
    """
    amount, freq = s.split(' ')
    kwargs = {freq: int(amount)}
    return pd.tseries.offsets.DateOffset(**kwargs)


def years_difference(start, end):
    print(start, end)
    return (end - start).total_seconds() / 31536000
