from __future__ import absolute_import, division, print_function

import datetime as dt


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
