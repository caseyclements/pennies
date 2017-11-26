"""Daycount calculations, Schedule creation, and so on."""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from pandas import Series
from datetime import date


def act365_fixed(start, end):
    """Compute accrual fraction using convention: Actual/365 Fixed"""
    if isinstance(start, pd.DatetimeIndex):
        start = Series(start)
    if isinstance(end, pd.DatetimeIndex):
        end = Series(end)
    if isinstance(start, Series) or isinstance(end, Series):
        return (end - start) / np.timedelta64(365, 'D')
    else:
        return (end - start).days / 365.0


def year_30360(dt):
    """Helper function for thirty360"""
    return dt.year + dt.month / 12 + dt.day / 360


def thirty360(start, end):
    """Compute accrual fraction using convention: 30/360 Unadjusted

    This version does not apply any End-Of-Month (EOM) rules.
    ==> It is rarely used in practice!

    It is fast and simple, so valuable for development and testing.
    """
    if isinstance(start, date):
        start = year_30360(start)
    else:
        start = Series(start).map(year_30360)
    if isinstance(end, date):
        end = year_30360(end)
    else:
        end = Series(end).map(year_30360)
    return end - start

daycount_conventions = {
    'ACT365FIXED': act365_fixed,
    '30360': thirty360,
}


def daycounter(name=None):
    """Function to compute accrual, given name from daycount_conventions"""
    return daycount_conventions.get(name, thirty360)

if __name__ == '__main__':
    # TODO Move this into tests
    today = pd.to_datetime('today')
    years_later = today + pd.DateOffset(years=4)
    print('scalar - act365: {}'.format(act365_fixed(today, years_later)))
    print('scalar - thirty360: {}'.format(thirty360(today, years_later)))

    dt_settlement = today
    frequency, tenor = 2, 6
    dt_maturity = dt_settlement + pd.DateOffset(months=tenor)
    period = pd.DateOffset(months=frequency)
    sched_end = pd.date_range(dt_settlement, dt_maturity,
                              freq=period, closed='right')

    print('Series - act365: {}'.format(act365_fixed(today, sched_end)))
    print('Series - thirty360: {}'.format(thirty360(today, sched_end)))





