from __future__ import absolute_import, division, print_function

import math
import numbers
import pandas as pd
from pennies.assets.bonds import Bond
from pennies.dispatch import dispatch
from pennies.utils.time import date_range, freq_to_frac


def _cash_flows(principal, start_date, end_date, frequency, coupon):
    date_index = date_range(start_date, end_date, frequency)
    if frequency is None:
        frequency = 'A'
    payments = [principal * coupon * freq_to_frac(frequency)] * len(date_index)
    payments[-1] += principal
    return pd.Series(payments, index=date_index)


@dispatch(Bond, numbers.Number)
def present_value(b, discount_factor):
    payments = _cash_flows(b.principal, b.start_date, b.end_date, b.frequency, b.coupon)
    discount_factors = [1 + discount_factor] * len(payments)
    for i, df in enumerate(discount_factors):
        discount_factors[i] = math.pow(df, i+1)
    return (payments / discount_factors).sum()

