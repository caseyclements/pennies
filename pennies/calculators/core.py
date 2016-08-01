from __future__ import absolute_import, division, print_function

import math
import pandas as pd
from datetime import date, datetime
from pennies.assets.core import Asset, CashFlow
from pennies.dispatch import dispatch
from pennies.utils.time import years_difference

BASE_TYPES = (object, Asset)


@dispatch(BASE_TYPES, object)
def present_value(asset, discount_factor):
    raise NotImplementedError("Not available for base types")


@dispatch(pd.DataFrame, object)
def present_value(asset_list, discount_factor):
    return asset_list.applymap(lambda x: present_value(x, discount_factor))


@dispatch(pd.Series, object)
def present_value(asset_list, discount_factor):
    return asset_list.apply(lambda x: present_value(x, discount_factor))


@dispatch(CashFlow, object)
def present_value(cf, discount_factor):
    today = datetime.combine(date.today(), datetime.min.time())

    # if payment was in the past, ignore cash flow
    if today > cf.payment_date:
        return 0

    years = years_difference(today, cf.payment_date)
    return cf.amount * math.exp(-1.0 * discount_factor * years)


@dispatch(BASE_TYPES)
def future_value(x):
    raise NotImplementedError("Not available for base types")


@dispatch(pd.DataFrame)
def future_value(x):
    return x.applymap(future_value)


@dispatch(BASE_TYPES)
def yield_(x):
    raise NotImplementedError("Not available for base types")


@dispatch(pd.DataFrame)
def yield_(x):
    return x.applymap(yield_)
