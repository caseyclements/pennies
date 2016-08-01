from __future__ import absolute_import, division, print_function

import math
import pandas as pd
from datetime import date, datetime
from pennies.assets.core import Asset, CashFlow
from pennies.dispatch import dispatch
from pennies.utils.time import years_difference

BASE_TYPES = (object, Asset)


@dispatch(BASE_TYPES, object)
def present_value(asset, market):
    """Base present value calculation.

    Given an asset (or sequence of assets), calculate it's present
    value as of today.  Use the supplied market to identify the
    relevant curve and discount factor.

    Parameters
    ----------

    asset: Asset
        Asset to calculate present value
    market: Market
        Market to retrieve discount factor
    """
    raise NotImplementedError("Not available for base types")


@dispatch(pd.DataFrame, object)
def present_value(asset_list, market):
    """Wrapper to apply present value to a pandas DataFrame of Assets."""
    return asset_list.applymap(lambda x: present_value(x, market))


@dispatch(pd.Series, object)
def present_value(asset_list, market):
    """Wrapper to apply present value to a pandas Series of Assets."""
    return asset_list.apply(lambda x: present_value(x, market))


@dispatch(CashFlow, object)
def present_value(cf, market):
    """Calculate present value of a single cash flow.

    Calculates the present value of a cash flow as of today.
    """
    today = datetime.combine(date.today(), datetime.min.time())

    # if payment was in the past, ignore cash flow
    if today > cf.payment_date:
        return 0

    years = years_difference(today, cf.payment_date)
    return cf.amount * math.exp(-1.0 * market.discount_factor(cf.payment_date, cf.currency) * years)


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
