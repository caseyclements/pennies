from __future__ import absolute_import, division, print_function
from pennies.assets.core import Asset
from pennies.dispatch import dispatch
from pandas import DataFrame

BASE_TYPES = (object, Asset)


@dispatch(BASE_TYPES, object)
def present_value(asset, discount_factor):
    raise NotImplementedError("Not available for base types")


@dispatch(DataFrame, object)
def present_value(asset_list, discount_factor):
    return x.applymap(present_value)


@dispatch(BASE_TYPES)
def future_value(x):
    raise NotImplementedError("Not available for base types")


@dispatch(DataFrame)
def future_value(x):
    return x.applymap(future_value)


@dispatch(BASE_TYPES)
def yield_(x):
    raise NotImplementedError("Not available for base types")


@dispatch(DataFrame)
def yield_(x):
    return x.applymap(yield_)
