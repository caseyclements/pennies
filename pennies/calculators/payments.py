from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch

from pennies.trading.assets import BulletPayment
from pennies.market.market import RatesTermStructure


@dispatch(BulletPayment, RatesTermStructure, str)
def present_value(contract, market, reporting_ccy):
    """Present Value as sum of discount cash flows.

    This assumes that one has already computed the rates.
    For fixed rate annuities, this will be done during construction.
    For floating rate annuities, this will have to be pre-computed,
    typically via psuedo-discount factors of other curves."""

    df = market.discount_factor(contract.dt_payment, contract.currency)
    pv = contract.notional * df
    if reporting_ccy != contract.currency:
        pv *= market.fx(reporting_ccy, contract.currency)
    return pv
