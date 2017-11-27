from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch

from pennies.trading.trades import Trade, Portfolio
from pennies.market.market import RatesTermStructure
from pennies.calculators.assets import present_value


@dispatch(Trade, RatesTermStructure, str)
def present_value(trade, market, reporting_ccy):
    """Present Value of Trade and RatesTermStructure"""
    pv = present_value(trade.contract, market, reporting_ccy)
    if trade.settlement is not None:
        pv += present_value(trade.settlement, market, reporting_ccy)
    return pv


@dispatch(Portfolio, RatesTermStructure, str)
def present_value(portfolio, market, reporting_ccy):
    """Present Value of Trade and RatesTermStructure"""
    pv = 0.0
    for t in portfolio.trades:
        pv += present_value(t, market, reporting_ccy)
    for p in portfolio.subportfolios:
        pv += present_value(p, market, reporting_ccy)
    return pv
