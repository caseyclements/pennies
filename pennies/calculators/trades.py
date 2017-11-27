from __future__ import absolute_import, division, print_function

from pennies.trading.trades import Trade, Portfolio
from pennies.market.market import RatesTermStructure
from multipledispatch import dispatch
# TODO Refactor to remove the Calculator
from pennies.calculators.payments import BulletPaymentCalculator
from pennies.calculators.assets import default_calculators
from pennies.calculators.swaps import present_value


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


class TradeCalculator(object):
    """Calculator for all Trades."""

    def __init__(self, trade, market):
        self.contract = trade.contract
        self.market = market
        asset_ccr_cls = default_calculators()[str(type(self.contract))]
        self.asset_ccr = asset_ccr_cls(self.contract, market)
        if trade.settlement is None:
            self.settlement_ccr = None
        else:
            self.settlement_ccr = BulletPaymentCalculator(trade.settlement,
                                                          market)

    # TODO - Is there a way to get all calculators available for trade?
    def present_value(self):
        pv = self.asset_ccr.present_value()
        if self.settlement_ccr is not None:
            pv += self.settlement_ccr.present_value()
        return pv

