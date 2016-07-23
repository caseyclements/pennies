from __future__ import absolute_import, division, print_function

from pennies.calculators.payments import BulletPaymentCalculator
import pennies.calculators.payments as payments
from pennies.trading import assets


# TODO: Rethink default_calculators as we examine dispatch
def default_calculators():
    return {
        str(assets.BulletPayment): payments.BulletPaymentCalculator,
        str(assets.DiscountBond): payments.BulletPaymentCalculator,
        str(assets.SettlementPayment): payments.BulletPaymentCalculator,
        str(assets.Zero): payments.BulletPaymentCalculator,
        str(assets.ZeroCouponBond): payments.BulletPaymentCalculator,
        str(assets.CompoundAsset): None
    }


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

    # TODO - Is there a way that I can get the calculators available for trade?
    def present_value(self):
        pv = self.asset_ccr.present_value()
        if self.settlement_ccr is not None:
            pv += self.settlement_ccr.present_value()
        return pv
