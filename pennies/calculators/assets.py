from __future__ import absolute_import, division, print_function

from pennies.trading import assets
from . import payments
class AssetCalculator(object):

    def __init__(self, contract, market):
        """
        Parameters
        ----------
        contract: Asset
            Asset (or Trade) representing the payment of a fixed amount
        market: Market
            Market values required to price the Asset.
        """
        self.contract = contract
        self.market = market


def all_calculators():
    return AssetCalculator.__subclasses__()

# TODO Is there a way to avoid setting this manually?
def default_calculators():
    return {
        str(assets.BulletPayment): payments.BulletPaymentCalculator,
        str(assets.DiscountBond): payments.BulletPaymentCalculator,
        str(assets.SettlementPayment): payments.BulletPaymentCalculator,
        str(assets.Zero): payments.BulletPaymentCalculator,
        str(assets.ZeroCouponBond): payments.BulletPaymentCalculator,
        str(assets.CompoundAsset): None
    }
