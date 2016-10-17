from __future__ import absolute_import, division, print_function

from pennies.dispatch import dispatch
from six import string_types

from pennies.trading.assets import Asset
from pennies.market.market import Market

BASE_ASSET = (Asset, object)
BASE_MARKET = (Market, object)
BASE_STRING = string_types + (object,)


@dispatch(BASE_ASSET, BASE_MARKET, BASE_STRING)
def present_value(contract, market, reporting_ccy):
    """Base present value calculation.

    Given an asset (or sequence of assets), calculate it's present
    value as of today.  The supplied market provides prices, curves, and so on.

    Parameters
    ----------
    contract: Asset
        Asset to calculate present value
    market: Market
        Market to retrieve raw and computed market values
        such as prices, curves, and surfaces.
    reporting_ccy: str
        Specifies which currency to report value in

    Returns
    -------
    float
        Present Value in the reporting currency
    """
    raise NotImplementedError("Not available for base types")


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


def default_calculators():
    from pennies.trading import assets
    from pennies.calculators import payments
    return {
        str(assets.BulletPayment): payments.BulletPaymentCalculator,
        str(assets.DiscountBond): payments.BulletPaymentCalculator,
        str(assets.SettlementPayment): payments.BulletPaymentCalculator,
        str(assets.Zero): payments.BulletPaymentCalculator,
        str(assets.ZeroCouponBond): payments.BulletPaymentCalculator,
        str(assets.CompoundAsset): None,
        str(assets.Annuity): None,
        str(assets.FixedLeg): None,
        str(assets.IborLeg): None,
        str(assets.Swap): None,
        str(assets.VanillaSwap): None,
        str(assets.CurrencySwap): None,
        str(assets.TenorSwap): None,
        str(assets.Deposit): None,
        str(assets.StirFuture): None,
        str(assets.FRA): None,
        str(assets.IborFixing): None}
