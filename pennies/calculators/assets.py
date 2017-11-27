from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch
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


@dispatch(BASE_ASSET, BASE_MARKET, BASE_STRING, object, BASE_STRING)
def sens_to_zero_rates(contract, market, curve_ccy, curve_key, reporting_ccy):
    """Sensitivity of each cashflow to the curve specified by currency and key

    Given an asset contract (or sequence of assets), calculate the change in
     Present Value to a unit shift in the zero rate of the curve specified
     by curve_ccy and curve_key.

    Parameters
    ----------
    contract: Asset
        Calculate Sensitivity to this Asset.
    market: Market
        Market to retrieve raw and computed market values
        such as prices, curves, and surfaces.
    curve_ccy: str
        Specifies currency of the curve
    curve_key: str
        Along with curve_ccy, specifies which curve to compute sensitivity to.
    reporting_ccy: str
        Specifies which currency to report value in

    Returns
    -------
    DataFrame
        Table containing maturities, sensitivities, curve currency and key,
        columns=['ttm', 'sens', 'ccy', 'curve'].
    """
    raise NotImplementedError("Not available for base types")


@dispatch(BASE_ASSET, BASE_MARKET, str)
def sens_to_market_rates(contract, market, reporting_ccy):
    """Compute sensitivity of contract to each node in the market's curves.

    Market Curve Calibration consists of finding the discount rates for a
    desired set of curves that correctly prices a target set of contracts.

    To do this, we form a Jacobian matrix consisting of the sensitivities
     of each contract's Present Value, V_i,
     to each node of each curve in the market, r_j: d(V_i)/dr_j.

    This function produces a single row of the Jacobian. Note the indices:
    i = index over market contracts. (rows)
    j = index over model curve nodes, for all curves. (columns)
        Thus, if one had a discount curve with N nodes,
        and a single LIBOR curve with M nodes. j = 0..N+M-1

    V_i = Present Value of the i'th contract.
    r_j = Discount rate of node j.
    t_k = Maturity of some bond required to price V_i.
    r_{c,k} = discount rate of curve, c, at time t_k.
    z_{c,k} = discount bond price of curve, c, at time t_k.

    dP/dr_j = ( dV/dz_{c,k} ) * ( dz_{c,k} / dr_{c,k} ) * ( dr_{c,k} / dr_j )

    Note that dr_{c,k} / dr_j == 0 if node j does not belong to curve c.

    Parameters
    ----------
    contract: Asset
        Asset to calculate present value
    market: Market
        Market to retrieve raw and computed market values
        such as prices, curves, and surfaces.

    Returns
    -------
    Array of CurrencyAmount's
        Derivative of Present Value with respect to each node's rate
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

