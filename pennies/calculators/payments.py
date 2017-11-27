from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch

from pennies.trading.assets import BulletPayment
from pennies.calculators.assets import AssetCalculator
from pennies import CurrencyAmount
from pennies.market.market import Market, RatesTermStructure


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


class BulletPaymentCalculator(AssetCalculator):
    """(Deprecated) Calculator for BulletPayments and its aliases."""

    measures = {  # TODO Complete this. Consider visitor pattern for ccrs
        "present_value",
        #"pv01",
        #"position",
        #"cashflow_dates"  # Should this be here?
    }

    def __init__(self, contract, market):
        """
        Parameters
        ----------
        contract: BulletPayment
            Asset (or Trade) representing the payment of a fixed amount
        market: Market
            Market values required to price the Asset. Here, a DiscountCurve
        """
        super(BulletPaymentCalculator, self).__init__(contract, market)
        # TODO As it stands, init isn't needed as it just calls super
        # TODO It is here as a reminder to refactor if market gets specific

    def present_value(self):
        """Present, or Fair, Value of a known BulletPayment."""
        df = self.market.discount_factor(self.contract.dt_payment,
                                         self.contract.currency)
        return CurrencyAmount(self.contract.notional * df, self.contract.currency)
