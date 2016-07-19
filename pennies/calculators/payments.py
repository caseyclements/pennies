from __future__ import absolute_import, division, print_function

from pennies.trading.assets import BulletPayment
from .assets import AssetCalculator
from pennies.core import CurrencyAmount


class BulletPaymentCalculator(AssetCalculator):
    """Calculator for BulletPayments and its aliases."""

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
        return CurrencyAmount(self.contract.amount * df, self.contract.currency)
