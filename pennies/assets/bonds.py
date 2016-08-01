from __future__ import absolute_import, division, print_function

import pandas as pd
from pennies.assets.core import Asset


class Bond(Asset):
    """Base bond class.

    Base class for bonds to store attributes relevant to most bonds.
    """

    def __init__(self, principal, start_date, maturity, frequency, coupon, currency='USD'):

        self.principal = principal
        self.start_date = pd.to_datetime(start_date)
        self.maturity = maturity
        self.frequency = frequency
        self.coupon = coupon
        self.currency = currency


class ZeroCouponBond(Bond):
    """A single payment of an amount of currency on a given date.

    This has a number of aliases:  ZCB, Zero, DiscountBond, Bullet
    By default, the amount is $1 received.

    Attributes
    ----------
    dt_payment: datetime
        Date (and time) on which amount is received
    currency : str
        Currency code of amount received
    amount: float
        Currency Amount. Received if positive, else paid.
    """

    def __init__(self, principal, start_date, maturity, coupon, currency='USD'):
        super(ZeroCouponBond, self).__init__(principal, start_date, maturity, None, coupon, currency)
