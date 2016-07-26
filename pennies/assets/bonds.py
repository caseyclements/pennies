from __future__ import absolute_import, division, print_function
from pennies.assets.core import Asset

class Bond(Asset):

    def __init__(self, principal, start_date, end_date, frequency, coupon, currency='USD'):

        self.principal = principal
        self.start_date = start_date
        self.end_date = end_date
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

    def __init__(self, principal, start_date, end_date, coupon, currency='USD'):
        super(ZeroCouponBond, self).__init__(principal, start_date, end_date, 0, coupon, currency)
