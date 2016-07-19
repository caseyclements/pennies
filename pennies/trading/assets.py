"""Assets represent Financial Assets, these are claims based on a contract.

These are often referred to as Securities or Products in other libraries.
"""

from __future__ import absolute_import, division, print_function


class Asset(object):
    """Base class of all Financial Assets"""

    def __init__(self):
        pass

    # TODO - Ask whether the Visitor Pattern is a good idea in Python
    def accept(self, visitor, *args, **kwargs):
        """Accepts visitors that calculate various measures on the Asset"""
        return visitor.visit(Asset, *args, **kwargs)


def all_assets():
    """Provides a list of all available Assets"""
    return Asset.__subclasses__()


class ZeroCouponBond(Asset):
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

    def __init__(self, dt_payment, currency='USD', amount=1.0):
        """
        dt_payment: datetime
            Date (and time) on which amount is received
        currency : str, optional
            Currency code of amount received
        amount: float, optional
            Currency Amount. Received if positive, else paid.
        """
        self.dt_payment = dt_payment
        self.currency = currency
        self.amount = amount


# TODO: Check whether this sort of aliasing is a good idea
Zero = ZeroCouponBond
"""Alias for a ZeroCouponBond"""
ZCB = ZeroCouponBond
"""Alias for a ZeroCouponBond"""
DiscountBond = ZeroCouponBond
"""Alias for a ZeroCouponBond"""
BulletPayment = ZeroCouponBond
"""Alias for a ZeroCouponBond"""
SettlementPayment = ZeroCouponBond
"""BulletPayment used to settle trades"""


class CompoundAsset(Asset):
    """This Asset is composed of a list of Assets.

    This is a convenient way to structure a bespoke trade that contains
    numerous parts, like embedded options, or different first coupons.

    Attributes
    ----------

    underlying_contracts: list
        List of instances of Assets
    """

    def __init__(self, underlying_contracts):
        """
        Parameters
        ----------
        underlying_contracts: list of Asset
        """
        self.underlying_contracts = underlying_contracts





