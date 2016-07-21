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


class Annuity(object):
    """Fixed Rate Annuity.

    This is used as the fixed leg of a Swap, as the core of fixed rate Bonds,
    and the natural numeraire when pricing swaptions

    """

    # TODO This will remain incomplete, in the sense that it only captures
    # TODO the Vanilla case. For example, it does not handle stubs.

    # TODO Need to define set of conventions for daycount calculations
    # Todo Need to define set of conventions business day adjustments

    def __init__(self, dt_settlement, duration, frequency, receive=True,
                 daycount=None, payment_lag=0, busday='Following', stub='Front',
                 currency='USD', fixed_rate=1.0, notional=1.0,):
        """
        dt_payment: datetime
            Date (and time) on which amount is received
        currency : str, optional
            Currency code of amount received
        notional: float, optional
            Notional, provides a scaling factor on prices. Received if positive, else paid.
        """

        # Use inputs to create the required vectors
        self.payment_dts = None
        self.year_fractions
        self.currency = currency
        self.notional = notional

        @classmethod
        def from_schedules():
            """Compute from formed, or bespoke payment date schedule

            Probably also require year_fractions
            """
            pass


class IborLeg(object):
    """Series of coupons based on fixings of an IBOR.

        IBOR = Inter-Bank Offered Rate, eg 3M USD LIBOR (3-month dollar Libor)
        Used as Floating Leg of a Swap or Floating Rate Note.
    """
    pass

    @classmethod
    def from_schedules(cls):
        """Compute from formed, or bespoke date schedules

        Probably also require year_fractions
        """
        pass


class Swap(object):
    def __init__(self, annuity, floating_leg, swap_rate, notional):
        """Probably need more here"""
        pass


    @classmethod
    def vanilla(cls):
        """From typical inputs"""
        pass

    @classmethod
    def vanilla_from_conventions(cls):
        """From typical inputs, and conventions table"""
        raise NotImplementedError


class FRA(object):
    """Forward Rate Agreement"""

    def __init__(self, fixed_rate, dt_fixing, dt_payment,
                 dt_accrual_start=None, dt_accrual_end=None,
                daycount=None, notional=1.0,  pay_upfront=True):
        raise NotImplementedError


class StirFuture(object):
    """Short term interest rate Future"""
    def __init__(self):
        raise NotImplementedError


class Deposit(object):
    """Short term cash deposit paying simple (not compounded) interest"""

    def __init__(self):
        raise NotImplementedError


class IborFixing(object):
    """Current Fixing of an Inter-Bank Offered Rate

        Used to calibrate yield curves. Not an asset per-se.
    """

    def __init__(self):
        raise NotImplementedError

class TenorSwap(object):
    """Swap with two floating legs, each of different rate tenors"""

    def __init__(self):
        raise NotImplementedError

class CurrencySwap(object):
    """Swap with two floating legs, each of different currencies

    and often rate and payment frequency"""

    def __init__(self):
        raise NotImplementedError

    
















