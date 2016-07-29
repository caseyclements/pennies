"""Assets represent Financial Assets, these are claims based on a contract.

These are often referred to as Securities or Products in other libraries.
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
from pandas.tseries.offsets import DateOffset, CustomBusinessDay
from pennies.time import daycount


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
    notional: float
        Notional in given currency. Received if positive, else paid.
        notional: float
        Notional in given currency. Received if positive, else paid.
    """

    def __init__(self, dt_payment, currency='USD', notional=1.0, bday=None):
        """
        Parameters
        ----------
        dt_payment: datetime
            Date (and time) on which notional is received
        currency : str, optional
            Currency code of notional received
        notional: float, optional
            Currency Amount. Received if positive, else paid.
        bday: str, optional
            Rule to adjust dates that fall on weekends and holidays.
        """
        super(ZeroCouponBond, self).__init__()
        self.dt_payment = dt_payment
        self.currency = currency
        self.notional = notional

        self.frame = pd.DataFrame({
            'pay': dt_payment,
            'notional': notional,
            'currency': currency},
            index=[0])

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
        underlying_contracts: list of Asset's
        """
        super(CompoundAsset, self).__init__()
        self.underlying_contracts = underlying_contracts


class Annuity(object):
    """Fixed Rate Annuity.

    This is used as the fixed leg of a Swap, as the core of fixed rate Bonds,
    and the natural Numeraire when pricing Swaptions.

    The primary representation of the asset is a dataframe where each
    row is a single cashflow.
    """

    # TODO This will remain incomplete, in the sense that it only captures
    # TODO the Vanilla case. For example, it does not handle stubs.

    # TODO Need to define set of conventions for daycount calculations
    # TODO Need to define set of conventions business day adjustments
    # TODO Need to add holiday calendars

    def __init__(self, dt_settlement, duration, frequency, dcc=None,
                 fixed_rate=1.0, notional=1.0, currency='USD', receive=True,
                 payment_lag=0, bday=None, stub='front'):
        """Construct a fixed rate Annuity from dates and frequency.

        Parameters
        ----------
        dt_settlement: datetime
            Date (and time) on which leg begins to accrue interest
        duration: int
            Length of the entire leg, (currently) as number of months
        frequency: int
            Number of months between cash flows
        dcc: str, optional
            Daycount Convention for computing accrual of interest
        fixed_rate: float, optional
            Rate of interest accrual. Simple accrual, no compounding in period.
        notional: float, optional
            Notional amount. Received if positive, else paid.
        currency : str, optional
            Currency code of amount received
        receive: bool, optional
            Alternative method of specifying sign of notional.
            Multiplies given notional by -1 if False
        payment_lag: int, optional
            Number of days after accrual end dates that payments are made.
        bday: str, optional
            Rule to adjust dates that fall on weekends and holidays.
        stub: str, optional
            If schedule building leads to one period of different length,
            this decides if it is the first ('front') or last period ('back').
        """
        # TODO: Check behavior when stubs implied
        super(Annuity, self).__init__()
        dt_maturity = dt_settlement + DateOffset(months=duration)
        period = DateOffset(months=frequency)
        sched_end = pd.date_range(dt_settlement, dt_maturity,
                                  freq=period, closed='right')
        sched_start = sched_end - period  # starts of accrual periods
        sched_pay = sched_end + CustomBusinessDay(payment_lag, holidays=None)
        fn_accrue = daycount(dcc)
        year_frac = fn_accrue(sched_start, sched_pay)

        # primary representation of leg as pandas dataframe
        self.frame = pd.DataFrame({
            'start': sched_start,
            'end': sched_end,
            'pay': sched_pay,
            'period': year_frac,
            'rate': fixed_rate,
            'notional': notional,
            'currency': currency,
            'dcc': dcc,
            'pay_lag': payment_lag,
            'bday_adj': bday,
            'stub': stub})
        # TODO - Finish this and add tests

        @classmethod
        def from_frame(dataframe):
            """Compute from dataframe

            Parameters
            ----------
            Required columns =  ['start','end', 'pay', 'period',
            'rate', 'notional', 'dcc','lag_pay', 'bday_adj']
            """
            self.frame = dataframe


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
