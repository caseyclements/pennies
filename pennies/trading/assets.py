"""Assets represent Financial Assets, these are claims based on a contract.

These are often referred to as Securities or Products in other libraries.
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
from pandas import DataFrame
from numbers import Number

from pandas.tseries.offsets import DateOffset, CustomBusinessDay
from pennies.time import daycounter


RATETYPES = ['FIXED', 'IBOR']


class Asset(object):
    """Base class of all Financial Assets"""

    def __init__(self):
        self.frame = pd.DataFrame()

    def accept(self, visitor, *args, **kwargs):
        """Accepts visitors that calculate various measures on the Asset.

        Stub if we wish to use visitor pattern.
        Currently using multipledispatch
        """
        return visitor.visit(Asset, *args, **kwargs)

    def __eq__(self, other):
        return self.frame.equals(other.frame)


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


class Annuity(Asset):
    """Fixed Rate Annuity.

    This is used as the fixed leg of a Swap, as the core of fixed rate Bonds,
    and the natural Numeraire when pricing Swaptions.

    The primary representation of the asset is a dataframe where each
    row is a single cashflow.
    """

    # TODO Capture additional cases outlined below
    # TODO Stubs: Short and Long,  Front and Back
    # TODO Daycount conventions: add more
    # TODO Business day adjustments conventions: add more
    # TODO Holiday calendars: add

    def __init__(self, df, notl_exchange=True):
        """Create Annuity from DataFrame.

        Not meant to be the primary constructor.
        Instead, calls like Annuity.from_tenor will be more common.
        This is here because classmethods must return a call to constructor
        so that return type is known.

        Parameters
        ----------
        df: DataFrame
            Required columns =  ['start','end', 'pay', 'fixing',
        'period', 'frequency', 'notional', 'dcc','lag_pay', 'bday_adj', 'stub']
        notl_exchange: bool
            If true, notional is paid at the final pay date
        """
        super(Annuity, self).__init__()
        # Primary representation
        self.frame = df
        # Scalar Metadata
        self.notl_exchange = notl_exchange
        try:
            vals = set(df.currency)
            assert len(vals) == 1, ('currency column should have just one '
                                    'value: Found {}'.format(vals))
            self.currency = vals.pop()
        except KeyError:
            print('Required key, currency, not contained in frame')
            raise
        try:
            vals = set(df.frequency)
            assert len(vals) == 1, ('frequency column should have just one '
                                    'value: Found {}'.format(vals))
            self.frequency = vals.pop()
        except KeyError:
            print('Required key, frequency, not contained in frame')
            raise
        try:
            vals = set(df.type)
            assert len(vals) == 1, ('type column should have just one '
                                    'value: Found {}'.format(vals))
            self.type = vals.pop()
        except KeyError:
            print('Optional key, type, not contained in frame. Set to None')
            self.type = None

    @classmethod
    def from_tenor(cls, dt_settlement, tenor, frequency, rate=1.0, dcc=None,
                   notional=1.0, currency='USD', receive=True, payment_lag=0,
                   bday=None, stub='front', notl_exchange=True,
                   rate_type='FIXED'):
        """Construct a fixed rate Annuity from start date, length and frequency.

        Parameters
        ----------
        dt_settlement: datetime
            Date (and time) on which leg begins to accrue interest
        tenor: int
            Length of the entire leg, as number of months
        frequency: int
            Number of months between cash flows
        dcc: str, optional
            Daycount Convention for computing accrual of interest
        rate: float, optional
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
        notl_exchange: bool
            If true, notional is paid at the final pay date
        rate_type: str, optional
            Defines whether the rate being paid is fixed, or of some floating
            index such as an IBOR.
        """
        # TODO: Check behavior when stubs implied
        dt_maturity = dt_settlement + DateOffset(months=tenor)
        period = DateOffset(months=frequency)
        sched_end = pd.date_range(dt_settlement, dt_maturity,
                                  freq=period, closed='right')
        sched_start = sched_end - period  # TODO Test stub cases. start[i] should be end[i-1]
        if bday or payment_lag:
            sched_pay = sched_end + CustomBusinessDay(payment_lag, holidays=None)
        else:
            sched_pay = sched_end

        # Primary representation of leg as Pandas DataFrame
        assert rate_type in RATETYPES
        frame = pd.DataFrame({
            'start': sched_start,
            'end': sched_end,
            'pay': sched_pay,
            'rate': rate,
            'notional': notional,
            'frequency': frequency,
            'currency': currency,
            'dcc': dcc,
            'pay_lag': payment_lag,
            'bday_adj': bday,
            'stub': stub,
            'type': rate_type})
        year_frac = daycounter(dcc)(frame.start, frame.end)
        frame['period'] = year_frac

        return Annuity(frame, notl_exchange=notl_exchange)

    @classmethod
    def from_frame(cls, df, notl_exchange=True):
        return Annuity(df, notl_exchange=notl_exchange)

    def __str__(self):
        return str(self.frame)

    def __eq__(self, other):
        return (isinstance(other, Annuity) and
               super(Annuity, self).__eq__(other))


class FixedLeg(Annuity):

    def __init__(self, df, fixed_rate=None, notl_exchange=True):
        super(FixedLeg, self).__init__(df, notl_exchange=notl_exchange)
        self.type = 'FIXED'
        self.frame['type'] = self.type
        if fixed_rate:
            self.frame['rate'] = fixed_rate

    def __eq__(self, other):
        return (isinstance(other, FixedLeg) and
                super(FixedLeg, self).__eq__(other))

    @classmethod
    def from_tenor(cls, dt_settlement, tenor, frequency, rate=1.0, dcc='30360',
                   notional=1.0, currency='USD', receive=True, payment_lag=0,
                   bday=None, stub='front', notl_exchange=True):
        annuity = Annuity.from_tenor(dt_settlement, tenor, frequency, rate,
                                     dcc, notional, currency, receive,
                                     payment_lag, bday, stub, notl_exchange,
                                     rate_type='FIXED')
        if isinstance(rate, Number):
            return FixedLeg(annuity.frame, fixed_rate=rate)
        else:
            raise NotImplementedError("FixedLeg requires scalar rate.")

    @classmethod
    def from_frame(cls, df, fixed_rate=1.0, notl_exchange=True):
        return FixedLeg(df, fixed_rate=fixed_rate, notl_exchange=notl_exchange)


class IborLeg(Annuity):
    """Series of coupons based on fixings of an IBOR.

        IBOR = Inter-Bank Offered Rate, eg 3M USD LIBOR (3-month dollar Libor)
        Used as Floating Leg of a Swap or Floating Rate Note.
    """
    def __init__(self, df, notl_exchange=True):
        """Compute from DataFrame.

        This is unlikely to be the primary constructor, but classmethods must
        return a call to constructor so that type is known.

        Parameters
        ----------
        df: DataFrame
        Required columns =  ['start','end', 'pay', 'fixing',
        'period', 'frequency', 'notional', 'dcc','lag_pay', 'bday_adj', 'stub']
        """
        # Primary representation
        super(IborLeg, self).__init__(df, notl_exchange=notl_exchange)
        self.type = 'IBOR'
        self.frame['rate_type'] = self.type

    @classmethod
    def from_tenor(cls, dt_settlement, tenor, frequency, rate=None, dcc=None,
                   notional=1.0, currency='USD', receive=True, payment_lag=0,
                   fixing_lag=0, bday=None, stub='front', notl_exchange=True):

        annuity = Annuity.from_tenor(dt_settlement, tenor, frequency, rate,
                                     dcc, notional, currency, receive,
                                     payment_lag, bday, stub, notl_exchange,
                                     rate_type='IBOR')

        df = annuity.frame
        if bday or fixing_lag:
            df['fixing'] = df['start'] + CustomBusinessDay(fixing_lag, holidays=None)
        else:
            df['fixing'] = df['start']
        return IborLeg(df, notl_exchange=notl_exchange)

    @classmethod
    def from_frame(cls, df, notl_exchange=True):
        return IborLeg(df, notl_exchange=notl_exchange)

    def __eq__(self, other):
        return (isinstance(other, IborLeg) and
                super(IborLeg, self).__eq__(other))


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


class Swap(CompoundAsset):
    def __init__(self, receive_leg, pay_leg):
        """ This takes two frames"""
        self.underlying_contracts = [receive_leg, pay_leg]
        self.leg_receive = receive_leg
        self.leg_pay = pay_leg

    def __eq__(self, other):
        return (isinstance(other, Swap) and
                self.leg_pay == other.leg_pay and
                self.leg_receive == other.leg_receive)

    def __str__(self):
        return ('\nPay Leg:\n' + str(self.leg_pay) +
                '\nReceive Leg:\n' + str(self.leg_receive))


class VanillaSwap(Swap):
    def __init__(self, fixed_leg, floating_leg):
        assert isinstance(fixed_leg, FixedLeg)
        assert isinstance(floating_leg, IborLeg)
        assert fixed_leg.currency == floating_leg.currency, \
            'Currencies differ in legs of VanillaSwap'
        assert fixed_leg.type == 'FIXED'
        self.leg_fixed = fixed_leg
        assert floating_leg.type == 'IBOR'
        self.leg_float = floating_leg
        initial_notl_fixed = fixed_leg.frame.notional.iloc[0]
        initial_notl_float = floating_leg.frame.notional.iloc[0]
        if initial_notl_fixed * initial_notl_float > 0.0:
            raise ValueError("Notional values of both legs have same sign")
        elif initial_notl_fixed >= 0.0:
            super(VanillaSwap, self).__init__(receive_leg=fixed_leg,
                                              pay_leg=floating_leg)
        else:
            super(VanillaSwap, self).__init__(receive_leg=floating_leg,
                                              pay_leg=fixed_leg)

    def __eq__(self, other):
        return (isinstance(other, VanillaSwap) and
                self.leg_fixed == other.leg_fixed and
                self.leg_float == other.leg_float)


class FRA(Asset):
    """Forward Rate Agreement"""

    def __init__(self, fixed_rate, dt_fixing, dt_payment,
                 dt_accrual_start=None, dt_accrual_end=None,
                daycount=None, notional=1.0,  pay_upfront=True):
        raise NotImplementedError


class StirFuture(Asset):
    """Short term interest rate Future"""
    def __init__(self):
        raise NotImplementedError


class Deposit(Asset):
    """Short term cash deposit paying simple (not compounded) interest"""

    def __init__(self):
        raise NotImplementedError


class IborFixing(Asset):
    """Current Fixing of an Inter-Bank Offered Rate

        Used to calibrate yield curves. Not an asset per-se.
    """

    def __init__(self):
        raise NotImplementedError


class TenorSwap(Swap):
    """Swap with two floating legs, each of different rate tenors"""

    def __init__(Asset):
        raise NotImplementedError


class CurrencySwap(Swap):
    """Swap with two floating legs, each of different currencies

    and often rate and payment frequency"""

    def __init__(self):
        raise NotImplementedError

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