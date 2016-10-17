"""For now, playground while developing assets"""

import datetime as dt
import numpy as np
from pandas import DataFrame, Timedelta
from pennies.trading.assets import Annuity, FixedLeg, IborLeg, Swap, VanillaSwap

dt_settle = dt.date.today()
length = 24  # months
frqncy = 6  # months
fixed_rate = 0.03
notional = 100
payment_lag = 2  # days


def test_annuity_from_tenor():
    fixed = Annuity.from_tenor(dt_settle, tenor=length,
                               frequency=frqncy, rate=fixed_rate,
                               notional=notional, payment_lag=payment_lag)
    assert type(fixed) == Annuity
    df = fixed.frame
    assert type(df) == DataFrame
    cols = df.columns
    assert len(cols) == 13
    assert len(df) == 4


def test_fixedleg():
    """Test equality of various construction calls"""
    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional)

    # from_frame
    annuity_from_frame = Annuity.from_frame(fixed_leg.frame)
    assert annuity_from_frame != fixed_leg

    fixed_leg_from_frame = FixedLeg.from_frame(fixed_leg.frame)
    assert fixed_leg_from_frame == fixed_leg

    # init
    annuity_init = Annuity(fixed_leg.frame)
    assert annuity_init != fixed_leg

    fixed_leg_init = FixedLeg(fixed_leg.frame)
    assert fixed_leg_init == fixed_leg


def test_iborleg():
    """Test equality of various construction calls"""

    float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional, fixing_lag=2)
    # from_frame
    annuity_from_frame = Annuity.from_frame(float_leg.frame)
    assert annuity_from_frame != float_leg

    fixed_leg_from_frame = FixedLeg.from_frame(float_leg.frame)
    assert fixed_leg_from_frame != float_leg

    ibor_leg_from_frame = IborLeg.from_frame(float_leg.frame)
    assert ibor_leg_from_frame == float_leg

    # init
    annuity_init = Annuity(float_leg.frame)
    assert annuity_init != float_leg

    fixed_leg_init = FixedLeg(float_leg.frame)
    assert annuity_init != float_leg

    ibor_leg_init = IborLeg(float_leg.frame)
    assert ibor_leg_init == float_leg


def test_swap():
    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional)
    float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=-1 * notional, fixing_lag=2)
    swap = Swap(fixed_leg, float_leg)
    assert swap.leg_receive == fixed_leg
    assert swap.leg_pay == float_leg


def test_fixing_lag():
    """Ensure lag of 2 provides minimum of 2 days lag.

    This is under default business day adjustment.
    """

    # Todo - what is the default business day adjustment???
    leg_no_lag = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=-1 * notional, fixing_lag=0)

    leg_lag = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                 frequency=frqncy, rate=fixed_rate,
                                 notional=-1 * notional, fixing_lag=2)

    assert np.all(leg_lag.frame['fixing'] >
                  leg_no_lag.frame['fixing'] + Timedelta(days=1))



def test_vanilla_swap():
    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional, currency='EUR')
    float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                   frequency=frqncy, rate=fixed_rate,
                                   notional=-1 * notional, currency='EUR')

    swap = VanillaSwap(fixed_leg, float_leg)
    assert swap.leg_receive == fixed_leg
    assert swap.leg_pay == float_leg

if __name__ == '__main__':
    #test_annuity_from_tenor()
    #test_fixedleg()
    #test_iborleg()
    #test_swap()
    test_fixing_lag()
    #test_vanilla_swap()
