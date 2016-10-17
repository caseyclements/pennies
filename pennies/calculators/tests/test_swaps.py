from __future__ import division, print_function

import datetime as dt
import numpy as np

from pennies.trading.assets import Asset, FixedLeg, IborLeg, VanillaSwap
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.market.market import Market, RatesTermStructure
from pennies.calculators.swaps import present_value, ibor_rate, par_rate

dt_val = dt.date.today()  # note: date
dt_settle = dt_val - dt.timedelta(days=200)
length = 24  # months
frqncy = 6  # months
fixed_rate = 0.03
notional = 100
curr = "USD"
payment_lag = 2  # days

# 1. Create a Swap
fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                frequency=frqncy, rate=fixed_rate,
                                notional=notional)
# Note that fixed rates are provided to IborLeg. In pricing, only those in the
# past will be used. Others will be replaced by their Forward rates.
float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                               frequency=frqncy, rate=fixed_rate,
                               notional=-1 * notional, fixing_lag=0)  # TODO Changed from 2

swap = VanillaSwap(fixed_leg, float_leg)

# 2. Create Market with Ibor and Discount Curves
rate_discount = 0.05
crv_discount = ConstantDiscountRateCurve(
    dt_valuation=dt_val, zero_rate=rate_discount,
    daycount_conv='ACT365FIXED', currency=curr)

spread = 0.002
crv_ibor = ConstantDiscountRateCurve(  # Dummy IBOR Curve
    dt_valuation=dt_val, zero_rate=rate_discount + spread,
    daycount_conv='ACT365FIXED', currency=curr)

curve_map = {fixed_leg.frame.currency.iloc[0]:
                 {'discount': crv_discount, frqncy: crv_ibor}}
simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)


def test_present_value():
    """Test behavior of multipledispatched present_value."""
    # Test dispatch catches base classes
    try:
        pv = present_value(Asset(), Market(dt_val))
        assert False
    except NotImplementedError:
        pass
    try:
        pv = present_value(Asset(), simple_rates_market)
        assert False
    except NotImplementedError:
        pass
    try:
        pv = present_value(swap, Market(dt_val))
        assert False
    except NotImplementedError:
        pass

    # Test dispatch works, and pv is self-consistent
    pv_fix = present_value(fixed_leg, simple_rates_market, curr)
    pv_flt = present_value(float_leg, simple_rates_market, curr)
    pv_swap = present_value(swap, simple_rates_market, curr)
    assert np.isclose(pv_swap, pv_fix + pv_flt)

    # Test present_value at fixed leg at par equals ibor pv
    swap_rate = par_rate(swap, simple_rates_market)
    df_fixed_at_market = fixed_leg.frame.copy()
    df_fixed_at_market['rate'] = swap_rate
    fixed_at_market = FixedLeg(df_fixed_at_market)
    pv_fix_at_market = present_value(fixed_at_market, simple_rates_market, curr)
    assert np.isclose(pv_fix_at_market, -1 * pv_flt)

    # Test pv of fixed leg with rates equal to forwards is also par
    forwards = ibor_rate(float_leg, simple_rates_market)
    df_fixed_at_forwards = fixed_leg.frame.copy()
    unfixed = float_leg.frame.fixing > dt_val
    df_fixed_at_forwards.loc[unfixed, 'rate'] = forwards.loc[unfixed]
    fixed_at_forwards = FixedLeg(df_fixed_at_forwards)
    pv_fixed_at_forward = present_value(fixed_at_forwards, simple_rates_market, curr)
    assert np.isclose(pv_fixed_at_forward, -1 * pv_flt)

if __name__ == '__main__':
    test_present_value()
