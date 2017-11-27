from __future__ import division, print_function

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import pytest

from pennies.trading.assets import Asset, FixedLeg, IborLeg, Swap, VanillaSwap
from pennies.market.curves import ConstantDiscountRateCurve, DiscountCurveWithNodes
from pennies.market.interpolate import PiecewiseLinear
from pennies.market.market import Market, RatesTermStructure
from pennies.calculators import present_value
from pennies.calculators.swaps import ibor_rate, par_rate

dt_val = pd.to_datetime('today')
dt_settle = dt_val - pd.Timedelta(days=200)

length = 24  # months
frqncy = 6  # months
fixed_rate = 0.03
notional = 100
curr = "USD"
payment_lag = 2  # days

# 1. Create a Vanilla Swap
fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                frequency=frqncy, rate=fixed_rate,
                                notional=notional)
# Note that fixed rates are provided to IborLeg. In pricing, only those in the
# past will be used. Others will be replaced by their Forward rates.
float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                               frequency=frqncy, rate=fixed_rate,
                               notional=-notional, fixing_lag=0)

swap = VanillaSwap(fixed_leg, float_leg)

# 2. Create Market with Ibor and Discount Curves - both with Constant Rates
rate_discount = 0.05
crv_discount = ConstantDiscountRateCurve(
    dt_valuation=dt_val, zero_rate=rate_discount,
    daycount_conv='30360', currency=curr)

spread = 0.002
crv_disc = ConstantDiscountRateCurve(  # Dummy IBOR Curve
    dt_valuation=dt_val, zero_rate=rate_discount + spread,
    daycount_conv='30360', currency=curr)

curve_map = {fixed_leg.frame.currency.iloc[0]:
                 {'discount': crv_discount, frqncy: crv_disc}}
simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

# 3. Create Market with Ibor and Discount Curves - Nodal Curves
interpolator = PiecewiseLinear  # CubicSplineWithNodeSens, PiecewiseLinear
n_nodes = 8
node_dates = pd.date_range(dt_val, periods=n_nodes, freq=DateOffset(years=1))
node_rates = np.linspace(0.02, 0.05, num=n_nodes)
crv_disc = DiscountCurveWithNodes(dt_val, node_dates, node_rates,
                                  interpolator=interpolator,
                                  extrapolate=('clamped', 'natural'))
crv_fwd = DiscountCurveWithNodes(dt_val, node_dates, node_rates,
                                 interpolator=interpolator,
                                 extrapolate=('clamped', 'natural'))
nodal_curve_map = {curr: {'discount': crv_disc, frqncy: crv_fwd}}
nodal_rates_market = RatesTermStructure.from_curve_map(dt_val, nodal_curve_map)


'''Test behavior of multiple dispatched present_value.'''


def test_present_value_dispatch_catches_base_classes():
    """Test behavior of multipledispatched present_value."""
    with pytest.raises(NotImplementedError):
        pv = present_value(Asset(), Market(dt_val))

    with pytest.raises(NotImplementedError):
        pv = present_value(Asset(), simple_rates_market)

    with pytest.raises(NotImplementedError):
        pv = present_value(swap, Market(dt_val))
        assert False


def test_present_value_swap_equals_sum_of_legs():
    """Test dispatch works, and pv is self-consistent"""
    pv_fix = present_value(fixed_leg, simple_rates_market, curr)
    pv_flt = present_value(float_leg, simple_rates_market, curr)
    pv_swap = present_value(swap, simple_rates_market, curr)
    assert np.isclose(pv_swap, pv_fix + pv_flt)


def test_present_value_fixed_leg_at_par_equals_ibor_pv():
    swap_rate = par_rate(swap, simple_rates_market)
    df_fixed_at_market = fixed_leg.frame.copy()
    df_fixed_at_market['rate'] = swap_rate
    fixed_at_market = FixedLeg(df_fixed_at_market)
    pv_fix_at_market = present_value(fixed_at_market, simple_rates_market, curr)
    pv_flt = present_value(float_leg, simple_rates_market, curr)
    assert np.isclose(pv_fix_at_market, -pv_flt)


def test_present_value_fixed_leg_with_rates_equal_to_forwards_is_also_par():
    forwards = ibor_rate(float_leg, simple_rates_market)
    df_fixed_at_forwards = fixed_leg.frame.copy()
    unfixed = float_leg.frame.fixing > dt_val
    df_fixed_at_forwards.loc[unfixed, 'rate'] = forwards.loc[unfixed]
    fixed_at_forwards = FixedLeg(df_fixed_at_forwards)
    pv_fixed_at_forward = present_value(fixed_at_forwards, simple_rates_market, curr)
    pv_flt = present_value(float_leg, simple_rates_market, curr)
    assert np.isclose(pv_fixed_at_forward, -pv_flt)


def test_present_value_vanilla_ibor_leg_at_fixing_date_equals_notional():
    """Confirm spot starting Iborleg with Notional Exchange is worth par

    This is a canonical result of funding at ibor.
    Though there are two curves, both discount and ibor curves are equal.
    """

    # Constant Curves
    zero_spread = 0.0
    crv_ibor_no_spread = ConstantDiscountRateCurve(
        dt_valuation=dt_val, zero_rate=rate_discount + zero_spread,
        daycount_conv='30360', currency=curr)
    curves = {curr: {'discount': crv_discount, frqncy: crv_ibor_no_spread}}
    two_constant_curves = RatesTermStructure.from_curve_map(dt_val, curves)
    spot_starting = IborLeg.from_tenor(dt_settlement=dt_val, tenor=length,
                                   frequency=frqncy, rate=np.nan,
                                   notional=-notional, fixing_lag=0)
    pv_flt = present_value(spot_starting, two_constant_curves, curr)
    assert np.isclose(pv_flt, -notional)

    # Nodal Curves
    assert np.isclose(present_value(spot_starting,nodal_rates_market, curr),
                      -notional)


def test_present_value_of_swap_after_expiry():
    dt_settle = dt_val - pd.Timedelta(days=1000)
    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional)

    float_leg = IborLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                   frequency=frqncy, rate=fixed_rate,
                                   notional=-notional, fixing_lag=0)

    pv_fix = present_value(fixed_leg, simple_rates_market, curr)
    assert np.isclose(pv_fix, 0.0)
    pv_flt = present_value(float_leg, simple_rates_market, curr)
    assert np.isclose(pv_flt, 0.0)


def test_basisswap_pv_zero_if_onecurve_termstructure():
    notl = 1e8
    dt_start = dt_val
    leg6m = IborLeg.from_tenor(dt_start, length, frequency=6, notional=notl)
    leg3m = IborLeg.from_tenor(dt_start, length, frequency=3, notional=-notl)
    basis_swap = Swap(leg3m, leg6m)
    mkt_1crv = RatesTermStructure(dt_val, {curr: {'discount': crv_disc}})

    pv_6m_1crv = present_value(leg6m, mkt_1crv, curr)
    pv_3m_1crv = present_value(leg3m, mkt_1crv, curr)
    pv_1crv = present_value(basis_swap, mkt_1crv, curr)
    assert np.isclose(pv_6m_1crv, notl)
    assert np.isclose(pv_3m_1crv, -notl)
    assert np.isclose(pv_1crv, 0.0)

    spread = 0.005  # 5 basis point spread
    crv_6m = DiscountCurveWithNodes(dt_val, node_dates, node_rates + spread,
                                      interpolator=interpolator,
                                      extrapolate=('clamped', 'natural'))
    mkt_2crv = RatesTermStructure(dt_val, {curr: {'discount': crv_disc,
                                                  6: crv_6m}})
    pv_2crv = present_value(basis_swap, mkt_2crv, curr)
    assert not np.isclose(pv_2crv, 0.0)
