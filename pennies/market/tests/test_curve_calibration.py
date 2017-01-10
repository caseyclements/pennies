import numpy as np
import pandas as pd
import pytest
from copy import deepcopy


from datetime import datetime
from pennies.time import normalize_date, daycounter

from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure
from pennies.market.interpolate import PiecewiseLinear
from pennies.calculators.trades import present_value
from pennies.calculators.swaps import par_rate
from pennies.trading.assets import VanillaSwap, FixedLeg, IborLeg
from pennies.trading.trades import Portfolio
from pennies.market.curve_calibration import calibrate_rates

# GENERIC SETUP  # TODO Turn this into a pytest fixture

# 1. Define the Market Prices to Calibrate to.
# In our example, we create n_contracts Swaps, with an upward sloping yield
dt_val = normalize_date(datetime.now())  # Valuation Date
dt_settle = dt_val  # Spot starting without typical n business day lag
ccy = "USD"
n_contracts = 4
mkt_ids = np.arange(n_contracts)
durations = 12 * (1 + mkt_ids)
fixed_rates = 0.02 + np.log(1.0 + mkt_ids) / 50  # Effective market 'prices'
frqncy = 12
notional = 100
dcc = '30360'

fixed_legs, float_legs, swaps = [], [], []
for i in mkt_ids:
    fixed_legs.append(
        FixedLeg.from_tenor(dt_settle, durations[i], frqncy, fixed_rates[i],
                            notional=notional, currency=ccy))
    float_legs.append(
        IborLeg.from_tenor(dt_settle, durations[i], frqncy,
                           notional=-1 * notional, currency=ccy))
    swaps.append(VanillaSwap(fixed_legs[i], float_legs[i]))

market_portfolio = Portfolio.of_trades(swaps)

# 2. Create Market Model. Specify number of curves to use, and the interpolators
interpolator = PiecewiseLinear  # CubicSplineWithNodeSens, PiecewiseLinear
n_curves = 1
accrual_fn = daycounter(dcc)

node_dates = []
for i in range(n_contracts):  # final pay date of each swap
    node_dates.append(fixed_legs[i].frame.pay.iloc[-1])
node_dates = pd.Series(node_dates)
n = len(node_dates)
rates_guess = 0.05
rates_guess = rates_guess / n * np.arange(1, n + 1)


def test_one_curve():
    """1 Nodal Curve: 1 discount, used to produce discount and forward rates"""
    # Create market
    crv_disc = DiscountCurveWithNodes(dt_val, node_dates,
                                      rates_guess,
                                      interpolator=interpolator,
                                      extrapolate=('clamped', 'clamped'))

    curve_map = {ccy: {'discount': crv_disc}}
    rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # Calibrate market model to prices
    result = calibrate_rates(rates_market, market_portfolio)
    assert result.success

    # Test that it's worked by pricing swaps
    pv_portfolio = present_value(market_portfolio, rates_market, ccy)
    assert np.isclose(pv_portfolio, 0.0), 'PV after calibration is non-zero, {}'.format(pv_portfolio)
    pv_trades = [present_value(trade, rates_market, ccy) for trade in market_portfolio.trades]
    assert np.allclose(pv_trades, 0.0), "PVs after calibration are non-zero"


def test_two_curve_starting_from_solution():
    """Confirm calibration stops immediately when rate guess is the solution

    2 Nodal Curves: 1 discount, 1 ibor

    Set discount rates of two curves
    Create market from two-curves of arbitrary discount rates
    Compute par rates of market (target) swaps, given known curves.
    Set target market prices equal to the par rates calculated
    Calibrate market to the market prices starting from set discount rates.
    This should already be at solution.
    """

    # Set arbitrary curve rates
    rates_discount = rates_guess[::2]
    rates_ibor = rates_guess[1::2]
    dts_discount = node_dates[::2]
    dts_ibor = node_dates[1::2]

    # Create TermStructure
    crv_disc = DiscountCurveWithNodes(dt_val, dts_discount, rates_discount,
                                      interpolator=PiecewiseLinear,
                                      extrapolate=('clamped', 'clamped'))

    crv_ibor = DiscountCurveWithNodes(dt_val, dts_ibor, rates_ibor,
                                      interpolator=PiecewiseLinear,
                                      extrapolate=('clamped', 'natural'))

    curve_map = {ccy: {'discount': crv_disc, frqncy: crv_ibor}}
    rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # Create Market 'at Par' with TermStructure
    swaps = []
    for trade in market_portfolio.trades:
        assert isinstance(trade, VanillaSwap)
        swap_rate = par_rate(trade, rates_market)
        fixed_leg = FixedLeg.from_frame(trade.leg_fixed.frame, fixed_rate=swap_rate)
        swaps.append(VanillaSwap(fixed_leg, trade.leg_float))
    target_portfolio = Portfolio.of_trades(swaps)

    # Calibrate TermStructure to market
    new_curves = deepcopy(rates_market)
    result = calibrate_rates(new_curves, target_portfolio)
    assert result.success

    # Test that it's worked by pricing swaps
    pv_portfolio = present_value(market_portfolio, new_curves, ccy)
    assert np.allclose(pv_portfolio, 0.0), \
        'PV of market portfolio after calibration is non-zero'
    pv_trades = ([present_value(trade, new_curves, ccy)
                  for trade in market_portfolio.trades])
    assert np.allclose(pv_trades, 0.0), \
        "PVs of market portfolio's trades after calibration are non-zero"


def test_two_curve():
    """2 Nodal Curves: 1 discount, 1 ibor"""

    # Create market
    crv_disc = DiscountCurveWithNodes(dt_val, node_dates[::2],
                                      rates_guess[::2],
                                      interpolator=PiecewiseLinear,
                                      extrapolate=('clamped', 'clamped'))

    crv_ibor = DiscountCurveWithNodes(dt_val, node_dates[1::2],
                                      rates_guess[1::2],
                                      interpolator=PiecewiseLinear,
                                      extrapolate=('clamped', 'natural'))

    curve_map = {ccy: {'discount': crv_disc, frqncy: crv_ibor}}
    rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # Calibrate market model to prices
    result = calibrate_rates(rates_market, market_portfolio)
    assert result.success
    print(result)

    # Test that it's worked by pricing swaps
    pv_portfolio = present_value(market_portfolio, rates_market, ccy)
    assert np.isclose(pv_portfolio, 0.0), 'PV of market portfolio after calibration are non-zero'
    pv_trades = [present_value(trade, rates_market, ccy) for trade in market_portfolio.trades]
    assert np.allclose(pv_trades, 0.0), "PVs of market portfolio's trades after calibration are non-zero"


@pytest.mark.skip("Not implemented yet")
def test_one_curve_annuities_only():
    """Test calibration to spot starting Annuities

    These annuities pay notional at settlement, and receive it at maturity.
    """
    assert False

@pytest.mark.skip("Not implemented yet")
def test_two_curve_annuities_only():
    """Test calibration to spot starting Annuities

    These annuities pay notional at settlement, and receive it at maturity.
    """
    assert False


if __name__ == '__main__':
    #test_one_curve()
    #test_two_curve_starting_from_solution()
    test_two_curve()



