"""Tests calibration for single and multi curve setups.
Tests implicitly the price jacobian for Cubic Spline Interpolation."""

from pennies.time import daycounter
from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure
from pennies.market.interpolate import PiecewiseLinear, CubicSplineWithNodeSens
from pennies.calculators.trades import present_value
from pennies.trading.trades import Portfolio
from pennies.market.curve_calibration import calibrate_rates, strip_of_swaps

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset


# GENERIC SETUP
# TODO Expand coverage by turning these into fixtures
# 1. Define the Market Prices to Calibrate to.
# In our example, we create n Swaps, (or 2n) with upward sloping yield
dt_val = pd.to_datetime(np.datetime64('today'))  # Valuation Date
dt_settle = dt_val  # Spot starting without typical spot starting lag
ccy = "USD"
n_nodes = 10    
idx = np.arange(n_nodes)
maturities = 12 * (1 + idx)  # annual, in months
notional = 100
dcc = '30360'

# The fixed rate acts as the effective market 'price'
freq3 = 3  # Quarterly
fixed_rates_quarterly = 0.02 + np.log(1.0 + idx) / 50  # Pays ibor quarterly
freq12 = 12  # Annually
spread_3s12s = 0.002  # 200bp more expensive to fund at 12M than 3M
fixed_rates_annual = fixed_rates_quarterly + spread_3s12s  # Pays ibor annually

# 2. Create Model for the Term Structure.
# Specify number of curves to use, and the interpolators
accrual_fn = daycounter(dcc)
node_dates = [dt_val + DateOffset(months=maturities[i]) for i in range(n_nodes)]
rates_guess = 0.05 / n_nodes * np.arange(1, n_nodes + 1)


@pytest.fixture(params=[PiecewiseLinear, CubicSplineWithNodeSens])
def interpolator(request):
    return request.param


@pytest.fixture(params=[3, 12])
def frequency(request):
    return request.param


def test_one_curve(frequency, interpolator):
    """1 Nodal Curve: 1 discount, used to produce discount and forward rates"""
    # Create the swap strip (target prices)
    market_portfolio = Portfolio.of_trades(
        strip_of_swaps(dt_settle, ccy, frequency, maturities, fixed_rates_annual))

    # Create market model / term structure
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
    assert np.isclose(pv_portfolio, 0.0), ('Non-Zero PV after calibration is {}'
                                           .format(pv_portfolio))
    pv_trades = [present_value(trade, rates_market, ccy)
                 for trade in market_portfolio.trades]
    assert np.allclose(pv_trades, 0.0), "PVs after calibration are non-zero"


def test_two_curve(interpolator):
    """Calibrate Term Structure to strips of swaps of 2 frequencies

    The primary curve is used to produce forward rates for 1st frequency
    AND discount rates. This is typically chosen to be the smaller frequency.
    The secondary curve is used to produce forward rates for the 2nd frequency.
    """

    # Targets are 2 strips, Swaps paying quarterly, and ones paying annually
    market_portfolio_3s12s = Portfolio.of_trades(
        strip_of_swaps(dt_settle, ccy, freq3, maturities, fixed_rates_quarterly) +
        strip_of_swaps(dt_settle, ccy, freq12, maturities, fixed_rates_annual))

    # Create TermStructure model
    crv1 = DiscountCurveWithNodes(dt_val, node_dates, rates_guess,
                                  interpolator=interpolator,
                                  extrapolate=('clamped', 'clamped'))

    crv2 = DiscountCurveWithNodes(dt_val, node_dates, rates_guess,
                                  interpolator=interpolator,
                                  extrapolate=('clamped', 'natural'))

    curve_map = {ccy: {'discount': crv1, freq12: crv2}}

    rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # Calibrate market model to prices
    result = calibrate_rates(rates_market, market_portfolio_3s12s)
    assert result.success

    # Test that it's worked by pricing swaps
    pv_portfolio = present_value(market_portfolio_3s12s, rates_market, ccy)
    assert np.isclose(pv_portfolio, 0.0), ('PV of market portfolio after '
                                           'calibration is non-zero')
    pv_trades = [present_value(trade, rates_market, ccy)
                 for trade in market_portfolio_3s12s.trades]
    assert np.allclose(pv_trades, 0.0), ("PVs of market portfolio's trades "
                                         "after calibration are non-zero")
