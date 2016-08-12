"""Calibrate a RatesTermStructure to a set of market instruments"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
import datetime as dt

from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure
from pennies.calculators.trades import present_value
from pennies.trading.assets import VanillaSwap, FixedLeg, IborLeg
from pennies.trading.trades import Portfolio


def calibrate_rates(curves_to_calibrate: RatesTermStructure,
                    target_portfolio: Portfolio):
    """Calibrate a map of curves that models market to a target set of prices.

    Parameters
    ----------
    curves_to_calibrate: RatesTermStructure
        Market model to calibrate, with initial guesses for par rates.
        For rates, the structure is a set of funding and forward curves,
        along with fx rates, if multiple currencies are required.
    target_portfolio: Portfolio
        The Target portfolio is a representation of all the instruments that
        one wishes to calibrate the curves to.
        These are not actual trades. They provide standard contracts and prices.

    Returns
    -------
    RatesTermStructure
        Same structure as input, with rates that re-price the target_portfolio.

    The core of a RatesTermStructure is a dt_valuation, and a dict of curves,
    map_curves. The is keyed off currency, then name, which must include
    'discount'. The others will typically be ibor frequencies: 3, 6, ..

    """
    # TODO Implement this function. This is here as a dummy to design UI
    return curves_to_calibrate


if __name__ == '__main__':
    '''
    The first thing I'd like to do is define n swaps and compute a single curve.

    If n = 1, we could compute a flat curve of ConstantRate.

    If n = 2, we could define 2 curves, 1 funding, 1 ibor

    If n = 3, we could compute 1 curve by linear oe cubic-spline interpolation

    We need to provide:

    -1. Valuation Date: dt_valuation

    -2. Market Prices: n contracts with par rates.
    ---- Present Value of each contract == 0

    -3. An interpolator, from scipy.interpolate, or similar, a fcn y=f(x)

    -4. The structure of the RatesTermStructure
    ---- RatesTermStructure with the following:
    ---- c currencies
    ---- c 'discount' curves
    ---- b ibor curves
    ---- c-1 fx rates
    ---- For each curve, m_i node dates, and guesses for (pseudo-)zero rates
    ---- -- Default to what? 0? 2?
    ---- total number of nodes

    ----------------------------------------


    --> n, # contracts must be == sum(m_i)


    '''

    # 1. Define Valuation Date
    dt_val = dt.datetime.now()

    # 2. Define the Market Prices to Calibrate to.
    # In our example, we create 3 Swaps, with an upward sloping yield

    dt_settle = dt_val
    frqncy = 6
    curr = "USD"
    durations = [12, 24, 60]
    fixed_rates = [0.03, 0.04, 0.05]  # These are the 'prices'
    notional = 100

    fixed_legs = []
    float_legs = []
    swaps = []

    for i in range(len(durations)):
        fixed_legs.append(
            FixedLeg.from_tenor(dt_settle, durations[i], frqncy, fixed_rates[i],
                                notional=notional, currency=curr))
        float_legs.append(
            IborLeg.from_tenor(dt_settle, durations[i], frqncy,
                               notional=-1 * notional, currency=curr))
        swaps.append(VanillaSwap(fixed_legs[i], float_legs[i]))

    market_portfolio = Portfolio.of_trades(swaps)

    # 3. Specify the interpolator
    # This takes an interpolator and does flat extrapolation,
    # i.e. derivatives are zero outside of boundaries
    def flat_extrap(interp):
        xs = interp.x
        ys = interp.y

        def new_interp(x):
            x_new = np.maximum(np.min(xs), np.minimum(np.max(xs), x))
            return interp(x_new)

        return new_interp

    interp_linear = lambda x, y: interp1d(x, y, kind='linear')
    in_linear_ex_flat = lambda x, y: flat_extrap(interp1d(x, y, kind='linear'))

    # 4. Create Market with 1 Nodal Curve
    rate_guess = 0.05
    node_dates = []
    for i in range(len(durations)):  # final pay date of each swap
        node_dates.append(fixed_legs[i].frame.pay.iloc[-1])
    node_dates = pd.Series(node_dates)
    node_rates = rate_guess * np.ones(len(node_dates))
    crv = DiscountCurveWithNodes(dt_val, node_dates, node_rates,
                                 interpolator=in_linear_ex_flat)
    curve_map = {curr: {'discount': crv}}
    simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # !!! Made DiscountCurve class callable... Do we want this?
    print(crv(dt_val))  # single date returns scalar
    print(crv(node_dates))  # Series returns series

    # Calibrate market model to prices
    calibrated_market = calibrate_rates(simple_rates_market, market_portfolio)

    # Test that it's worked by pricing swaps
    reporting_ccy = "USD"
    pv_port = present_value(market_portfolio, calibrated_market, reporting_ccy)
    print('pv of portfolio: {}'.format(pv_port))

    print('FIN')
