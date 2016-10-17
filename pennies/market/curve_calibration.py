"""Calibrate a RatesTermStructure to a set of market instruments """

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from pennies.market.interpolate import CubicSplineWithNodeSens
import datetime as dt

from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure
from pennies.calculators.trades import present_value
from pennies.trading.assets import VanillaSwap, FixedLeg, IborLeg
from pennies.trading.trades import Portfolio
from pennies.calculators.swaps import sens_to_market_rates


def calibrate_rates(curves_to_calibrate: RatesTermStructure,
                    target_portfolio: Portfolio):
    """Calibrate a map of curves that models market to a target set of prices.

    This is vector root finding problem to find N discount rates of curve nodes
    from M market contract prices. We form a Jacobian from the sensitivities of
    contract Present Values to curve rates.

    scipy.optimize.root is used to find the solution.

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
    """
    # TODO Implement this function. This is here as a dummy to design UI
    return curves_to_calibrate

    # fun takes ndarray x as arguments, and any additional arguments can be passed in args
    # => vector PV(x) (as market PVs are 0).
    # ==> Will have to generalize for instruments with non-zero initial PV
    # => Use lambda x: f(x,y,z) to turn into correct form
    # jac take same x,args as fun
    # method's have different options


    # Need a function that takes x and returns PV(contracts(x)), both of length n


    sol = scipy.optimize.root(fun, x0, args=(), method='hybr',
                              jac=None, tol=None, callback=None, options=None)




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
    curr = "USD"
    fixed_legs, float_legs, swaps = [], [], []
    n_contracts = 6
    mkt_ids = np.arange(n_contracts)
    durations = 12 * (1 + mkt_ids)
    fixed_rates = [0.03, 0.04, 0.05]
    fixed_rates = 0.02 + np.log(1.0 + mkt_ids) / 50  # These are the 'prices'
    frqncy = 6
    notional = 100

    for i in mkt_ids:
        fixed_legs.append(
            FixedLeg.from_tenor(dt_settle, durations[i], frqncy, fixed_rates[i],
                                notional=notional, currency=curr))
        float_legs.append(
            IborLeg.from_tenor(dt_settle, durations[i], frqncy,
                               notional=-1 * notional, currency=curr))
        swaps.append(VanillaSwap(fixed_legs[i], float_legs[i]))

    market_portfolio = Portfolio.of_trades(swaps)

    # 3. Specify the interpolator

    # FOR TESTING PURPOSES
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
    interpolator = in_linear_ex_flat

    # MORE REALISTIC: CubicSplineWithNodeSens
    interpolator = CubicSplineWithNodeSens

    # 4. Create Market with 2 Nodal Curves: 1 discount, 1 ibor
    rate_guess = 0.05
    node_dates = []
    for i in range(len(durations)):  # final pay date of each swap
        node_dates.append(fixed_legs[i].frame.pay.iloc[-1])
    node_dates = pd.Series(node_dates)
    node_rates = rate_guess * np.ones(len(node_dates))
    # Divide up nodes between two curves
    crv_disc = DiscountCurveWithNodes(dt_val, node_dates[::2],
                                      node_rates[::2],
                                      interpolator=interpolator)
    crv_ibor = DiscountCurveWithNodes(dt_val, node_dates[1::2],
                                      node_rates[1::2],
                                      interpolator=interpolator)

    curve_map = {curr: {'discount': crv_disc, frqncy: crv_ibor}}
    rates_market_2crv = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # !!! Made DiscountCurve class callable... Do we want this?
    print('disc ttm: {}'.format(crv_disc.node_ttm))
    print('ibor ttm: {}'.format(crv_disc.node_ttm))  # Series returns series
    # single date returns scalar. Also checking extrapolation
    print('rate at dt_val: {}'.format(crv_disc(dt_val)))

    # --------- Test Calibration Pieces -----------------
    reporting_ccy = "USD"

    sens_float = sens_to_market_rates(float_legs[i],
                                               rates_market_2crv,
                                               reporting_ccy)

    sens_fixed = sens_to_market_rates(fixed_legs[i],
                                               rates_market_2crv,
                                               reporting_ccy)
    sens_swap = sens_to_market_rates(swaps[i],
                                              rates_market_2crv,
                                              reporting_ccy)

    print('sens_fixed: {}'.format(sens_fixed))
    print('sens_float: {}'.format(sens_float))
    print('sens fixed+float: {}'.format(sens_float + sens_fixed))
    print('sens_swap: {}'.format(sens_swap))
    assert(np.allclose(sens_swap, sens_fixed + sens_float))

    # ----------------------------------------------------

    # Calibrate market model to prices
    calibrated_market = calibrate_rates(rates_market_2crv, market_portfolio)

    # Test that it's worked by pricing swaps
    reporting_ccy = "USD"
    pv_port = present_value(market_portfolio, calibrated_market, reporting_ccy)
    print('pv of portfolio: {}'.format(pv_port))

    print('FIN')


