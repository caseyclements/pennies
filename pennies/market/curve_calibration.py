"""Calibrate a RatesTermStructure to a set of market instruments """

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy.optimize import root

from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure
from pennies.market.interpolate import PiecewiseLinear, CubicSplineWithNodeSens
from pennies.calculators.trades import present_value
from pennies.trading.assets import VanillaSwap, FixedLeg, IborLeg
from pennies.trading.trades import Portfolio
from pennies.calculators.swaps import sens_to_market_rates


def strip_of_swaps(dt_settlement, currency, tenor, maturities, rates):
    """Strip / List of Swaps of increasing length and a single currency and tenor

    These strips are used to calibrate nodes in discount and forward curves.
    If we are interested in pricing derivatives that depends on rates of two
    different frequencies (eg 3M & 12M), we will need one strip per frequency.
    Curve

    TODO Business date treatment
    TODO Add proper conventions
    """
    idx = np.arange(len(maturities))
    fixed_legs = [FixedLeg.from_tenor(dt_settlement, maturities[i], tenor,
                                      rates[i], currency=currency) for i in idx]
    float_legs = [IborLeg.from_tenor(dt_settlement, maturities[i], tenor,
                                     notional=-1.,currency=currency) for i in idx]
    swaps = [VanillaSwap(fixed_legs[i], float_legs[i]) for i in idx]
    return swaps


def calibrate_rates(rates_mkt,
                    target_portfolio,
                    rates_guess=None,
                    **root_kwargs):
    """Calibrate a map of curves that models market to a target set of prices.

    This is vector root finding problem to find N discount rates of curve nodes
    from M market contract prices. We form a Jacobian from the sensitivities of
    contract Present Values to curve rates.

    To create a coherent market model, we need to provide:
    -1. Valuation Date: dt_valuation
    -2. Market Prices: N contracts with par rates.
        - Present Value of each contract == 0
        - The vector of PV's forms the objective function in the calibration.
    -3. A Market Model, asset of curves and fx rates that price the market
        - RatesTermStructure with the following:
         C currencies
         C 'discount' curves
         B ibor curves
         C-1 fx rates
         M Curves:
         - M == C + B == N
         - For each curve..
           - m_i node dates, and guesses for (pseudo-)zero rates
           - An interpolator, providing rates given dates or year fractions
             - This also gives the sensitivity of an arbitrary point in time,
             - given a move of one of the nodes (a vector of length m_i)
             - See market.interpolate

    scipy.optimize.root is used to find the solution. It requires two inputs:

    1. fun. PV(x). Vector of length N
    2. x0 = rates, of length N.  This is all rates in the curves_to_calibrate
        x0 is the initial guess.

    And takes a number of optimal arguments.

    3. args. Additional arguments to fun and jac, beyond x
    4. jac. We produce a semi-analytic Jacobian, an (N x M) matrix,
        for Piecewise -Linear and -CubicTakes interpolators.
    5. Addl kwargs: method='hybr', tol=None, callback=None, options=None
        These are described in scipy.optimize.root docs

    Parameters
    ----------
    rates_mkt: RatesTermStructure
        Market model to calibrate, with initial guesses for par rates.
        For rates, the structure is a set of funding and forward curves,
        along with fx rates, if multiple currencies are required.
    target_portfolio: Portfolio
        The Target portfolio is a representation of all the instruments that
        one wishes to calibrate the curves to.
        These are not actual trades. They provide standard contracts and prices.
    rates_guess: array, optional
        Guess for discount rates. Either a 1D numpy array or a scalar.
        If an array length must be equal to sum of all nodes in curves.
    root_kwargs: dictionary, optional
        Keyword Args to be passed to scipy.optimize.root. eg method, options

    Returns
    -------
    scipy.optimize.OptimizeResult
        Note: rates_mkt is updated in place
    """

    def update_rates(rates, market):  # TODO Consider Moving this function to RatesTermStructure ==> market.update(rates)
        """Changes rates in current namespace's rates_mkt"""
        nodes = market.nodes
        nodes['rates'] = rates  # df exists to order rates
        # TODO Worried about the index here, with the fact that we keep reconstructing curves, but not market..
        # TODO - Is the order of nodes to ensure consistency while looping through a dict's items?
        for ccy, mkt in market.map_curves.items():
            for key, crv in mkt.items():
                assert isinstance(crv, DiscountCurveWithNodes)
                new_rates = nodes[(nodes.ccy == ccy) &
                                  (nodes.curve == key)]['rates'].values
                #mkt[key] = DiscountCurveWithNodes(crv.dt_valuation, crv.node_dates, new_rates, crv.daycount_fn, crv._interpolator.__class__, **crv._interp_kwargs)
                mkt[key].update_rates(new_rates)
                if key == 'discount':
                    market.map_discount_curves[ccy] = mkt[key]

    def vector_pv(rates, market, portfolio):
        """Present Value of each asset in target portfolio

        Objective function for root finder.
        """
        update_rates(rates, market)
        return [present_value(x, market, 'USD') for x in portfolio.trades]

    def pv_jacobian(rates, market, portfolio):
        """ Jacobian of sensitivities of each of the PV's to market moves

        For each asset in the target_portfolio, we compute the sensitivity
        of the price (present_value) to a unit move in each node
        in each of the market's curves."""

        update_rates(rates, market)
        sens = [sens_to_market_rates(contract, market, 'USD')
                for contract in portfolio.trades]
        return np.array(sens).T

    assert isinstance(rates_mkt, RatesTermStructure)
    assert isinstance(target_portfolio, Portfolio)
    if rates_guess is None:
        rates_guess = np.full(len(target_portfolio.trades), 0.02)

    return root(vector_pv, rates_guess, jac=pv_jacobian,
                args=(rates_mkt, target_portfolio),
                **root_kwargs)


if __name__ == '__main__':
    '''
    The first test is define N swaps and compute a single curve of N nodes.

    If N = 1, we could, in theory, compute a flat curve of ConstantRate.
    If N = 2, we could, in theory, define 2 constant curves: 1 funding, 1 ibor
    If N >= 3, we could compute 1 curve by piecewise linear
    If N >> 3, we can produce markets with additional complexity

    Questions
     - How many points do we need for cubic-spline interpolation?
    '''

    # 1. Define Valuation Date
    from pennies.time import daycounter
    dt_val = pd.to_datetime('today')

    # 2. Define the Market Prices to Calibrate to.
    # In our example, we create n_contracts Swaps, with an upward sloping yield
    dt_settle = dt_val
    curr = "USD"
    n_contracts = 6
    mkt_ids = np.arange(n_contracts)
    durations = 12 * (1 + mkt_ids)
    fixed_rates = 0.02 + np.log(1.0 + mkt_ids) / 50  # Effective market 'prices'
    frqncy = 6
    notional = 100
    dcc = '30360'

    fixed_legs, float_legs, swaps = [], [], []
    for i in mkt_ids:
        fixed_legs.append(
            FixedLeg.from_tenor(dt_settle, durations[i], frqncy, fixed_rates[i],
                                notional=notional, currency=curr))
        float_legs.append(
            IborLeg.from_tenor(dt_settle, durations[i], frqncy,
                               notional=-1 * notional, currency=curr))
        swaps.append(VanillaSwap(fixed_legs[i], float_legs[i]))

    market_portfolio = Portfolio.of_trades(swaps)

    # 3. Create Market. Specify number of curves to use, and the interpolators
    interpolator = CubicSplineWithNodeSens  # CubicSplineWithNodeSens, PiecewiseLinear
    n_curves = 2
    accrual_fn = daycounter(dcc)

    node_dates = []
    for i in range(n_contracts):  # final pay date of each swap
        node_dates.append(fixed_legs[i].frame.pay.iloc[-1])
    node_dates = pd.Series(node_dates)
    n = len(node_dates)
    rate_guess = 0.05
    node_rates = rate_guess / n * np.arange(1, n+1)
    print('input ttm: {}'.format(accrual_fn(dt_val, node_dates).values))
    print('market rates: {}'.format(fixed_rates))
    print('rate guess: {}'.format(node_rates))

    # 1 Curve for both discount and forward rates
    if n_curves == 1:
        crv_disc = DiscountCurveWithNodes(dt_val, node_dates,
                                          node_rates,
                                          interpolator=interpolator,
                                          extrapolate=('clamped', 'clamped'))

        curve_map = {curr: {'discount': crv_disc}}
        rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # 2 Nodal Curves: 1 discount, 1 ibor
    elif n_curves == 2:
        # Divide up nodes between two curves
        crv_disc = DiscountCurveWithNodes(dt_val, node_dates[::2],
                                          node_rates[::2],
                                          interpolator=interpolator, extrapolate=('clamped', 'clamped'))
        crv_ibor = DiscountCurveWithNodes(dt_val, node_dates[1::2],
                                          node_rates[1::2],
                                          interpolator=interpolator, extrapolate=('clamped', 'natural'))

        curve_map = {curr: {'discount': crv_disc, frqncy: crv_ibor}}
        rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # --------- Test Calibration Pieces -----------------
    rep_ccy = "USD"
    sens_float = sens_to_market_rates(float_legs[i],
                                      rates_market,
                                      rep_ccy)

    sens_fixed = sens_to_market_rates(fixed_legs[i],
                                      rates_market,
                                      rep_ccy)
    sens_swap = sens_to_market_rates(swaps[i],
                                     rates_market,
                                     rep_ccy)

    print('sens_fixed: {}'.format(sens_fixed))
    print('sens_float: {}'.format(sens_float))
    print('sens fixed+float: {}'.format(sens_float + sens_fixed))
    print('sens_swap: {}'.format(sens_swap))
    assert(np.allclose(sens_swap, sens_fixed + sens_float))

    # --------- Test Calibration -------------------------
    from time import time
    t = time()
    # Show the price of the target portfolio given the initial rate guess of 5%
    print('pv of portfolio before calibration: {}'.format(
        present_value(market_portfolio, rates_market, rep_ccy)))
    print('pv of each trade BEFORE: {}'.format(
        [present_value(trade, rates_market, rep_ccy)
         for trade in market_portfolio.trades]))

    # Calibrate market model to prices
    method = 'hybr'  # hybr, lm, broyden1, anderson
    options = None
    result = calibrate_rates(rates_market, market_portfolio, method=method, options=options)

    print(result)
    print('discount rates: {}'.format(
        rates_market.discount_curve('USD').node_rates))
    print('ibor rates: {}'.format(
        (rates_market.curve('USD', 6)[0]).node_rates))

    # Test that it's worked by pricing swaps
    print('pv of portfolio after calibration: {}'.format(
        present_value(market_portfolio, rates_market, rep_ccy)))
    print('pv of each trade AFTER: {}'.format(
        [present_value(trade, rates_market, rep_ccy)
         for trade in market_portfolio.trades]))

    print('time to run calibration, {}, contracts, {}, method, {}'.format(time() - t, n_contracts, method))
