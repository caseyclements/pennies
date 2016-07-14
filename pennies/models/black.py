"""Black model, of the Forward, not the Spot, price of the underlying asset.

Present Value takes the form, PV = Z(0,T) * Black(F,K,T,vol)
where Z(O,T) is a zero coupon bond maturing on the same date as the option.
'price' refers to what is called the 'Forward Value' = Present Value / Z(0,T).
This removes discounting from the model. Any exception are noted in docstrings.

Note on broadcasting. The caller is responsible for ensuring matching shapes.
Note on docstrings. Any numeric input may be a scalar or a numpy.ndarray
"""

from __future__ import division, print_function

import numpy as np
from scipy.stats import norm
from scipy.optimize import root


def price(forward, strike, maturity, vol, is_call=True):
    """The Forward Price of a European option in the Black model
    
    Parameters
    ----------
    forward : numeric ndarray
        Forward price or rate of the underlying asset
    strike : numeric ndarray
        Strike price of the option
    maturity : numeric ndarray
        Time to maturity of the option, expressed in years
    vol : numeric ndarray
        Lognormal (Black) volatility
    is_call : bool, optional
        True if option is a Call, False if a Put.

    Returns
    -------
    numeric ndarray
        Forward price of the European Option
    
    Examples
    --------
    >>> from pennies.models import black
    >>> f = np.array([100.0, 125.0, 150.0])
    >>> print(black.price(forward=f, strike=100, maturity=2.0, vol=0.2))
    [ 11.2462916   28.85331538  51.17090066]
    >>> vol = np.array([0.2, 0.5])
    >>> print(black.price(forward=f[None,:], strike=100, maturity=2.0, \
                vol=vol[:,None], is_call=False))
    [[ 11.2462916    3.85331538   1.17090066]
     [ 27.63263902  20.05140562  14.75322583]]
    """

    stddev = vol * np.sqrt(maturity)
    sign = np.where(is_call, 1, -1)
    d1 = _d1(forward, strike, stddev)
    d2 = d1 - stddev
    return np.maximum(0.0, sign * (forward * norm.cdf(sign * d1) -
                                   strike * norm.cdf(sign * d2)))


def _d1(forward, strike, sigma_root_t):
    d1 = np.log(forward / strike) / sigma_root_t + 0.5 * sigma_root_t
    # replace nan's in common special case: f==k & sigma*sqrt(t) == 0
    atm = np.isclose(forward, strike)
    if np.any(atm):
        d1 = np.where(np.logical_and(atm, np.isclose(sigma_root_t, 0.0)),
                      np.inf, d1)
    return d1


def _d1_using_mask(forward, strike, sigma_root_t):
    """" d1 of Black Formula using mask instead of where"""
    d1 = np.log(forward / strike) / sigma_root_t + 0.5 * sigma_root_t
    mask = np.isclose(forward, strike)
    if np.any(mask):  # replace nan's in special case
        mask = np.logical_and(mask, np.isclose(sigma_root_t, 0))
        d1 = np.array(d1, copy=False)  # necessary to handle scalar
        d1[(mask)] = np.inf
    return d1


def delta(forward, strike, maturity, vol, is_call=True):
    """The Forward (Driftless) Delta of an option in the Black model
    
    Parameters
    ----------
    forward : numeric ndarray
        Forward price or rate of the underlying asset
    strike : numeric ndarray
        Strike price of the option
    maturity : numeric ndarray
        Time to maturity of the option, expressed in years
    vol : numeric ndarray
        Lognormal (Black) volatility
    is_call : bool, optional
        True if option is a Call, False if a Put.

    Returns
    -------
    numeric ndarray
        Forward price of the European Option

    Examples
    --------
    >>> from pennies.models import black
    >>> f = np.array([100.0, 125.0, 150.0])
    >>> print(black.delta(forward=f, strike=100, maturity=2.0, vol=0.2))
    [ 0.55623146  0.82390581  0.94236681]
    >>> vol = np.array([0.2, 0.5])
    >>> print(black.delta(forward=f[None,:], strike=100, maturity=2.0, \
                vol=vol[:,None], is_call=False))
    [[-0.44376854 -0.17609419 -0.05763319]
     [-0.3618368  -0.25170754 -0.17697167]]
    """

    sign = np.where(is_call, 1, -1)
    d1 = _d1(forward, strike, vol * np.sqrt(maturity))
    return sign * norm.cdf(sign * d1)


def strike_from_delta(forward, fwd_delta, maturity, vol, is_call=True):
    """Return absolute value of strike price, given delta

    In many lines of business, the strike price is specified as the Option's Delta.
    This function converts that back to a strike in price-space for use in models.
    """

    variance = vol ** 2 * maturity
    sign = np.where(is_call, 1, -1)
    d1 = sign * norm.ppf(sign * fwd_delta)
    return forward * np.exp(-d1 * np.sqrt(variance) + 0.5 * variance)


def vega(forward, strike, maturity, vol):
    """The forward vega of a European Option.

    Forward Vega = Spot Vega / Zero Coupon Bond with maturity date of Option.

    See docstring of price for description of Parameters
    """

    d1 = _d1(forward, strike, vol * np.sqrt(maturity))
    return forward * np.sqrt(maturity) * norm.pdf(d1)


def theta_forward(forward, strike, maturity, vol, is_call=True):
    """The Forward Theta of a European option in the Black model

    This is also described as the Driftless Theta

    See docstring of price for description of Parameters
    """

    root_t = np.sqrt(maturity)
    d1 = _d1(forward, strike, vol * root_t)
    return -0.5 * forward * norm.pdf(d1) * vol / root_t


def theta_spot(forward, strike, maturity, vol, is_call=True, interest_rate=0.0):
    """Spot Theta, the sensitivity of the present value to a change in time to maturity.

     Unlike most of the methods in this module, this requires an interest rate,
     as we are dealing with option price with respect to the Spot underlying

    Parameters
    ----------
    forward : numeric ndarray
        Forward price or rate of the underlying asset
    strike : numeric ndarray
        Strike price of the option
    maturity : numeric ndarray
        Time to maturity of the option, expressed in years
    vol : numeric ndarray
        Lognormal (Black) volatility
    is_call : bool, optional
        True if option is a Call, False if a Put.
    interest_rate : numeric ndarray, optional
        Continuously compounded interest rate, i.e. Zero = Z(0,T) = exp(-rT)

    """

    zero = np.exp(-interest_rate * maturity)
    stddev = vol * np.sqrt(maturity)
    sign = np.where(is_call, 1, -1)
    d1 = _d1(forward, strike, stddev)
    d2 = d1 - stddev
    price_ish = sign * (forward * norm.cdf(sign * d1) -
                        strike * zero * norm.cdf(sign * d2))
    theta_fwd = theta_forward(forward, strike, maturity, vol, is_call)
    return theta_fwd + interest_rate * price_ish


def implied_vol_otm_scalars(pv, f, k, t, call, vol_guess, **kwargs):
    """Lognormal volatility that reproduces the target price.

    Option described should be out-of-the-money (OTM) for good convergence.
    All arguments are assumed to be scalars.
    """
    fn_price = lambda vol: pv - price(f, k, t, vol, call)
    fn_vega = lambda vol: vega(f, k, t, vol)
    solution = root(fun=fn_price, x0=vol_guess, jac=fn_vega, **kwargs)
    if solution['success']:
        return solution['x']
    else:
        raise ValueError(solution['message'])

implied_vol_otm_arrays = np.vectorize(implied_vol_otm_scalars,
                            doc='vectorized version of implied_vol_otm_scalars')


# TODO: Add available kwargs to docstring
def implied_vol(forward_price, forward, strike, maturity,
                is_call=True, vol_guess=0.2, **kwargs):
    """Lognormal volatility that reproduces the target, price

    Find the log-normal (Black) implied volatility
    of an out-the-money European option starting from an initial guess

    Parameters
    ----------
    forward_price : numeric ndarray
        The Present Value (PV) of the option scaled by Zero Coupon Bond.
        Same as that returned by black.price()
    forward : numeric ndarray
        Forward price or rate of the underlying asset
    strike : numeric ndarray
        Strike price of the option
    maturity : numeric ndarray
        Time to maturity of the option, expressed in years
    is_call : bool, optional
        True if option is a Call, False if a Put.

    Returns
    -------
    numeric ndarray
        Lognormal volatility that reproduces forward_price in Black formula
    """

    # Solver performs better if we use out-of-the-money option prices
    sign = np.where(is_call, 1, -1)
    price_intrinsic = sign * np.maximum(0.0, sign * (forward - strike))
    price_otm = forward_price - price_intrinsic
    otm_is_call = np.where(strike >= forward, True, False)
    return implied_vol_otm_arrays(price_otm, forward, strike, maturity,
                                  otm_is_call, vol_guess, **kwargs)
