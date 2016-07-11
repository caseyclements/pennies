""" In the Black model, we model the Forward, not the Spot, price of the underlying asset.
The numeraire is the zero bond, Z(O,T). and Present Value takes the form,
 PV = Z(0,T) * Black(F,K,T,vol)
The formulas here refer to what is called the 'Forward Value' = Present Value / Z(0,T).
This removes all discounting from the model.

Note on broadcasting. The caller is responsible for ensuring that shapes are unambiguous
Note on docstrings. Numeric types in general can be scalar or numpy ndarrays  
"""

from __future__ import division, print_function

import numpy as np
from scipy.stats import norm



def price(forward, strike, maturity, vol, isCall=True):
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
    isCall : bool
        True if option is a Call, False if a Put.

    Returns
    -------
    numeric ndarray
        Forward price of the European Option
    
    Examples
    --------
    >>> from pennies.models import black
    >>> f = np.array([100.0, 125.0, 150.0])
    >>> print(black.price(forward=f, strike=100, maturity=2.0, vol=0.2, isCall=False))
    [ 11.2462916    3.85331538   1.17090066]
    >>> vol = nd.array([0.2, 0.5])
    >>> print(black.price(forward=f[None,:], strike=100, maturity=2.0, vol=vol[:,None], isCall=False))
    [[ 11.2462916    3.85331538   1.17090066]
     [ 27.63263902  20.05140562  14.75322583]]
    """

    stddev = vol * np.sqrt(maturity)
    d1 = np.log(forward / strike) / stddev + 0.5 * stddev
    # replace nan's in common special case
    d1 = np.where(np.logical_and(np.isclose(forward, strike),
                                 np.isclose(stddev, 0.0)),
                                 np.inf, d1)
    d2 = d1 - stddev
    sign = np.where(isCall, 1, -1)
    
    return np.maximum(0.0, sign * (forward * norm.cdf(sign * d1) 
                            - strike * norm.cdf(sign * d2)))


def delta(forward, strike, maturity, vol, isCall=True):
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
    isCall : bool
        True if option is a Call, False if a Put.

    Returns
    -------
    numeric ndarray
        Forward price of the European Option

    Examples
    --------
    >>> from pennies.models import black
    >>> f = np.array([100.0, 125.0, 150.0])
    >>> print(black.delta(forward=f, strike=100, maturity=2.0, vol=0.2, isCall=False))
    [-0.         -0.17609419 -0.05763319]
    >>> vol = nd.array([0.2, 0.5])
    >>> print(black.delta(forward=f[None,:], strike=100, maturity=2.0, vol=vol[:,None], isCall=False))
    [[-0.         -0.17609419 -0.05763319]
     [-0.         -0.25170754 -0.17697167]]
    """

    stddev = vol * np.sqrt(maturity)
    d1 = np.log(forward / strike) / stddev + 0.5 * stddev
    d1 = np.where(np.isclose(forward, strike) + np.isclose(stddev, 0.0),
                    np.inf, d1) # handle common special case
    sign = np.where(isCall, 1, -1)
    
    return sign * norm.cdf(sign * d1) 
