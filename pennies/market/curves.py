"""Yield and Forward Curves, with a focus on Interest Rates.

    Notes
    -----
    The design of Curves has only just begun.

    To reproduce a realistic set of the interest rate products in a given
    currency, one has to produce a coherent set of curves. In the method that
    I plan to use, this will take the form of one 'discount' curve per currency,
    and then some number of forward rate curves. Others may be used to capture
    other spreads.

    Questions
    ---------

    X-values - Do we use dates, or year-fractions? ==> Year fractions, for rates

    Do we assume nodal forms? If so...
     how do we allow for general interpolation / extrapolation?

    Yield curves, assume calibrated continuously-compounded zero rates?

    How much hierarchy do we put in? YieldCurve, DiscountRate, DiscountFactor,
     Constant, Forward, Equity, ...

    I would really like all of this to be vectorized

"""
from __future__ import division, print_function

from pennies import time
import numpy as np


class Curve(object):

    def __init__(self, dt_valuation=None):
        self.dt_valuation = dt_valuation


class ConstantDiscountRateCurve(Curve):
    """ Curve consisting of a constant continuously-compounded discount rate.

    This means that the price of a $1 at date T is exp(-r * (T-t))
     where t is dt_valuation, and '-' represents the years between t and T
     according to the curve's daycount convention.

    Attributes
    ----------
    dt_valuation: datetime.date
        Date on which the curve is valid, and reference date for x-values
    zero_rate: float
        Continuously-compounded zero (discount) interest rate
    daycount: function
        Returns fractional years between dates
        according to providing time.day_count
    """
    def __init__(self, dt_valuation,
                 zero_rate=0.0,
                 daycount_function=time.daycount('Act/365 Fixed'),
                 currency="USD"):
        super(ConstantDiscountRateCurve, self).__init__(dt_valuation)
        self.zero_rate = zero_rate
        self.daycount = daycount_function
        self.currency = currency

    def discount_factor(self, date):
        """Discount Factor implied by curve's term structure"""
        t = self.daycount(self.dt_valuation, date)
        return np.exp(-t * self.zero_rate)

    def rate_sensitivity(self, date):
        """ Sensitivity of the discount_factor to a unit change in the rate.

        In the simple case of a constant curve of continuously-compounded
        discount (zero) rates, d(PV)/dr = -(T-t) * PV(r, T-t)

        Notes
        -----
        This is the sensitivity to the discount rate, not the rate associated
        with any market par-rate.
        """
        t = self.daycount(self.dt_valuation, date)
        return -t * np.exp(-t * self.zero_rate)
