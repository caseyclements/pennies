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
import pandas as pd
from scipy.interpolate import interp1d


class Curve(object):

    def __init__(self, dt_valuation=None):
        self.dt_valuation = dt_valuation


class DiscountCurve(Curve):
    """Base Yield Curve class for providing discount rates and bond prices."""

    def __init__(self, dt_valuation):
        super(DiscountCurve, self).__init__(dt_valuation)
        pass

    def __call__(self, date, *args, **kwargs):
        """Discount Factor implied by curve's term structure"""
        return self.discount_factor(date)

    def discount_factor(self, date):
        """Discount Factor implied by curve's term structure"""
        raise NotImplementedError("DiscountCurve is an abstract base class")


class ConstantDiscountRateCurve(DiscountCurve):
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
    daycount_fn: function
        Returns fractional years between dates
        according to providing time.day_count
    daycount_conv: str
        Returns fractional years between dates
        according to providing time._map_daycounts
    """
    def __init__(self, dt_valuation,
                 zero_rate=0.0,
                 daycount_conv='ACT365FIXED',
                 currency="USD"):
        super(ConstantDiscountRateCurve, self).__init__(dt_valuation)
        self.zero_rate = zero_rate
        self.daycount_fn = time._map_daycounts[daycount_conv]
        self.currency = currency
        """
        Parameters
        ----------
        dt_valuation: datetime.date
            Date on which the curve is valid, and reference date for x-values
        zero_rate: float
            Continuously-compounded zero (discount) interest rate
        daycount_conv: str
            Returns fractional years between dates
            according to providing time._map_daycounts
        """

    def discount_factor(self, dates):
        """Discount Factor implied by curve's term structure"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return np.exp(-ttm * self.zero_rate)

    def rate_sensitivity(self, dates):
        """ Sensitivity of the discount_factor to a unit change in the rate.

        In the simple case of a constant curve of continuously-compounded
        discount (zero) rates, d(PV)/dr = -(T-t) * PV(r, T-t)

        Notes
        -----
        This is the sensitivity to the discount rate, not the rate associated
        with any market par-rate.
        """
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return -ttm * np.exp(-ttm * self.zero_rate)


class DiscountCurveWithNodes(DiscountCurve):
    """ A Yield Curve defined by nodal points (time-to-maturity, rate).

    Rates/Prices between nodes are provided by interpolation

    Attributes
    ----------
    dt_valuation: datetime or date
        Date on which the curve is valid, and reference date for x-values
    node_dates: Series of date or datetime
        Dates on which rates are applicable.
    node_rates: Series of float
        Continuously-compounded zero (discount) interest rates.
        Note: Rates are fractional, not percentage. i.e. 0.005, not 5%
    daycount_fn: function
        Returns fractional years between dates
    node_ttm: Series of float
        Time to maturity daycount(dt_valuation, node_dates).
        Interpolation is performed in this space, instead of date space.
    """
    def __init__(self, dt_valuation, node_dates, node_rates,
                 daycount_conv='ACT365FIXED',
                 interpolator=interp1d, **interp_kwargs):
        """
        Parameters
        ----------
        dt_valuation: datetime
            Date on which curve is valid
        node_dates: Series or DateTimeIndex
            Maturity dates of the discount bonds implied by the curve
        node_rates: Series of
        daycount_conv: str
            Defines how dates are turned into times-to-maturities on curve.
            String to key in time._map_daycounts
        interpolator: function, optional
            A function that takes t,y nodes and returns a function of t.
            Examples are those in scipy.interpolate.
            If not provided, linear will be used without extrapolation.
        interp_kwargs: keyword args, optional
            Any additional kwargs will be passed as kwargs to interpolator.
        """
        super(DiscountCurveWithNodes, self).__init__(dt_valuation)
        self.node_dates = node_dates
        self.node_rates = node_rates
        self.daycount_fn = time.daycount(daycount_conv)
        self.node_ttm = self.daycount_fn(dt_valuation, node_dates)
        self.frame = pd.DataFrame({
            'ttm': self.node_ttm,  # TODO - Should this be index?
            'dt': node_dates,
            'rates': node_rates})

        self.rate_fn = interpolator(self.node_ttm, node_rates, **interp_kwargs)

    def discount_rates(self, dates):
        """Interpolated discount rates, ZeroCouponBond(ttm) = exp(-r * ttm)"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return self.rate_fn(ttm)

    def discount_factor(self, dates):
        """Discount Factor implied by curve's term structure"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return np.exp(-ttm * self.rate_fn(ttm))

    def rate_sensitivity(self, dates):
        """ Sensitivity of discount_factor(s) to a unit change in the rate(s).

        In the simple case of a constant curve of continuously-compounded
        discount (zero) rates, d(PV)/dr = -(T-t) * PV(r, T-t)

        Notes
        -----
        This is the sensitivity to the discount rate, not the rate associated
        with any market par-rate.

        Values are not scaled. For example, to get sens to 1bp, divide by 10000.
        """
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return -ttm * np.exp(-ttm * self.rate_fn(ttm))
