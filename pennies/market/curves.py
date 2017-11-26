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
from pennies.market.interpolate import CubicSplineWithNodeSens, PiecewiseLinear


class Curve(object):

    def __init__(self, dt_valuation=None):
        self.dt_valuation = dt_valuation


class DiscountCurve(Curve):
    """Base Yield Curve class for providing discount rates and bond prices."""

    def __init__(self, dt_valuation):
        super(DiscountCurve, self).__init__(dt_valuation)
        pass

    def __call__(self, date):
        """Rates implied by curve's term structure"""
        raise NotImplementedError("DiscountCurve is an abstract base class")

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
    dt_valuation: date or datetime, preferably datetime64
        Date on which the curve is valid, and reference date for x-values
    zero_rate: float
        Continuously-compounded zero (discount) interest rate
    daycount_fn: function
        Returns fractional years between dates
        according to providing time.day_count
    daycount_conv: str
        Returns fractional years between dates
        according to providing time.daycount_conventions
    """
    def __init__(self, dt_valuation,
                 zero_rate=0.0,
                 daycount_conv='30360',
                 currency="USD"):
        super(ConstantDiscountRateCurve, self).__init__(dt_valuation)
        self.zero_rate = zero_rate
        self.daycount_fn = time.daycounter(daycount_conv)
        self.currency = currency
        """
        Parameters
        ----------
        dt_valuation: date or datetime, preferably datetime64
            Date on which the curve is valid, and reference date for x-values
        zero_rate: float
            Continuously-compounded zero (discount) interest rate
        daycount_conv: str
            Returns fractional years between dates
            according to providing time.daycount_conventions
        """

    def __call__(self, dates):
        return self.zero_rate

    def discount_factor(self, dates):
        """Discount Factor implied by curve's term structure"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return np.exp(-ttm * self.zero_rate)

    @staticmethod
    def rate_sensitivity(self, ttm):
        """Sensitivity of interpolated point to node: dy(x)/dy_i

        Sensitivity of rate at time, ttm, to a unit move in the rate of each
         node in the curve. As this curve has a constant flat rate, this simply
         returns 1.

        Parameters
        ----------
        ttm: array-like
            Time in years from valuation date to some rate's maturity date.

        Returns
        -------
        array-like
            shape = ttm.shape + curve.node_dates.shape
        """
        return np.ones_like(ttm)


# TODO - Consider caching discount factors
class DiscountCurveWithNodes(DiscountCurve):
    """ A Yield Curve defined by nodal points (time-to-maturity, rate).

    Rates/Prices between nodes are provided by interpolation

    Attributes
    ----------
    dt_valuation: date or datetime, preferably datetime64
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
                 daycount_conv='30360',
                 interpolator=PiecewiseLinear, **interp_kwargs):
        """
        Parameters
        ----------
        dt_valuation: date or datetime, preferably datetime64
            Date on which curve is valid
        node_dates: Series or DateTimeIndex
            Maturity dates of the discount bonds implied by the curve
        node_rates: Series of
        daycount_conv: str
            Defines how dates are turned into times-to-maturities on curve.
            String to key in time.daycount_conventions
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
        self.daycount_fn = time.daycounter(daycount_conv)
        self.node_ttm = self.daycount_fn(dt_valuation, node_dates)
        self.frame = pd.DataFrame({
            'ttm': self.node_ttm,
            'rates': node_rates,
            'dates': node_dates})

        self._interpolator = interpolator(np.array(self.node_ttm), node_rates,
                                          **interp_kwargs)
        self._interp_kwargs = interp_kwargs

    @classmethod
    def from_frame(cls, frame, dt_valuation, daycount_conv='30360',
                   interpolator=PiecewiseLinear, **interp_kwargs):
        node_dates = frame['dates']
        node_rates = frame['rates']
        return DiscountCurveWithNodes(dt_valuation, node_dates, node_rates,
          daycount_conv=daycount_conv, interpolator=interpolator, **interp_kwargs)

    def update_rates(self, rates):
        """Update rates, rebuild interpolator"""
        self.node_rates = rates
        self.frame.rates = rates
        self._interpolator.__init__(np.array(self.node_ttm), rates,
                                    **self._interp_kwargs)


    def rates_given_dates(self, dates):
        """Interpolated rates, r(ttm) = -log(ZeroCouponBond(ttm)) / ttm"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return self._interpolator(ttm)

    def rates_given_ttm(self, ttm):
        """Interpolated rates, r(ttm) = -log(ZeroCouponBond(ttm)) / ttm"""
        return self._interpolator(ttm)

    def discount_factor(self, dates):
        """Discount Factor implied by curve's term structure"""
        ttm = self.daycount_fn(self.dt_valuation, dates)
        return np.exp(-ttm * self._interpolator(ttm))

    def __call__(self, dates):
        """Return interpolated discount rates at given dates.

        r(ttm) = -log(ZeroCouponBond(ttm)) / ttm
        """
        return self.rates_given_dates(dates)

    def __str__(self):
            return str(self.frame)

    def rate_sensitivity(self, ttm):
        """Sensitivity of interpolated point to node: dy(x)/dy_i

        Sensitivity of rate at time, ttm, to a unit move in the rate of each
         node in the curve. Hence, for each ttm, a vector is returned.

        Parameters
        ----------
        ttm: array-like
            Time in years from valuation date to some rate's maturity date.

        Returns
        -------
        array-like
            shape = ttm.shape + curve.node_dates.shape
        """
        if (isinstance(self._interpolator, CubicSplineWithNodeSens) or
            isinstance(self._interpolator, PiecewiseLinear)):
            return self._interpolator.node_derivative(ttm)
        else:
            raise NotImplementedError(
                "Requires an interpolator with node_derivative method, "
                "such as PiecewiseLinear or CubicSplineWithNodeSens")
