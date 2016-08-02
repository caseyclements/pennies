
"""
This is going to undergo a great deal of refactoring as I expand from a single
fixed cashflow to forward interest rates and then options.


We will begin with the most simple market that we can.
One that is composed strictly of a single curve providing discount factors.

Question. In all situations, must a market require a disount curve? YES

Most likely, Markets will be constructed by @classmethod

"""
from __future__ import absolute_import, division, print_function
from pennies.market.curves import Curve

class Market(object):
    """ Market base class.


    Attributes
    ----------
    dt_valuation: datetime
        Time at which market is valid
    """
    def __init__(self, dt_valuation):
        """
        Parameters
        ----------
        dt_valuation: datetime
            Time at which market is valid
        """
        self.dt_valuation = dt_valuation


class RatesTermStructure(Market):
    """Provider of required Interest Rates.

    This contains a coherent set of curves to provide discount and forward
    interest rates, along with associated discount factors, and meta data.
    Calibration should be such that only one RatesTermStructure

    Attributes
    ----------
    map_curves: dict of dict
        Contains a dictionary of dictionaries. The top key is currency strings.
        Within each currency dictionary, 'disount' must exist.
    """

    def __init__(self, dt_valuation, map_curves=None):
        super(RatesTermStructure, self).__init__(dt_valuation)
        self.map_curves = map_curves
        self.map_discount_curves = {ccy: self.map_curves[ccy]["discount"]
                                    for ccy in map_curves}

    def discount_curve(self, currency):
        return self.map_curves[currency]["discount"]

    def discount_factor(self, date, currency=None):
        if currency is None:
            if len(self.map_discount_curves) == 1:
                currency = list(self.map_discount_curves)[0]
            else:
                raise ValueError("currency argument must be defined if Market "
                                 "contains more than one of them.")
        return self.map_discount_curves[currency].discount_factor(date)

    def ibor_curve(self, currency, frequency) -> Curve:
        """ Return IBOR curve for requested frequency and currency"""
        return self.map_curves[currency][frequency]

    @classmethod
    def of_single_curve(cls, dt_valuation, yield_curve):
        """Create market consisting of a single discount curve, and a valid date

        If forward rates are to be computed, the discount curve will be used.
        """
        curve_map = {yield_curve.currency: {"discount": yield_curve}}
        return cls(dt_valuation, curve_map)

    @classmethod
    def from_curve_map(cls, dt_valuation, curve_map):
        """Create market consisting of a map of curves.

        The map is a dictionary of dictionaries.
        The top level keys are currency strings.
        The next level are strings describing the curves in that currency.
        One curve key in each currency must be named 'discount'.
        The others are frequencies defined as an integer number of months.

        These 'frequency' curves are used to produce forward ibor rates,
        from pseudo-discount factors.
        """
        return cls(dt_valuation, curve_map)
