from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
from pandas import DataFrame
from pennies.market.curves import Curve, DiscountCurveWithNodes


class Market(object):
    """ Market base class.


    Attributes
    ----------
    dt_valuation: date or datetime, preferably datetime64
        Time at which market is valid
    """
    def __init__(self, dt_valuation):
        """
        Parameters
        ----------
        dt_valuation: date or datetime, preferably datetime64
            Time at which market is valid
        """
        self.dt_valuation = dt_valuation


class RatesTermStructure(Market):
    """Provider of required Interest Rates.

    This contains a coherent set of curves to provide discount and forward
    interest rates, along with associated discount factors, and meta data.

    Calibration generally is performed to reprice N market assets,
    where N is the total of all nodes in all the curves.

     The core of a RatesTermStructure is a dt_valuation, and a dict of curves,
    map_curves. The is keyed off currency, then name, which must include
    'discount'. The others will typically be ibor frequencies: 3, 6, ..

    Attributes
    ----------
    map_curves: dict of dict
        Contains a dictionary of dictionaries. The top key is currency strings.
        Within each currency dictionary, 'discount' must exist.
    """

    def __init__(self, dt_valuation, map_curves=None, map_fx=None):
        super(RatesTermStructure, self).__init__(dt_valuation)
        self.map_curves = map_curves
        self.map_discount_curves = {ccy: self.map_curves[ccy]["discount"]
                                    for ccy in map_curves}
        self.map_fx = map_fx

        self.nodes = DataFrame(columns=['ttm', 'dates', 'rates', 'ccy', 'curve'])
        for ccy, curves in self.map_curves.items():
            df_ccy = DataFrame(columns=['curve', 'ttm', 'dates', 'rates'])
            for key, crv in curves.items():
                try:
                    df_crv = crv.frame.loc[:, ['ttm', 'dates', 'rates']]
                    df_crv['ccy'] = ccy
                    df_crv['curve'] = key
                    if key == 'discount':
                        df_ccy = pd.concat([df_crv, df_ccy], ignore_index=True)
                    else:
                        df_ccy = pd.concat([df_ccy, df_crv], ignore_index=True)
                except AttributeError:
                    pass  # Curves without nodes will not be calibrated
            self.nodes = pd.concat([self.nodes, df_ccy], ignore_index=True)
        #self.nodes.sort_values(by=['ccy', 'ttm'], inplace=True)
        # TODO - What's the point of if key == blah, if we sort? Why are we sorting?

    @classmethod
    def from_frame(cls, dt_valuation, frame, map_fx=None):
        map_curves = {}
        for ccy in np.unique(frame.ccy):
            mp_ccy = {}
            df_ccy = frame.loc[frame.ccy == ccy]
            for key in np.unique(df_ccy.curve):
                mp_ccy[key] = DiscountCurveWithNodes.from_frame(
                    df_ccy[df_ccy.curve == key]['dates', 'rates'],
                    dt_valuation)
            map_curves[ccy] = mp_ccy
        return RatesTermStructure(dt_valuation, map_curves, map_fx)

    def discount_curve(self, currency):
        """Access to discount curve for given currency"""
        return self.map_curves[currency]["discount"]

    def discount_factor(self, date, currency=None):
        if currency is None:
            if len(self.map_discount_curves) == 1:
                currency = list(self.map_discount_curves)[0]
            else:
                raise ValueError("currency argument must be defined if Market "
                                 "contains more than one of them.")
        return self.map_discount_curves[currency].discount_factor(date)

    def curve(self, currency, key):
        """ Provides curve and its market key that provide rates for input

        Parameters
        ----------
        currency: str
            Currency of the rates required.
        key:
            Key used to describe rate.
            For IBOR, this will be the integer frequency in months

        Returns
        -------
        tuple
            (curve, key) curve in market that produces rates for key requested
        """
        try:
            ccy_map = self.map_curves[currency]
        except KeyError:
            raise ValueError('Requested currency not present in market: {}'
                             .format(currency))
        try:
            return ccy_map[key], key
        except KeyError:
            try:
                return ccy_map['discount'], 'discount'
            except KeyError:
                raise KeyError('Curve requested with key,{}, and currency ,{}, '
                               'that has neither that key, nor "discount". '
                               'Perhaps the currency is wrong.'
                               .format(key, currency))

    def fx(self, this_ccy, per_that_ccy):
        raise NotImplementedError('Foreign Exchange not yet implemented')

    def rate_sensitivity(self, ttm, currency, curve_key):
        """Sensitivity of interpolated point to node: dy(x)/dy_i

        Sensitivity of rate at time, ttm, to a unit move in the rate of each
         node in the curve. Hence, for each ttm, a vector is returned.

        Parameters
        ----------
        ttm: array-like
            Time in years from valuation date to some rate's maturity date.
        currency: str
            Currency of the curve.
        curve_key:
            Key to the curve in map_curves.

        Returns
        -------
        array-like
            shape = ttm.shape + curve.node_dates.shape
        """
        crv, key = self.curve(currency, curve_key)
        return crv.rate_sensitivity(ttm)

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

    def __str__(self):
        return str(self.nodes)
