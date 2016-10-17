
"""
This is going to undergo a great deal of refactoring as I expand from a single
fixed cashflow to forward interest rates and then options.


We will begin with the most simple market that we can.
One that is composed strictly of a single curve providing discount factors.

Question. In all situations, must a market require a disount curve? YES

Most likely, Markets will be constructed by @classmethod

"""
from __future__ import absolute_import, division, print_function
import pandas as pd
from pandas import DataFrame
from pennies.market.curves import Curve, DiscountCurveWithNodes


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

     The core of a RatesTermStructure is a dt_valuation, and a dict of curves,
    map_curves. The is keyed off currency, then name, which must include
    'discount'. The others will typically be ibor frequencies: 3, 6, ..

    Attributes
    ----------
    map_curves: dict of dict
        Contains a dictionary of dictionaries. The top key is currency strings.
        Within each currency dictionary, 'disount' must exist.
    """

    def __init__(self, dt_valuation, map_curves=None, map_fx=None):
        super(RatesTermStructure, self).__init__(dt_valuation)
        self.map_curves = map_curves
        self.map_discount_curves = {ccy: self.map_curves[ccy]["discount"]
                                    for ccy in map_curves}
        self.map_fx = map_fx

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
        try:
            ccy_map = self.map_curves[currency]
        except KeyError:
            raise ValueError('Requested currency not present in market: {}'
                             .format(currency))
        try:
            return ccy_map[frequency]
        except KeyError as err:
            if len(ccy_map) == 1:
                return ccy_map['discount']
            else:
                raise KeyError('Ibor rate requested from market that contains'
                               'neither a discount curve or frequency: {}.'
                               'Perhaps the currency is wrong.'
                               .format(frequency))

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
        return self.map_curves[currency][curve_key].rate_sensitivity(ttm)


    @property
    def nodes_dataframe(self):
        df = DataFrame(columns=['ttm', 'rates', 'ccy', 'curve'])
        for ccy, curves in self.map_curves.items():
            df_ccy = DataFrame(columns=['curve', 'ttm', 'rates'])
            for key, crv in curves.items():
                if not isinstance(crv, DiscountCurveWithNodes):
                    raise ValueError('curve with ccy={}, id={}, is not a '
                                     'DiscountCurveWithNodes'.format(ccy, key))
                df_crv = crv.frame.loc[:, ['ttm', 'rates']]
                df_crv['ccy'] = ccy
                df_crv['curve'] = key
                if key == 'discount':
                    df_ccy = pd.concat([df_crv, df_ccy], ignore_index=True)
                else:
                    df_ccy = pd.concat([df_ccy, df_crv], ignore_index=True)
            df = pd.concat([df, df_ccy], ignore_index=True)
        df.sort(columns=['ccy','ttm'], inplace=True)
        return df

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
