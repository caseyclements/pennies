import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from pandas.tseries.offsets import DateOffset

from pennies.market.curves import DiscountCurveWithNodes
from pennies.market.market import RatesTermStructure

# 1. Define Valuation Date
dt_val = pd.to_datetime('today')

# 2. Define the Market Prices to Calibrate to.
# In our example, we create 3 Swaps, with an upward sloping yield

dt_settle = dt_val
frqncy = 6
curr = "USD"
durations = [12, 24, 60]
fixed_rates = [0.03, 0.04, 0.05]  # These are the 'prices'
notional = 100

def test_rates_term_structure():

    freq = DateOffset(months=6)
    sched = pd.date_range(dt_val, dt_val + DateOffset(months=24), freq=freq, closed='right')
    rates = np.random.rand(len(sched))
    crv = DiscountCurveWithNodes(dt_val, sched, rates, interpolator=CubicSpline)
    curve_map = {'USD': {'discount': crv}}

    freq = DateOffset(months=8)
    sched = pd.date_range(dt_val, dt_val + DateOffset(months=80), freq=freq, closed='right')
    rates = np.random.rand(len(sched))
    crv = DiscountCurveWithNodes(dt_val, sched, rates, interpolator=CubicSpline)
    curve_map['USD'][8] = crv

    freq = DateOffset(months=12)
    sched = pd.date_range(dt_val, dt_val + DateOffset(months=60), freq=freq, closed='right')
    rates = np.random.rand(len(sched))
    crv = DiscountCurveWithNodes(dt_val, sched, rates, interpolator=CubicSpline)
    curve_map['USD'][12] = crv

    freq = DateOffset(months=3)
    sched = pd.date_range(dt_val, dt_val + DateOffset(months=18), freq=freq, closed='right')
    rates = np.random.rand(len(sched))
    crv = DiscountCurveWithNodes(dt_val, sched, rates, interpolator=CubicSpline)
    curve_map['EUR'] = {'discount': crv}

    market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    df = market.nodes

    assert len(df) == 25
    assert set(df.curve[df.ccy == 'USD'].unique()) == set(['discount', 8, 12])

if __name__ == '__main__':
    test_rates_term_structure()
