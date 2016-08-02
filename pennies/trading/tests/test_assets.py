"""For now, playground while developing assets"""

import datetime as dt
from pandas import DataFrame
from pennies.trading.assets import Annuity

dt_settlement = dt.date.today()
duration = 24  # months
frequency = 6  # months
fixed_rate = 0.03
notional = 100
payment_lag = 2  # days


def test_annuity_from_tenor():
    fixed = Annuity.from_tenor(dt_settlement, tenor=duration,
                               frequency=frequency, rate=fixed_rate,
                               notional=notional, payment_lag=payment_lag)
    assert type(fixed) == Annuity
    df = fixed.frame
    assert type(df) == DataFrame
    cols = df.columns
    assert len(cols) == 13
    assert len(df) == 4


if __name__ == '__main__':
    test_annuity_from_tenor()
