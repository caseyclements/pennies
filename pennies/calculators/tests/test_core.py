import pytest
from datetime import datetime
from pennies import present_value
from pennies.assets.core import CashFlow
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.market.market import RatesTermStructure

@pytest.mark.parametrize("today,cf_date,rate,expected", [
    (datetime(2015, 1, 1), datetime(2016, 1, 1), 0.0, 100.0),
    (datetime(2015, 1, 1), datetime(2016, 1, 1), 0.01, 99.005),
    (datetime(2015, 1, 1), datetime(2014, 12, 31), 0.01, 0),
])
def test_cash_flow_pv(today, cf_date, rate, expected):
    curve = ConstantDiscountRateCurve(today, rate)
    market = RatesTermStructure.of_single_curve(today, curve)

    cf1 = CashFlow(100, cf_date)
    assert round(present_value(cf1, market, as_of_date=today), 3) == expected
