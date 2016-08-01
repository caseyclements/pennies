import pytest
from datetime import datetime
from pennies import present_value
from pennies.assets.core import CashFlow
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.market.market import RatesTermStructure


def test_cash_flow_pv():
    today = datetime(2015, 1, 1)
    curve = ConstantDiscountRateCurve(today, 0.0)
    market = RatesTermStructure.of_single_curve(today, curve)

    cf1 = CashFlow(100, datetime(2016, 1, 1))
    assert present_value(cf1, market, as_of_date=today) == 100.0

    curve = ConstantDiscountRateCurve(today, 0.01)
    market = RatesTermStructure.of_single_curve(today, curve)

    cf1 = CashFlow(100, datetime(2016, 1, 1))
    assert round(present_value(cf1, market, as_of_date=today), 3) == 99.005
