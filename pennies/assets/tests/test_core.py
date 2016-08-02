import pytest
from datetime import datetime
from pennies.assets.core import CashFlow


def test_cashflow_init():
    cf1 = CashFlow(100, '2015-01-01')
    assert cf1.amount == 100
    assert cf1.currency == 'USD'
    assert cf1.payment_date == datetime(2015, 1, 1)

    cf1 = CashFlow(100, '1/1/2015', currency='JPY')
    assert cf1.currency == 'JPY'
    assert cf1.payment_date == datetime(2015, 1, 1)

    cf1 = CashFlow(100, datetime(2015, 1, 1))
    assert cf1.payment_date == datetime(2015, 1, 1)

    with pytest.raises(ValueError):
        CashFlow(100, 'abc')
