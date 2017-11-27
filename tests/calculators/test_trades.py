from __future__ import division, print_function
import numpy as np
import pandas as pd
from pennies.trading import assets
from pennies.trading import trades
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.market.market import RatesTermStructure
from pennies.calculators import present_value

dt_val = pd.to_datetime('today')
dt_pay = dt_val + pd.Timedelta(days=730)
notional = 5.5e6
ccy = "USD"
bullet = assets.BulletPayment(dt_payment=dt_pay, currency=ccy, notional=notional)
trade = trades.Trade(contract=bullet)

rate_discount = 0.05
crv_discount = ConstantDiscountRateCurve(
    dt_valuation=dt_val, zero_rate=rate_discount,
    daycount_conv='30360', currency=ccy)
market = RatesTermStructure.of_single_curve(dt_val, crv_discount)
expected_contract_pv = 4976605.8


def test_trade_present_value():
    pv = present_value(trade, market, ccy)
    assert np.isclose(pv, expected_contract_pv)


def test_trade_present_value_with_settlement_on_valuationdate():
    trade_w_settlement = trades.Trade(contract=bullet, settlement_dt=dt_val,
                                      settlement_amt=notional)
    pv = present_value(trade_w_settlement, market, ccy)
    assert np.isclose(pv, expected_contract_pv + notional)