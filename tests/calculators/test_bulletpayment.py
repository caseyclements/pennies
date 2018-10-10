from __future__ import division, print_function
import numpy as np
import pandas as pd
from pennies.trading import assets
from pennies.trading import trades
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.market.market import RatesTermStructure
from pennies.calculators.payments import BulletPaymentCalculator
from pennies.calculators.trades import TradeCalculator, present_value
from pennies import CurrencyAmount


dt_val = pd.to_datetime('2017-01-01')
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


def test_contract_present_value():
    calculator = BulletPaymentCalculator(bullet, market)
    pv_calc = calculator.present_value()
    assert isinstance(pv_calc, CurrencyAmount)
    assert pv_calc.currency == 'USD'
    assert np.allclose(pv_calc.amount, expected_contract_pv), \
        "calculated present value is not as expected."


def test_dispatch_present_value():
    pv_dispatch = present_value(bullet, market, ccy)
    calculator = BulletPaymentCalculator(bullet, market)
    pv_calc = calculator.present_value()
    assert np.isclose(pv_dispatch, pv_calc.amount)


def test_contract_present_value_with_rates_zero():
    interest = 0.00
    crv = ConstantDiscountRateCurve(dt_valuation=dt_val, zero_rate=interest)
    market = RatesTermStructure.of_single_curve(dt_val, crv)
    calculator = BulletPaymentCalculator(bullet, market)
    pv_calc = calculator.present_value()
    assert isinstance(pv_calc, CurrencyAmount)
    assert pv_calc.currency == 'USD'
    assert np.allclose(pv_calc.amount, notional), \
        "calculated present value is not as expected."

    pv_dispatch = present_value(bullet, market, ccy)
    assert np.isclose(pv_dispatch, pv_calc.amount)


def test_trade_present_value():
    calculator = TradeCalculator(trade, market)
    pv_calc = calculator.present_value()
    assert isinstance(pv_calc, CurrencyAmount)
    assert pv_calc.currency == 'USD'
    assert np.allclose(pv_calc.amount, expected_contract_pv), \
        "calculated present value is not as expected."

    pv_dispatch = present_value(trade, market, ccy)
    assert np.isclose(pv_dispatch, pv_calc.amount)


def test_trade_present_value_with_settlement_on_valuationdate():
    trade_w_settlement = trades.Trade(contract=bullet, settlement_dt=dt_val,
                                      settlement_amt=notional)
    calculator = TradeCalculator(trade_w_settlement, market)
    pv_calc = calculator.present_value()
    assert isinstance(pv_calc, CurrencyAmount)
    assert pv_calc.currency == 'USD'
    assert np.allclose(pv_calc.amount, expected_contract_pv + notional), \
        "calculated present value is not as expected."

    pv_dispatch = present_value(trade_w_settlement, market, ccy)
    assert np.isclose(pv_dispatch, pv_calc.amount)
