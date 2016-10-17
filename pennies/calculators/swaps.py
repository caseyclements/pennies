from __future__ import absolute_import, division, print_function

import datetime as dt
from pandas import DataFrame, Series

from pennies.trading.assets import Asset, Swap, Annuity, IborLeg, FixedLeg, VanillaSwap, RateType
from pennies.market.market import Market, RatesTermStructure
from pennies.market.curves import ConstantDiscountRateCurve
from pennies.dispatch import dispatch


  @dispatch(Annuity, RatesTermStructure, str)
def present_value(contract, market, reporting_ccy):
    """Present Value as sum of discount cash flows.

    This assumes that one has already computed the rates.
    For fixed rate annuities, this will be done during construction.
    For floating rate annuities, this will have to be pre-computed,
    typically via psuedo-discount factors of other curves."""

    a = contract.frame
    discount_factors = market.discount_factor(a.pay, currency=contract.currency)
    pv = (a.rate * a.period * discount_factors * a.notional)[a.pay >= market.dt_valuation].sum()
    if contract.notl_exchange:
        pv += a.notional.iloc[-1] * discount_factors.iloc[-1]
    if reporting_ccy != contract.currency:
        pv *= market.fx(reporting_ccy, contract.currency)
    return pv


@dispatch(Swap, RatesTermStructure, str)
def present_value(contract, market, reporting_ccy):
    """Present Value of a generic Swap."""
    return (present_value(contract.leg_receive, market, reporting_ccy) +
            present_value(contract.leg_pay, market, reporting_ccy))


@dispatch(IborLeg, RatesTermStructure, str)
def present_value(contract, market, reporting_ccy):
    """Present Value as sum of discounted IBOR cash flows.

    Forward LIBOR rates are calculated, and inserted into contract.frame.rate,
    for all fixing dates after dt_valuation. For fixing dates in the past,
    this assumes that contract.frame.rate is populated, and meaningful.
    """

    a = contract.frame
    forwards = ibor_rate(contract, market)
    # replace rate with forwards for any fixing date after valuation date
    a.rate = a.rate.where(a.fixing < market.dt_valuation, forwards)
    # do not sum past cash flows
    discount_factors = market.discount_factor(a.pay, currency=contract.currency)
    pv = (a.rate * a.period * discount_factors * a.notional)[a.pay >= market.dt_valuation].sum()
    if contract.notl_exchange:
        pv += a.notional.iloc[-1] * discount_factors.iloc[-1]
    if reporting_ccy != contract.currency:
        pv *= market.fx(reporting_ccy, contract.currency)
    return pv


@dispatch(VanillaSwap, RatesTermStructure)
def par_rate(contract, market):

    df_fixed = contract.leg_fixed.frame.copy()
    df_fixed.rate = 1.0
    df_fixed.notional *= -1
    annuity = FixedLeg.from_frame(df_fixed, notl_exchange=False)
    df_float = contract.leg_float.frame.copy()
    floating_no_xch = IborLeg.from_frame(df_float, notl_exchange=False)
    ccy = contract.leg_fixed.currency
    assert ccy == contract.leg_float.currency
    pv_fix = present_value(annuity, market, ccy)
    pv_flt = present_value(floating_no_xch, market, ccy)
    return pv_flt / pv_fix


def ibor_rate(contract: IborLeg, market: RatesTermStructure):
    """ALL the natural (L)IBOR rates implied by the start and end schedules.

    Returns
    -------
    Series
        Pandas Series containing Forward IBOR rates

    This assumes that there is no convexity caused by lags between accrual dates
    and fixing and payment dates.
    """

    crv_pay = market.ibor_curve(contract.currency, contract.frequency)
    zcb_pay = crv_pay.discount_factor(contract.frame.pay)
    zcb_fix = crv_pay.discount_factor(contract.frame.fixing)
    return (zcb_fix / zcb_pay - 1.0) / contract.frame.period


def d_price_d_rate(crv):
    """First derivative of each node in a discount curve to it's discount rates.

    The crv holds zero coupon bond prices in the form: z_i = exp(-r_i * ttm_i)
    """
    return -1 * crv.sched_maturity * crv.discount_factor(crv.sched_maturity)


if __name__ == '__main__':
    # TODO Turn this into tests !!!
    dt_val = dt.date.today()  # note: date
    dt_settle = dt_val - dt.timedelta(days=200)
    length = 24
    frqncy = 6
    fixed_rate = 0.03
    notional = 100
    curr = "USD"

    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=frqncy, rate=fixed_rate,
                                    notional=notional)

    #  Test @classmethod from_frame
    another_fixed_leg = Annuity.from_frame(fixed_leg.frame)
    and_another = Annuity(fixed_leg.frame)

    # 1. Create Market with Ibor and Discount Curves
    rate_discount = 0.05
    crv_discount = ConstantDiscountRateCurve(
        dt_valuation=dt_val, zero_rate=rate_discount,
        daycount_conv='ACT365FIXED', currency=curr)

    spread = 0.002
    crv_ibor = ConstantDiscountRateCurve(  # Dummy IBOR Curve
        dt_valuation=dt_val, zero_rate=rate_discount + spread,
        daycount_conv='ACT365FIXED', currency=curr)

    curve_map = {fixed_leg.frame.currency.iloc[0]:
                     {'discount': crv_discount, frqncy: crv_ibor}}
    simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # 2. Test IborLeg
    # Hack up what the frame might look like
    df_fixed = fixed_leg.frame
    df_float = df_fixed.copy()
    df_float.type = RateType.IBOR
    df_float['fixing'] = df_fixed.start
    df_float['frequency'] = frqncy
    df_float.notional *= -1
    float_leg = IborLeg.from_frame(df_float)

    # Calculate Forward Rates for each ibor cash flow
    forwards = ibor_rate(float_leg, simple_rates_market)

    # 3. Test VanillaSwap
    swap = VanillaSwap(fixed_leg, float_leg)

    # 4. Test pricing
    print('pv fixed_leg = {}'.format(present_value(fixed_leg, simple_rates_market, curr)))
    print('pv float_leg = {}'.format(present_value(float_leg, simple_rates_market, curr)))
    print('pv swap = {}'.format(present_value(swap, simple_rates_market, curr)))
    print('forward rates = {}'.format(forwards))
    print('swap rate = {}'.format(par_rate(swap, simple_rates_market)))

    print('FIN')
