from __future__ import absolute_import, division, print_function

import datetime as dt

from pennies.trading.assets import Asset, Swap, Annuity, IborLeg, FixedLeg, VanillaSwap
from pennies.market.market import Market, RatesTermStructure
from pennies.market.curves import ConstantDiscountRateCurve

from pennies.time import daycount
from pennies.core import CurrencyAmount
from pennies.dispatch import dispatch

BASE_ASSET = (Asset, object)
BASE_MARKET = (Market, object)


@dispatch(BASE_ASSET, BASE_MARKET)
def present_value(contract, market):
    """Base present value calculation.

    Given an asset (or sequence of assets), calculate it's present
    value as of today.  The supplied market provides prices, curves, and so on.

    Parameters
    ----------
    contract: Asset
        Asset to calculate present value
    market: Market
        Market to retrieve raw and computed market values
        such as prices, curves, and surfaces.

    Returns
    -------
    CurrencyAmount
        Present Value as named tuple containing amount and currency.
    """
    raise NotImplementedError("Not available for base types")


@dispatch(Annuity, RatesTermStructure)
def present_value(contract, market):  # type annotations not allowed? : Annuity. : RatesTermStructure
    """Present Value as sum of discount cash flows.

    This assumes that one has already computed the rates.
    For fixed rate annuities, this will be done during construction.
    For floating rate annuities, this will have to be pre-computed,
    typically via psuedo-discount factors of other curves."""

    a = contract.frame
    discount_factors = market.discount_factor(a.pay, currency=contract.currency)
    return CurrencyAmount(amount=(a.rate * a.period * discount_factors)
                          [a.pay >= market.dt_valuation].sum(),
                          currency=contract.currency)


@dispatch(Swap, RatesTermStructure)
def present_value(contract, market): # type annotations not allowed? : Swap. : RatesTermStructure

    return present_value(contract.leg_receive, market) - \
           present_value(contract.leg_pay, market)


@dispatch(IborLeg, RatesTermStructure)
def present_value(contract, market):  # type annotations not allowed? : Annuity. : RatesTermStructure
    """Present Value as sum of discount cash flows.

    This assumes that one has already computed the rates.
    For fixed rate annuities, this will be done during construction.
    For floating rate annuities, this will have to be pre-computed,
    typically via psuedo-discount factors of other curves."""

    a = contract.frame
    forwards = ibor_rate(contract, market)
    # replace rate with forwards for any fixing date after valuation date
    a.rate = a.rate.where(a.fixing < market.dt_valuation, forwards)
    # do not sum past cash flows
    discount_factors = market.discount_factor(a.pay, currency=contract.currency)
    return CurrencyAmount(amount=(a.rate * a.period * discount_factors)
                          [a.pay >= market.dt_valuation].sum(),
                          currency=contract.currency)


@dispatch(VanillaSwap, RatesTermStructure)
def par_rate(contract, market):

    df_fixed = contract.leg_fixed.frame.copy()
    df_fixed.rate = 1.0
    annuity = FixedLeg.from_frame(df_fixed)

    pv_fix = present_value(annuity, market)
    pv_flt = present_value(contract.leg_float, market)
    assert pv_fix.currency == pv_flt.currency
    return pv_flt.amount / pv_fix.amount


def ibor_rate(contract: IborLeg, market: RatesTermStructure):
    """Compute the natural (L)IBOR rate implied by the start and end schedules.

    This assumes that there is no convexity caused by lags between accrual dates
    and fixing and payment dates.
    """
    ccy = contract.frame.currency.iloc[0] #  TODO - One good reason not to use just a frame
    freq = contract.frequency  # TODO This is the alternative
    crv_pay = market.ibor_curve(ccy, freq)
    zcb_pay = crv_pay.discount_factor(contract.frame.pay)
    zcb_fix = crv_pay.discount_factor(contract.frame.fixing)
    return (zcb_fix / zcb_pay - 1.0) / contract.frame.period


if __name__ == '__main__':
    # TODO Turn this into tests !!!
    dt_val = dt.date.today()  # note: date
    dt_settle = dt_val - dt.timedelta(days=200)
    length = 24
    freq = 6
    fixed_rate = 0.03
    notional = 100
    ccy = "USD"

    fixed_leg = FixedLeg.from_tenor(dt_settlement=dt_settle, tenor=length,
                                    frequency=freq, rate=fixed_rate)

    #  Test @classmethod from_frame
    another_fixed_leg = Annuity.from_frame(fixed_leg.frame)
    and_another = Annuity(fixed_leg.frame)

    # 1. Create Market with Ibor and Discount Curves
    rate_discount = 0.05
    crv_discount = ConstantDiscountRateCurve(
        dt_valuation=dt_val, zero_rate=rate_discount,
        daycount_function=daycount('ACT365FIXED'), currency=ccy)

    spread = 0.002
    crv_ibor = ConstantDiscountRateCurve(  # Dummy IBOR Curve
        dt_valuation=dt_val, zero_rate=rate_discount + spread,
        daycount_function=daycount('ACT365FIXED'), currency=ccy)

    curve_map = {fixed_leg.frame.currency.iloc[0]:
                     {'discount': crv_discount, freq: crv_ibor}}
    simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # 2. Test IborLeg
    # Hack up what the frame might look like
    df_fixed = fixed_leg.frame
    df_float = df_fixed.copy().drop(['rate'], axis=1)
    df_float['fixing'] = df_fixed.start
    df_float['frequency'] = freq  # !!! TODO Dangerous reuse of common term. Are we ok to assume rate=coupon frequency? For now?
    df_float.notional *= -1
    float_leg = IborLeg.from_frame(df_float)

    # Calculate Forward Rates for each ibor cash flow
    forwards = ibor_rate(float_leg, simple_rates_market)

    # Add forwards to float_leg's frame so that it can priced by discounting
    #float_leg.frame['rate'] = forwards
    # No - instead. Set rates as if some have been fixed
    float_leg.frame['rate'] = 0.1

    # 3. Test VanillaSwap
    swap = VanillaSwap(fixed_leg, float_leg)

    # 4. Test pricing
    print('pv fixed_leg = {}'.format(present_value(fixed_leg, simple_rates_market)))
    print('pv float_leg = {}'.format(present_value(float_leg, simple_rates_market)))
    print('pv swap = {}'.format(present_value(swap, simple_rates_market)))
    print('swap rate = {}'.format(par_rate(swap, simple_rates_market)))
    print('FIN')
