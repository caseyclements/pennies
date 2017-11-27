from __future__ import absolute_import, division, print_function

from pandas import DataFrame, Series
from numpy import zeros

from pennies.trading.assets import Swap, Annuity, IborLeg, FixedLeg, VanillaSwap
from pennies.market.market import RatesTermStructure
from pennies.market.curves import ConstantDiscountRateCurve
from multipledispatch import dispatch


@dispatch(Annuity, RatesTermStructure, str)
def present_value(contract, market, reporting_ccy):
    """Present Value as sum of discount cash flows.

    This assumes that one has already computed the rates.
    For fixed rate annuities, this will be done during construction.
    For floating rate annuities, this will have to be pre-computed,
    typically via psuedo-discount factors of other curves."""

    a = contract.frame
    discount_factors = market.discount_factor(a.pay, currency=contract.currency)
    alive = a.pay >= market.dt_valuation
    if not alive.any():
        return 0.0
    pv = (a.rate * a.period * discount_factors * a.notional).loc[alive].sum()
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
    alive = a.pay >= market.dt_valuation
    if not alive.any():
        return 0.0
    pv = (a.rate * a.period * discount_factors * a.notional).loc[alive].sum()
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


def ibor_rate(contract, market):
    """ALL the natural (L)IBOR rates implied by the start and end schedules.

    Returns
    -------
    Series
        Pandas Series containing Forward IBOR rates

    This assumes that there is no convexity caused by lags between accrual dates
    and fixing and payment dates.
    """
    assert isinstance(contract, IborLeg)
    assert isinstance(market, RatesTermStructure)
    crv_fwd, key = market.curve(contract.currency, contract.frequency)
    zcb_pay = crv_fwd.discount_factor(contract.frame.pay)
    zcb_fix = crv_fwd.discount_factor(contract.frame.fixing)
    return (zcb_fix / zcb_pay - 1.0) / contract.frame.period


def d_price_d_rate(crv):
    """First derivative of each node in a discount curve to it's discount rates.

    The crv holds zero coupon bond prices in the form: z_i = exp(-r_i * ttm_i)
    """
    return -1 * crv.sched_maturity * crv.discount_factor(crv.sched_maturity)


@dispatch(Annuity, RatesTermStructure, object)
def sens_to_zero_price(contract, market, curve_key):
    """Return Series of sensitivities to the natural ZeroCouponBond prices.

    By natural, we mean each of the payment dates of the Annuity.
    Sensitivities are only to the curve specified in the RatesTermStructure.
    """
    if curve_key == 'discount':
        df = contract.frame
        alive = df.pay >= market.dt_valuation
        sens = (df.rate * df.period).loc[alive]
        if contract.notl_exchange and alive.any():
            sens.iloc[-1] += df.notional.iloc[-1]
    else:
        return 0


@dispatch(VanillaSwap, RatesTermStructure, object)
def sens_to_zero_price(contract, market, curve_key):
    """Return Series of sensitivities to the natural ZeroCouponBond prices.

    By natural, we mean each of the payment dates of the Annuity.
    Sensitivities are only to the curve specified in the RatesTermStructure.
    """
    raise NotImplementedError('For Swaps, call each leg separately.')


@dispatch(Annuity, RatesTermStructure, str, object, str)
def sens_to_zero_rates(contract, market, curve_ccy, curve_key, reporting_ccy):
    """Sensitivity of each cashflow to the curve specified by currency and key

    The fixed rate annuity is only sensitive to the discount curve
     of the currency in which the cash flows (coupons) are paid.

     If curve_ccy does not match contract.currency,
     and curve_key is not 'discount' an empty DataFrame is returned.
    """
    df_sens = DataFrame(columns=['ttm', 'sens', 'ccy', 'curve'])
    if curve_ccy == contract.currency:
        if curve_key == 'discount':
            a = contract.frame
            alive = a.pay >= market.dt_valuation
            crv = market.discount_curve(curve_ccy)
            pay_dates = a.pay[alive]
            ttm = crv.daycount_fn(market.dt_valuation, pay_dates)
            zcb = market.discount_factor(pay_dates, currency=contract.currency)
            sens = -ttm * zcb * (a.rate * a.period * a.notional).loc[alive]
            if contract.notl_exchange and alive.any():
                sens.iloc[-1] += a.notional.iloc[-1]
            if reporting_ccy != contract.currency:
                sens *= market.fx(reporting_ccy, contract.currency)
            df_sens = DataFrame({'ttm': ttm, 'sens': sens,
                                 'ccy': curve_ccy, 'curve': curve_key})
    return df_sens


@dispatch(IborLeg, RatesTermStructure, str, object, str)
def sens_to_zero_rates(contract, market, curve_ccy, rate_key, reporting_ccy):
    """Sensitivity of each cashflow to the curve specified by currency and key

    A leg that pays IBOR is sensitive to both the discount and tenor curve
     of the currency in which the cash flows (coupons) are paid.
    """
    df_sens = DataFrame(columns=['ttm', 'sens', 'ccy', 'curve'])
    if curve_ccy == contract.currency:

        forwards = ibor_rate(contract, market)
        # replace rate with forwards for any fixing date after valuation date
        a = contract.frame
        a.rate = a.rate.where(a.fixing < market.dt_valuation, forwards)

        zcb_pay = market.discount_factor(a.pay, currency=contract.currency)

        if rate_key == 'discount':
            unpaid = a.pay >= market.dt_valuation
            crv = market.discount_curve(curve_ccy)
            pay_dates = a.pay[unpaid]
            ttm_pay = crv.daycount_fn(market.dt_valuation, pay_dates)
            sens = -ttm_pay * (zcb_pay * a.notional * a.rate * a.period).loc[unpaid]
            if contract.notl_exchange and unpaid.any():
                sens.iloc[-1] += a.notional.iloc[-1]
            if reporting_ccy != contract.currency:
                sens *= market.fx(reporting_ccy, contract.currency)
            df_sens = DataFrame({'ttm': ttm_pay, 'sens': sens,
                                 'ccy': curve_ccy, 'curve': 'discount'})
        elif rate_key == contract.frequency:  # TODO - Review and add comments
            crv, crv_key = market.curve(contract.currency, contract.frequency)
            unfixed = a.fixing >= market.dt_valuation
            pay_dates = a.pay.loc[unfixed]
            ttm_pay = crv.daycount_fn(market.dt_valuation, pay_dates)
            zcbi_pay = crv.discount_factor(pay_dates)

            fix_dates = a.fixing.loc[unfixed]
            ttm_fix = crv.daycount_fn(market.dt_valuation, fix_dates)
            zcbi_fix = crv.discount_factor(contract.frame.fixing)

            scale_factor = zcbi_fix / zcbi_pay * (a.notional * zcb_pay).loc[unfixed]
            sens_pay = ttm_pay * scale_factor
            sens_fix = -ttm_fix * scale_factor

            if reporting_ccy != contract.currency:
                fx = market.fx(reporting_ccy, contract.currency)
                sens_pay *= fx
                sens_fix *= fx

            df_pay = DataFrame({'ttm': ttm_pay, 'sens': sens_pay}).set_index('ttm')
            df_fix = DataFrame({'ttm': ttm_fix, 'sens': sens_fix}).set_index('ttm')
            df_sens = df_pay.add(df_fix, fill_value=0)

            df_sens['ttm'] = df_sens.index
            df_sens['ccy'] = curve_ccy
            df_sens['curve'] = crv_key

    return df_sens


@dispatch(Annuity, RatesTermStructure, str)
def sens_to_market_rates(contract, market, reporting_ccy):
    """Compute sensitivity of contract to each node in the market's curves."""

    # 1. Sensitivity of the CONTRACT PV to CONTRACT RATES: dV/dR_k
    # i.e. rates at contract dates, such as fixing, and maturity
    # ==> Only sensitive to discount curve
    ccy = contract.currency
    df_pv_sens = sens_to_zero_rates(contract, market, ccy, 'discount', reporting_ccy)
    dv_drk = df_pv_sens.sens.values
    ttm_k = df_pv_sens.ttm

    # 2. Sensitivity of CONTRACT RATES to MARKET RATES: dR_k / dR_j
    # This is a function of the curve's interpolator
    drk_drj_disc = market.rate_sensitivity(ttm_k, ccy, 'discount')

    # 3. Sensitivity of the CONTRACT PV to MARKET RATES, dV / dR_j
    # Multiple 1 and 2, and sum over contract dates
    dv_drj = zeros(len(market.nodes))

    mask_disc = ((market.nodes.ccy == contract.currency) &
                 (market.nodes.curve == 'discount')).values
    dv_drj[mask_disc] = drk_drj_disc.T.dot(dv_drk)  # TODO NEED TO EXAMINE. Should this be: dv_drk.dot(drk_drj_disc) ?

    # 1d-array of sensitivities to each of the market's nodes. Lots of 0's
    return dv_drj


@dispatch(IborLeg, RatesTermStructure, str)
def sens_to_market_rates(contract, market, reporting_ccy):
    """Compute sensitivity of contract to each node in the market's curves."""

    # 1. Sensitivity of the CONTRACT PV to CONTRACT RATES: dV/dR_k
    # i.e. rates at contract dates, such as fixing, and maturity

    ccy = contract.currency
    # 1a. discount curve
    df_pv_sens = sens_to_zero_rates(contract, market, ccy, 'discount', reporting_ccy)
    dv_drk_disc = df_pv_sens.sens.values
    ttm_k_disc = df_pv_sens.ttm

    # 1b. ibor curve
    ibor_key = contract.frequency  # TODO Rate and frequency should be separate
    df_pv_sens = sens_to_zero_rates(contract, market, ccy, ibor_key, reporting_ccy)
    ibor_key = df_pv_sens.curve.iat[0]  # May be 'discount', not frequency
    dv_drk_ibor = df_pv_sens.sens.values
    ttm_k_ibor = df_pv_sens.ttm

    # 2. Sensitivity of CONTRACT RATES to MARKET RATES: dR_k / dR_j
    # This is a function of the curve's interpolator
    # Sensitivity to the discount curve
    drk_drj_disc = market.rate_sensitivity(ttm_k_disc, ccy, 'discount')
    # Sensitivity to the ibor curve
    drk_drj_ibor = market.rate_sensitivity(ttm_k_ibor, ccy, ibor_key)

    # 3. Sensitivity of the CONTRACT PV to MARKET RATES
    # For each curve, multiply 1 and 2, and sum over contract dates
    dv_drj = zeros(len(market.nodes))
    mask_disc = ((market.nodes.ccy == contract.currency) &
                 (market.nodes.curve == 'discount')).values
    dv_drj[mask_disc] = dv_drj[mask_disc] + drk_drj_disc.T.dot(dv_drk_disc)  # TODO NEED TO EXAMINE

    mask_ibor = ((market.nodes.ccy == contract.currency) &
                 (market.nodes.curve == ibor_key)).values
    dv_drj[mask_ibor] = dv_drj[mask_ibor] + drk_drj_ibor.T.dot(dv_drk_ibor)  # TODO NEED TO EXAMINE !!!!!!!!!!!!!!

    # 1d-array of sensitivities to each of the market's nodes.
    # May contain many 0's
    return dv_drj


@dispatch(Swap, RatesTermStructure, str)
def sens_to_market_rates(contract, market, reporting_ccy):
    """Compute sensitivity of contract to each node in the market's curves."""
    return (sens_to_market_rates(contract.leg_receive, market, reporting_ccy) +
            sens_to_market_rates(contract.leg_pay, market, reporting_ccy))


if __name__ == '__main__':
    import pandas as pd
    dt_val = pd.to_datetime('today')
    dt_settle = dt_val - pd.Timedelta(days=200)
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
        daycount_conv='30360', currency=curr)

    spread = 0.002
    crv_ibor = ConstantDiscountRateCurve(  # Dummy IBOR Curve
        dt_valuation=dt_val, zero_rate=rate_discount + spread,
        daycount_conv='30360', currency=curr)

    curve_map = {fixed_leg.frame.currency.iloc[0]:
                     {'discount': crv_discount, frqncy: crv_ibor}}
    simple_rates_market = RatesTermStructure.from_curve_map(dt_val, curve_map)

    # 2. Test IborLeg
    # Hack up what the frame might look like
    df_fixed = fixed_leg.frame
    df_float = df_fixed.copy()
    df_float.type = 'IBOR'
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
