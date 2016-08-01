from __future__ import absolute_import, division, print_function

import math
import numbers
import pandas as pd
from datetime import date
from pennies.assets.core import CashFlow
from pennies.assets.bonds import Bond
from pennies.dispatch import dispatch
from pennies.utils.time import date_range, freq_to_frac, to_offset


def _bond_cash_flows(principal, start_date, maturity, frequency, coupon, currency):

    # get payment dates
    end_date = start_date + to_offset(maturity)
    date_index = date_range(start_date, end_date, frequency)

    # get coupon cash flows
    if frequency is None:
        frequency = 'A'
    payments = [principal * coupon * freq_to_frac(frequency)] * len(date_index)
    payments[-1] += principal

    # create cash flow objects
    cash_flows = []
    for p, dt in zip(payments, date_index):
        cash_flows.append(CashFlow(p, dt.to_datetime()))

    # create series and remove cash flows that already occurred
    return pd.Series(cash_flows, index=date_index)[date.today():]


@dispatch(Bond, numbers.Number)
def present_value(b, discount_factor):
    cash_flows = _bond_cash_flows(b.principal, b.start_date, b.maturity, b.frequency, b.coupon, b.currency)
    pv_cash_flows = present_value(cash_flows, discount_factor)
    return pv_cash_flows.sum()
