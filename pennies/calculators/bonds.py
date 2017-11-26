from __future__ import absolute_import, division, print_function

import math
import numbers
import pandas as pd
from datetime import date
from pennies.assets.core import CashFlow
from pennies.assets.bonds import Bond
from pennies.dispatch import dispatch
from pennies.time import date_range, freq_to_frac, to_offset


def _bond_cash_flows(b):
    """Generate a series of cash flows for a bond.

    Parameters
    ----------

    b: Bond
        bond to compute cash flows

    Returns
    -------

    pandas.Series
    """

    # get payment dates
    end_date = b.start_date + to_offset(b.maturity)
    date_index = date_range(b.start_date, end_date, b.frequency)

    # get coupon cash flows
    if b.frequency is None:
        b.frequency = 'A'
    payments = [b.principal * b.coupon * freq_to_frac(b.frequency)] * len(date_index)
    payments[-1] += b.principal

    # create cash flow objects
    cash_flows = []
    for p, dt in zip(payments, date_index):
        cash_flows.append(CashFlow(p, dt.to_datetime()))

    # create series and remove cash flows that already occurred
    return pd.Series(cash_flows, index=date_index)[date.today():]


@dispatch(Bond, numbers.Number)
def present_value(b, discount_factor):
    """Calculate the present value of a bond.

    Calculates the value of simple bonds by taking the sum
    of the discounted cash flows relative to the coupon, frequency,
    and term to maturity.
    """
    cash_flows = _bond_cash_flows(b)
    pv_cash_flows = present_value(cash_flows, discount_factor)
    return pv_cash_flows.sum()
