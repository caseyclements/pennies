"""Assets represent Financial Assets, these are claims based on a contract.

These are often referred to as Securities or Products in other libraries.
"""

from __future__ import absolute_import, division, print_function

import pandas as pd


class Asset(object):
    """Base class of all Financial Assets"""

    def __init__(self):
        pass

    # TODO - Ask whether the Visitor Pattern is a good idea in Python
    def accept(self, visitor, *args, **kwargs):
        """Accepts visitors that calculate various measures on the Asset"""
        return visitor.visit(Asset, *args, **kwargs)


class CashFlow(Asset):

    def __init__(self, amount, payment_date, currency='USD'):
        self.amount = amount
        self.payment_date = pd.to_datetime(payment_date)
        self.currency = currency
