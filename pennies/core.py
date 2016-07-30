from __future__ import absolute_import, division, print_function

"""Objects used throughout the pennies package"""


class CurrencyAmount(object):
    """ Amount of some given currency.

        Protects against naively adding amounts in different currencies.
    """
    # TODO Vectorize
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency

    def __add__(self, other):
        assert isinstance(other, CurrencyAmount),\
            "Cannot add CurrencyAmount to anything but another CurrencyAmount"
        if other.currency == self.currency:
            amt = self.amount + other.amount
            return CurrencyAmount(amount=amt, currency=self.currency)
        else:
            raise NotImplementedError("Adding of different currencies")

    def __iadd__(self, other):
        assert isinstance(other, CurrencyAmount),\
            "Cannot add CurrencyAmount to anything but another CurrencyAmount"
        if other.currency == self.currency:
            self.amount += other.amount
            return self
        else:
            raise NotImplementedError("Adding of different currencies")

    def __sub__(self, other):
        assert isinstance(other, CurrencyAmount), \
            "Can only subtract two CurrencyAmount's"
        if other.currency == self.currency:
            amt = self.amount - other.amount
            return CurrencyAmount(amount=amt, currency=self.currency)
        else:
            raise NotImplementedError("Adding of different currencies")

    def __isub__(self, other):
        assert isinstance(other, CurrencyAmount), \
            "Can only subtract two CurrencyAmount's"
        if other.currency == self.currency:
            self.amount -= other.amount
            return self
        else:
            raise NotImplementedError("Adding of different currencies")

    def __mul__(self, other):
        assert isinstance(other, (int, float)),\
            "Can only multiply/divide a CurrencyAmount with a number"
        amt = self.amount * other
        return CurrencyAmount(amount=amt, currency=self.currency)

    def __imul__(self, other):
        assert isinstance(other, (int, float)), \
            "Can only multiply/divide a CurrencyAmount with a number"
        self.amount *= other
        return self

    def __truediv__(self, other):
        assert isinstance(other, (int, float)), \
            "Can only multiply/divide a CurrencyAmount with a number"
        amt = self.amount / other
        return CurrencyAmount(amount=amt, currency=self.currency)

    def __idiv__(self, other):
        assert isinstance(other, (int, float)), \
            "Can only multiply/divide a CurrencyAmount with a number"
        self.amount /= other
        return self

    def __str__(self):
        return '[amount={}, currency={}]'.format(self.amount, self.currency)

    __repr__ = __str__
