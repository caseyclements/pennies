from __future__ import absolute_import, division, print_function


class CurrencyAmount(object):
    """ Amount of some given currency.

        Protects against naively adding amounts in different currencies.
    """
    # TODO Vectorize
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency

    def __str__(self):
        return '{} {}'.format(self.amount, self.currency)

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


class CurrencyWallet(object):
    """Dictionary of Currency (str) to CurrencyAmount

    This allows one to sum up the values of assets, or cashflows in different
    currencies without immediately converting them to a single currency given
    a foreign exchange rate, or raise an exception.
    """
    def __init__(self, ccy_amt=None):
        self._wallet = {}
        if ccy_amt is None:
            return
        elif isinstance(ccy_amt, CurrencyAmount):
            self._wallet[ccy_amt.currency] = ccy_amt
        else:
            raise ValueError('Constructor takes a CurrencyAmount or None')

    def __str__(self):
        return str(self._wallet)

    def __add__(self, other):
        res = CurrencyWallet()
        res._wallet = dict(self._wallet)
        if isinstance(other, None):
            return res
        elif isinstance(other, CurrencyAmount):
            ccy = other.currency
            if ccy in res._wallet:
                res._wallet[ccy].amount += other.amount
            else:
                res._wallet[ccy] = other
            return res
        elif isinstance(other, CurrencyWallet):
            for ccy in other._wallet.keys():
                if ccy in res._wallet:
                    res._wallet[ccy].amount += other.amount
                else:
                    res._wallet[ccy] = other
            return res
        else:
            raise ValueError('CurrencyWallet addition only works on '
                             'a CurrencyAmount or a CurrencyWallet')

    def __iadd__(self, other):
        if isinstance(other, None):
            return self
        if isinstance(other, CurrencyAmount):
            ccy = other.currency
            if ccy in self._wallet:
                self._wallet[ccy].amount += other.amount
            else:
                self._wallet[ccy] = other
            return self
        elif isinstance(other, CurrencyWallet):
            for ccy in other._wallet.keys():
                self  += other._wallet[ccy]
            return self
        else:
            raise ValueError('CurrencyWallet addition only works on '
                             'a CurrencyAmount or a CurrencyWallet')
