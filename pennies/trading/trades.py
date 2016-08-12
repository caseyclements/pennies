"""A Trade is a contract signed between two Counterparties at a given datetime.

The contract referred to is typically an instance of an Asset.

In addition to the claims described within the contract, there may optionally
be an initial settlement of a Bullet payment.
"""

from __future__ import absolute_import, division, print_function

from .assets import BulletPayment


class Trade(object):
    """Base class of all Financial Trades

    A Trade is a contract signed between two Counterparties on some date.

    Attributes
    ----------
    contract: Asset
        The Asset on which the Trade is based.
    date: datetime, optional
        Date on which trade was made.
    counterparty: Counterparty, optional
        The Counterparty with whom the trade is made. Not necess
    settlement: BulletPayment, optional
    """

    def __init__(self, contract, date=None, counterparty=None,
                 settlement_dt=None, settlement_amt=None, settlement_ccy=None):
        """
        Parameters
        ----------
        contract: Asset
            The Asset on which the Trade is based.
        date: datetime, optional
            Date on which trade was made.
        counterparty: Counterparty, optional
            The Counterparty with whom the trade is made. Not necess
        settlement_dt: date, optional
            Date on which a BulletPayment is made to settle trade.
        settlement_amt: float, optional
            Date on which a BulletPayment is made to settle trade.
        settlement_ccy: str
            Date on which a BulletPayment is made to settle trade.
        """
        self.contract = contract
        self.date = date
        self.counterparty = counterparty
        if settlement_dt is None and settlement_amt is None:
            self.settlement = None
        else:
            assert settlement_dt, \
                'settlement_dt provided, but not settlement_amt'
            assert settlement_amt, \
                'settlement_amt provided, but not settlement_dt'
            if settlement_ccy is None:
                # Check if currency is unambiguous
                try:
                    settlement_ccy = contract.currency
                except AttributeError:
                    raise AttributeError('settlement_ccy must be set.')

            self.settlement = BulletPayment(dt_payment=settlement_dt,
                                            currency=settlement_ccy,
                                            notional=settlement_amt)

    @property
    def contract_type(self):
        return type(self.contract)


    @classmethod
    def with_settlement_contract(cls, date, underlying_contract,
                                 settlement_contract=None, counterparty=None):
        """Create a Trade given a Settlement"""
        cls.date = date
        cls.contract = underlying_contract
        cls.counterparty = counterparty
        cls.settlement = settlement_contract


    # TODO Check naming convention of classmethods
    # TODO Design question - Should I include asset-specific constructors?
    @classmethod
    def bullet_payment(cls, dt_payment, currency="USD", notional=1.0,
                       dt_trade=None, counterparty=None, settlement_dt=None,
                       settlement_amt=None, settlement_ccy="USD"):
        """Create Trade of a BulletPayment"""
        payment_contract = BulletPayment(dt_payment, currency, notional)
        return cls(payment_contract, dt_trade, counterparty,
                   settlement_dt, settlement_amt, settlement_ccy)


class Counterparty(object):
    # TODO Define Counterparty
    pass


class Portfolio(object):
    """A Portfolio of Trades and/or subportolios

    Attributes
    ----------
    trades: list of Trade
    subportfolios: dict
        Dictionary where values are of Portfolio
    """
    def __init__(self, trades=None, subportfolios=None):
        self.trades = []
        self.subportfolios = {}
        if trades is not None:
            self.trades = list(trades)
        if subportfolios is not None:
            self.subportfolios = dict(subportfolios)

    @classmethod
    def of_trades(cls, trades):
        return cls(trades=trades)

    @classmethod
    def of_subportfolios(cls, subportfolios):
        return cls(subportfolios=subportfolios)
