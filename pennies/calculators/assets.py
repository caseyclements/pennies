from __future__ import absolute_import, division, print_function

class AssetCalculator(object):

    def __init__(self, contract, market):
        """
        Parameters
        ----------
        contract: Asset
            Asset (or Trade) representing the payment of a fixed amount
        market: Market
            Market values required to price the Asset.
        """
        self.contract = contract
        self.market = market


def all_calculators():
    return AssetCalculator.__subclasses__()
