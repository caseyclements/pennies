from __future__ import division, print_function

from pennies.trading.assets import Asset
from pennies.calculators.assets import default_calculators

# TODO Use this to compare constructors

def test_default_asset_calculators():
    keys = default_calculators().keys()
    for asset in Asset.__subclasses__():
        assert str(asset) in keys
