from __future__ import division, print_function

from pennies.trading import assets
from pennies.calculators.assets import default_calculators

# TODO Use this to compare constructors


def test_default_asset_calculators():
    keys = default_calculators().keys()
    for asset in assets.all_assets():
        assert str(asset) in keys
