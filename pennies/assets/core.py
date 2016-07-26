"""Assets represent Financial Assets, these are claims based on a contract.

These are often referred to as Securities or Products in other libraries.
"""

from __future__ import absolute_import, division, print_function


class Asset(object):
    """Base class of all Financial Assets"""

    def __init__(self):
        pass

    # TODO - Ask whether the Visitor Pattern is a good idea in Python
    def accept(self, visitor, *args, **kwargs):
        """Accepts visitors that calculate various measures on the Asset"""
        return visitor.visit(Asset, *args, **kwargs)


def all_assets():
    """Provides a list of all available Assets"""
    return Asset.__subclasses__()
