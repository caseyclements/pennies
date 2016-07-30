from __future__ import print_function

import logging  # TODO later

from collections import Callable
import datetime as dt

from pennies.core import CurrencyAmount
from pennies.trading.assets import ZeroCouponBond
from pennies.market.market import RatesTermStructure
from pennies.market.curves import ConstantDiscountRateCurve


_cached = {}  # TODO See https://github.com/UDST/orca/blob/master/orca/orca.py

_nodes_by_name = {}
_nodes_by_value = {}
_nodes_by_type = {}


class Node(object):
    """Nodes in the Calculation Framework

    Attributes
    ----------
    name: str
        eg
    value: object or function
    type:
        the type of the object, or return type of the function
    requires: tuple of Node type
        this might be easier than asking for a name, value, or type...
    properties
        this might get very complicated..
    """
    def __init__(self, value=None, type=None, name=None,
                 requires=None, properties=None):
        self.name = name
        self.value = value
        self.type = type
        self.requires = requires
        self.properties = properties


def add_node(value, type, name, requires=None, properties=None):
    """Very simple implementation of nodes.

    One way to differentiate similar nodes would be through properties
    """
    new_node = Node(value, type, name, requires, properties)
    if _nodes_by_name.get(name) is None:
        _nodes_by_name[name] = new_node
    else:
        raise ValueError('Node with name, {}, already exists'.format(name))

    if _nodes_by_value.get(value) is None:
        _nodes_by_value[value] = new_node
    else:
        raise NotImplementedError('Node with value, {}, already exists. '
                                  'Needs log to handle.'.format(value))

    if _nodes_by_type.get(type) is None:
        _nodes_by_type[type] = new_node
    else:
        raise NotImplementedError('Node with type, {}, already exists. '
                                  'Needs log to handle.'.format(type))


def register_node(type, name=None, requires=None, properties=None):
    """Decorator version of add_node """

    def decorator(fn):
        node_name = fn.__name__ if name is None else name
        add_node(fn, type, node_name, requires, properties)
        return fn
    return decorator


def calculate(value):
    """Simple call stack"""
    node = _nodes_by_name.get(value)
    if node is None:
        node = _nodes_by_value.get(value)
        if node is None:
            node = _nodes_by_type.get(value)
        if node is None:
            raise ValueError('No registered node matches {}'.format(value))

    if isinstance(node.value, Callable):
        reqs = node.requires
        if reqs is None:
            return node.value()
        else:
            args = []
            for req in reqs:
                args.append(calculate(req))
            return node.value(*args)
    else:
        return node.value


class Measure(object):
    """Base class for something that can be calculated."""
    pass


class PresentValue(Measure):
    """Common Measure available for Assets."""
    def __init__(self, pv: CurrencyAmount):
        self.pv = pv

    def __call__(self, *args, **kwargs):
        return self.pv

if __name__ == '__main__':

    # 1. Define and Register FUNCTION that computes PV given a contract and a market
    #    Here I am getting the name and value from the function itself
    @register_node(type=PresentValue,
                   requires=(ZeroCouponBond, RatesTermStructure),
                   properties=None)
    def zcb_pv(contract: ZeroCouponBond,
               market: RatesTermStructure) -> PresentValue:
        df = market.discount_factor(contract.dt_payment,
                                    contract.currency)
        return CurrencyAmount(contract.notional * df, contract.currency)


    # 2. Create a RatesTermStructure

    # 2a. Register a valuation date. type == dateteime
    dt_val = dt.datetime.now()
    add_node(name='VAL_DATE', value=dt_val, type=dt.datetime)

    # 2b. Register the market. NOTE: type == function
    market_maker = RatesTermStructure.of_single_curve
    add_node(name='USD-MARKET', value=RatesTermStructure.of_single_curve,
             type=RatesTermStructure,
             # TODO How come I can go type=int but I can't do type=function?
             requires=('VAL_DATE', ConstantDiscountRateCurve), properties=None)

    # 2c. Register the curve that market_maker function requires.
    rate_discount = 0.05
    crv_discount = ConstantDiscountRateCurve(dt_valuation=dt_val,
                                             zero_rate=rate_discount)
    add_node(name='USD-CRV', value=crv_discount, type=type(crv_discount))

    # 3a. Create a ZeroCouponBond

    dt_pay = dt_val + dt.timedelta(days=730)
    notional = 100.
    ccy = "USD"
    bullet = ZeroCouponBond(dt_payment=dt_pay)
    # 3b. Register this contract
    add_node(name='FRANKLIN', value=bullet, type=type(bullet),
             requires=None,
             properties=None)  # TODO Can I call decorator like this???

    # 4. Ask module to calculate the node

    pv = calculate('zcb_pv')
    print('PresentValue = {}'.format(pv))
