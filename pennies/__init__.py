from multipledispatch import dispatch
from functools import partial

NAMESPACE = dict()
dispatch = partial(dispatch, namespace=NAMESPACE)  # TODO Why is namespace here?
