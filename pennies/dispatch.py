from multipledispatch import dispatch
from functools import partial

NAMESPACE = dict()

dispatch = partial(dispatch, namespace=NAMESPACE)
