from multipledispatch import dispatch
from functools import partial

NAMESPACE =  "PENNIES_DISPATCH"

dispatch = partial(dispatch, namespace=NAMESPACE)
