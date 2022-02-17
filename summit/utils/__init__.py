import numpy as np
from copy import deepcopy
import numpy as np


def jsonify_dict(d, copy=True):
    """Make dictionary JSON serializable"""
    if copy:
        d = deepcopy(d)
    for k, v in d.items():
        if type(v) == np.ndarray:
            d[k] = v.tolist()
        elif type(v) == list:
            d[k] = jsonify_list(v)
        elif type(v) == dict:
            d[k] = jsonify_dict(v)
        elif type(v) in (np.int64, np.int32, np.int8):
            d[k] = int(v)
        elif type(v) in (np.float16, np.float32, np.float64):
            d[k] = float(v)
        elif type(v) in [str, int, float, bool, tuple] or v is None:
            pass
        else:
            raise TypeError(f"Cannot jsonify type for {v}: {type(v)}.")
    return d


def unjsonify_dict(d, copy=True):
    """Convert JSON back to proper types"""
    if copy:
        d = deepcopy(d)
    for k, v in d.items():
        if type(v) == list:
            d[k] = listtonumpy(v)
        elif type(v) == dict:
            d[k] = unjsonify_dict(v)
        elif type(v) in [str, int, float, bool, tuple] or v is None:
            pass
        else:
            raise TypeError(f"Cannot unjsonify type for {l}: {type(l)}.")
    return d


def jsonify_list(a, copy=True):
    if copy:
        a = deepcopy(a)
    for i, l in enumerate(a):
        if type(l) == list:
            a[i] = jsonify_list(l)
        elif type(l) == dict:
            a[i] = jsonify_dict(l)
        elif type(l) == np.ndarray:
            a[i] = l.tolist()
        elif type(l) in [str, int, float, bool, tuple] or l is None:
            pass
        else:
            raise TypeError(f"Cannot jsonify type for {l}: {type(l)}.")
    return a


def listtonumpy(a, copy=True):
    if copy:
        a = deepcopy(a)
    transform_all = True
    for i, l in enumerate(a):
        if type(l) == dict:
            a[i] = unjsonify_dict(l)
            transform_all = False
        elif type(l) == list:
            a[i] = listtonumpy(l)
            transform_all = False
        elif type(l) in [str, float, bool, int] or l is None:
            pass
        elif type(l) == tuple:
            transform_all = False
        else:
            raise TypeError(f"Cannot jsonify type for {l}: {type(l)}.")
    if transform_all:
        a = np.array(a)
    return a
