import numpy as np
from copy import deepcopy

def jsonify_dict(d, copy=True):
    if copy:
        d = deepcopy(d)
    for k,v in d.items():
        if type(v) == np.ndarray:
            d[k] = v.tolist()
    return d

def unjsonify_dict(d, copy=True):
    if copy:
        d = deepcopy(d)
    for k,v in d.items():
        if type(v) == list:
            d[k] = np.array(v)
    return d