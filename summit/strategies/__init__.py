from .base import *
from .random import Random, LHS
from .factorial_doe import FullFactorial
from .tsemo import TSEMO
from .neldermead import NelderMead
from .snobfit import SNOBFIT
from .sobo import SOBO
from .gryffin import GRYFFIN

def strategy_from_dict(d):
    if d["name"] == "TSEMO":
        return TSEMO.from_dict(d)
    elif d["name"] == "SNOBFIT":
        return SNOBFIT.from_dict(d)
    elif d["name"] == "NelderMead":
        return NelderMead.from_dict(d)
    elif d["name"] == "SOBO":
         return SOBO.from_dict(d)
    elif d["name"] == "FullFactorial":
        return FullFactorial.from_dict(d)
    elif d["name"] == "Random":
        return Random.from_dict(d)
    elif d["name"] == "LHS":
        return LHS.from_dict(d)
    elif d["name"] == "Gryffin":
        return GRYFFIN.from_dict(d)
    else:
        raise ValueError(f"""Strategy {d["name"]} not found.""")

