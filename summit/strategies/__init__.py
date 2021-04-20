from .base import __all__ as base_all
from .base import *
from .random import Random, LHS
from .factorial_doe import FullFactorial
from .tsemo import TSEMO
from .neldermead import NelderMead
from .snobfit import SNOBFIT
from .sobo import SOBO
from .multitask import MTBO, STBO
from .deep_reaction_optimizer import DRO
from .entmoot import ENTMOOT

__all__ = [
    "Random",
    "LHS",
    "FullFactorial",
    "TSEMO",
    "NelderMead",
    "SNOBFIT",
    "ENTMOOT",
    "MTBO",
    "STBO",
    "SOBO",
    "DRO",
    "strategy_from_dict",
] + base_all


def strategy_from_dict(d):
    if d["name"] == "STBO":
        return STBO.from_dict(d)
    elif d["name"] == "MTBO":
        return MTBO.from_dict(d)
    elif d["name"] == "TSEMO":
        return TSEMO.from_dict(d)
    elif d["name"] == "GRYFFIN":
        raise ValueError("Gryffin is now deprecated.")
    elif d["name"] == "SOBO":
        return SOBO.from_dict(d)
    elif d["name"] == "SNOBFIT":
        return SNOBFIT.from_dict(d)
    elif d["name"] == "NelderMead":
        return NelderMead.from_dict(d)
    elif d["name"] == "FullFactorial":
        return FullFactorial.from_dict(d)
    elif d["name"] == "Random":
        return Random.from_dict(d)
    elif d["name"] == "LHS":
        return LHS.from_dict(d)
    elif d["name"] == "DRO":
        return DRO.from_dict(d)
    elif d["name"] == "ENTMOOT":
        return ENTMOOT.from_dict(d)

    else:
        raise ValueError(f"""Strategy {d["name"]} not found.""")
