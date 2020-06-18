from .base import Strategy, Transform, MultitoSingleObjective, LogSpaceObjectives
from .random import Random, LHS
from .tsemo import TSEMO
from .snobfit import SNOBFIT
from .neldermead import NelderMead


def strategy_from_dict(d):
    if d["name"] == TSEMO:
        return TSEMO.from_dict(d)
    elif d["name"] == SNOBFIT:
        return SNOBFIT.from_dict(d)
    elif d["name"] == NelderMead:
        return NelderMead.from_dict(d)
