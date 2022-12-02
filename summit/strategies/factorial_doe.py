from .base import Strategy, Design, Transform
from summit.domain import *
from summit.utils.dataset import DataSet

import numpy as np


class FullFactorial(Strategy):
    """Full factorial DoE
    Strategy for full factorial design of experiments in all decision variables.

    Parameters
    ----------
    domain: :class:`~summit.domain.Domain`
        The Summit domain describing the optimization problem.

    Examples
    -------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import FullFactorial
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> levels = dict(temperature=[50,100], flowrate_a=[0.1,0.5], flowrate_b=[0.1,0.5])
    >>> strategy = FullFactorial(domain)
    >>> strategy.suggest_experiments(levels)
    NAME temperature flowrate_a flowrate_b       strategy
    TYPE        DATA       DATA       DATA       METADATA
    0           50.0        0.1        0.1  FullFactorial
    1          100.0        0.1        0.1  FullFactorial
    2           50.0        0.5        0.1  FullFactorial
    3          100.0        0.5        0.1  FullFactorial
    4           50.0        0.1        0.5  FullFactorial
    5          100.0        0.1        0.5  FullFactorial
    6           50.0        0.5        0.5  FullFactorial
    7          100.0        0.5        0.5  FullFactorial

    Notes
    -----

    We rely on the implementation from `pyDoE2 <https://github.com/clicumu/pydoe2>`_.

    """

    def __init__(self, domain: Domain, transform: Transform = None, **kwargs):
        super().__init__(domain, transform, **kwargs)

    def suggest_experiments(self, levels_dict, **kwargs) -> DataSet:
        """Suggest experiments for a full factorial experimental design

        Parameters
        ----------
        levels_dict : dict
            A dictionary with the number of levels for each variable. Keys are
            the variable names and values are arrays with the values of each level.

        Returns
        -------
        ds
            A `Dataset` object with the random design
        """
        num_experiments = np.prod([len(level) for level in levels_dict.values()])
        design = Design(self.domain, num_experiments, "random")
        levels = []
        for v in self.domain.input_variables:
            # Set number of levels per variable
            var_levels = levels_dict.get(v.name)
            num_levels = len(var_levels) if var_levels is not None else 2
            levels.append(num_levels)

        # Create full factorial design
        doe = fullfact(levels)
        for i, v in enumerate(self.domain.input_variables):
            indices = doe[:, i]
            indices = indices.astype(int)
            values = np.array([levels_dict[v.name][i] for i in indices])
            values = np.atleast_2d(values)
            design.add_variable(v.name, values, indices=indices[:, np.newaxis])

        ds = design.to_dataset()
        ds[("strategy", "METADATA")] = "FullFactorial"
        return ds

    def reset(self):
        pass


def fullfact(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor


    Notes
    ------
    This code is copied from pydoe2: https://github.com/clicumu/pyDOE2/blob/master/pyDOE2/doe_factorial.py

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H
