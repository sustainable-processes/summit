from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
import numpy as np

class DTLZ2(Experiment):
    def __init__(self, num_inputs, num_objectives=2):
        if num_objectives >= num_inputs:
            raise ValueError('Number of objectives must be greater than number of inputs.')
        self.nobjs = num_objectives
        self.nvars = num_inputs
        domain = self._setup_domain(self.nobjs, self.nvars)
        super().__init__(domain)

    def _setup_domain(self, nobjs, nvars):
        variables = [ContinuousVariable(f'x_{i}', 
                                        f'Decision variable {i}', 
                                        bounds=[0, 1.0])
                      for i in range(nvars)]
        variables += [ContinuousVariable(f'y_{i}', 
                                         f'Objective {i}', 
                                         bounds=[0, 1.0],
                                         is_objective=True,
                                         maximize=False)
                      for i in range(nobjs)]
        return Domain(variables)

    def _run(self, conditions, **kwargs):
        #Convert from dataframe
        x = conditions[[f'x_{i}' for i in range(self.nvars)]]
        x = x.to_numpy().astype(np.float64)
        x = np.atleast_2d(x)

        nobjs=self.nobjs
        """Copied from DEAP"""
        """https://github.com/DEAP/deap/blob/master/deap/benchmarks/__init__.py"""
        for individual in x:
            xc = individual[:nobjs-1]
            xm = individual[nobjs-1:]
            g = sum((xi-0.5)**2 for xi in xm)
            f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
            f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(nobjs-2, -1, -1))
        
        #Convert to dataset
        ds = DataSet(x, 
                     columns=[f'x_{i}' for i in range(self.nvars)])
        for i in range(nobjs):
            ds[(f'y_{i}', 'DATA')] = f[i]
        return ds, {}

class VLMOP2(Experiment):
    def __init__(self,):
        domain = self._setup_domain(2, 2)
        self.nvars = 2
        self.nobjs = 2
        super().__init__(domain)

    def _setup_domain(self, nobjs, nvars):
        variables = [ContinuousVariable(f'x_{i}', 
                                        f'Decision variable {i}', 
                                        bounds=[0, 1.0])
                      for i in range(nvars)]
        variables += [ContinuousVariable(f'y_{i}', 
                                         f'Objective {i}', 
                                         bounds=[0, 1.0],
                                         is_objective=True,
                                         maximize=False)
                      for i in range(nobjs)]
        return Domain(variables)

    def _run(self, conditions, **kwargs):
        #Convert from dataframe
        x = conditions[[f'x_{i}' for i in range(self.nvars)]]
        x = x.to_numpy().astype(np.float64)
        x = np.atleast_2d(x)
        
        transl = 1 / np.sqrt(2)
        part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
        part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
        y1 = 1 - np.exp(-1 * part1)
        y2 = 1 - np.exp(-1 * part2)
        f = np.hstack((y1, y2))

        #Convert to dataset
        ds = DataSet(x, 
                     columns=[f'x_{i}' for i in range(self.nvars)])
        for i in range(self.nobjs):
            ds[(f'y_{i}', 'DATA')] = f[:, i]
        return ds, {}