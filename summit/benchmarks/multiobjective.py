from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
import math
import functools
import operator

class DTLZ2(Experiment):
    def __init__(self, nobjs=2, nvars=11):
        domain = self._setup_domain(nobjs, nvars)
        self.nvars = nvars
        self.nobjs = nobjs
        super().__init__(domain)

    def _setup_domain(self, nobjs, nvars):
        variables = [ContinuousVariable(f'x_{i}', 
                                        f'Decision variable {i}', 
                                        bounds=[0, 1.0])
                      for i in range(nobjs)]
        variables += [ContinuousVariable(f'y_{i}', 
                                        f'Objective {i}', 
                                        bounds=[0, 1.0],
                                        is_objective=True)
                      for i in range(nobjs)]
        return Domain(variables)

    def _run(self, conditions, **kwargs):
        #Convert from dataframe
        x = conditions[[f'x_{i}' for i in range(self.nvars)]]
        x = x.to_numpy()
        x = np.atleast_2d(x)

        #Run calculations
        nobjs=self.nobjs
        square = (x-0.5)**2
        g = np.sum(square[:, :nobjs-1], axis=1)
        f = np.repeat(1.0+np.atleast_2d(g).T, nobjs, axis=1)
        cos_term = np.cos(0.5*np.pi*x)
        cos_term = [np.prod(cos_term[:, :nobjs-1-i], axis=1)
                    for i in range(nobjs)]
        cos_term = np.array(cos_term).T
        sin_term = np.ones([x.shape[0], nobjs])
        z =  np.array([np.sin(0.5*np.pi*x[:, nobjs-1-i]) for i in range(nobjs-1)])
        sin_term[:, 1:] = z.T
        f = f*cos_term*sin_term

        #Convert to dataset
        ds = DataSet(x, 
                     columns=[f'x_{i}' for i in range(self.nvars)])
        for i in range(nobjs):
            ds[(f'y_{i}', 'DATA')] = f[0, i]

        return ds, {}
