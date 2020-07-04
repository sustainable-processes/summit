"""
A large portion of this code is inspired by or copied from GPFlowOpt, which 
is Apache Licensed (open-source)
https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/optim.py

"""
from typing import List

# from summit.strategies import Random
from .dataset import DataSet

from abc import ABC, abstractmethod
import numpy as np
import platypus as pp
from scipy.optimize import OptimizeResult
import warnings
import numpy as np


class Optimizer(ABC):
    def __init__(self, domain):
        self.domain = domain
        self._multiobjective = False

    def optimize(self, objectivefx, **kwargs):
        """  Optimize the objective
        
        Parameters
        ---------- 
        models: List
            Should be a model or a list of models to optimize.
        
        Returns
        -------
        result: DataSet
            The result of the optimization as a DataSet
        
        Raises
        ------
        ValueError
            If multiple models are passed but the optimization method 
            is not multiobjective
               
        """
        objective = objectivefx
        result = self._optimize(objective, **kwargs)

        # result.nfev = objective.counter
        return result

    @abstractmethod
    def _optimize(self, models):
        raise NotImplementedError(
            "The Optimize class is not meant to be used directly. Instead use one of the specific optimizers such as NSGAII."
        )

    @property
    def is_multiobjective(self):
        """Return true if the algorithm does multiobjective optimization"""
        return self._multiobjective


class NSGAII(Optimizer):
    def __init__(self, domain):
        Optimizer.__init__(self, domain)
        # Set up platypus problem
        self.problem = pp.Problem(
            nvars=self.domain.num_variables(),
            nobjs=len(self.domain.output_variables),
            nconstrs=len(self.domain.constraints),
        )
        # Set maximization or minimization for each objective
        j = 0
        for i, v in enumerate(self.domain.variables):
            if v.is_objective:
                direction = self.problem.MAXIMIZE if v.maximize else self.problem.MINIMIZE
                self.problem.directions[j] = direction
                j+=1
            elif v.variable_type == "continuous":
                self.problem.types[i] = pp.Real(v.lower_bound, v.upper_bound)
            elif v.variable_type == "discrete":
                # Select a subset of one of the available options
                raise NotImplementedError(
                    "The NSGAII optimizer does not work with discrete variables"
                )
                # self.problem.types[i] = pp.Subset(elements=v.levels, size=1)
            elif v.variable_type == "descriptors":
                raise NotImplementedError(
                    "The NSGAII optimizer does not work with descriptors variables"
                )
            else:
                raise TypeError(f"{v.variable_type} is not a valid variable type.")

        # Set up constraints
        self.problem.constraints[:] = [
            c.constraint_type + "0" for c in domain.constraints
        ]

    def _optimize(self, models, **kwargs):
        input_columns = [v.name for v in self.domain.variables if not v.is_objective]
        output_columns = [v.name for v in self.domain.variables if v.is_objective]

        def problem_wrapper(X):
            X = DataSet(np.atleast_2d(X), columns=input_columns)
            result = models.predict(X, **kwargs)
            if self.domain.constraints:
                constraint_res = [
                    X.eval(c.lhs, resolvers=[X]) for c in self.domain.constraints
                ]
                constraint_res = [c.tolist()[0] for c in constraint_res]

                return result[0, :].tolist(), constraint_res
            else:
                return result[0, :].tolist()

        # Run optimization
        self.problem.function = problem_wrapper
        pop_size = kwargs.get('pop_size', 100)
        algorithm = pp.NSGAII(self.problem, population_size=pop_size)
        iterations = kwargs.get('iterations', 100)
        algorithm.run(iterations)

        x = [
            [s.variables[i] for i in range(self.domain.num_variables())]
            for s in algorithm.result
            if s.feasible
        ]
        x = DataSet(x, columns=input_columns)
        y = [
            [s.objectives[i] for i in range(len(self.domain.output_variables))]
            for s in algorithm.result
            if s.feasible
        ]
        y = DataSet(y, columns=output_columns)
        return OptimizeResult(x=x, fun=y, success=True)
