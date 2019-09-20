"""
A large portion of this code is inspired by or copied from GPFlowOpt, which 
is Apache Licensed (open-soruce)
https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/optim.py

"""
from typing import List
from summit.domain import Domain, DomainError
from summit.initial_design import RandomDesigner
from summit.data import DataSet

from abc import ABC, abstractmethod
import numpy as np
import platypus as pp
from scipy.optimize import OptimizeResult


class Optimizer(ABC):
    def __init__(self, domain: Domain):
        self.domain = domain
        self._multiobjective = False

    def optimize(self, objectivefx, **kwargs):
        '''  Optimize the objective
        
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
               
        ''' 
        objective = objectivefx
        try:
            result = self._optimize(objective, **kwargs)
        except KeyboardInterrupt:
            result = OptimizeResult(x=objective._previous_x,
                                    success=False,
                                    message="Caught KeyboardInterrupt, returning last good value.")
        # result.nfev = objective.counter
        return result

    @abstractmethod
    def _optimize(self, models):
        raise NotImplementedError('The Optimize class is not meant to be used directly. Instead use one of the specific optimizers such as NSGAII.')

    @property
    def is_multiobjective(self):
        '''Return true if the algorithm does multiobjective optimization'''
        return self._multiobjective

class NSGAII(Optimizer): 
    def __init__(self, domain: Domain):
        Optimizer.__init__(self, domain)
        #Set up platypus problem
        self.problem = pp.Problem(nvars=self.domain.num_variables(),
                                  nobjs=len(self.domain.output_variables),
                                  nconstrs=len(self.domain.constraints))
        #Set maximization or minimization for each objective                          
        j = 0
        for i, v in enumerate(self.domain.variables):
            if v.is_objective:
                direction = self.problem.MAXIMIZE if v.maximize else self.problem.MINIMIZE
                self.problem.directions[j] = direction
            elif v.variable_type == "continuous":
                self.problem.types[i] = pp.Real(v.lower_bound, v.upper_bound)
            elif v.variable_type == "discrete":
                #Select a subset of one of the available options
                raise NotImplementedError('The NSGAII optimizer does not work with discrete variables')
                # self.problem.types[i] = pp.Subset(elements=v.levels, size=1)
            elif v.variable_type == 'descriptors':
                raise NotImplementedError('The NSGAII optimizer does not work with descriptors variables')
            else:
                raise DomainError(f'{v.variable_type} is not a valid variable type.')

        #Set up constraints
        self.problem.constraints[:] = "<=0"

    def _optimize(self, models, **kwargs):
        input_columns = [v.name for  v in self.domain.variables if not v.is_objective]
        output_columns = [v.name for  v in self.domain.variables if v.is_objective]
        def problem_wrapper(X):
            X = DataSet(np.atleast_2d(X), 
                        columns=input_columns)
            result = models.predict(X)
            if self.domain.constraints:
                constraint_res = [X.eval(c.expression, resolvers=[X]) 
                                for c in self.domain.constraints]
                constraint_res = [c.tolist()[0] for c in constraint_res]

                return result[0, :].tolist(), constraint_res
            else:
                return result[0, :].tolist()
        
        #Run optimization
        self.problem.function = problem_wrapper
        algorithm = pp.NSGAII(self.problem)
        iterations = kwargs.get('iterations', 10)
        algorithm.run(iterations)
        
        x = [[s.variables[i] for i in range(self.domain.num_variables())]
             for s in algorithm.result]
        x = DataSet(x, columns = input_columns)
        y =[[s.objectives[i] for i in range(len(self.domain.output_variables))]
            for s in algorithm.result]
        y = DataSet(y, columns=output_columns)
        return OptimizeResult(x=x, fun=y, success=True)

class MCOptimizer(Optimizer):
    """
    Optimization of an objective function by evaluating a set of random points.
    Note: each call to optimize, a different set of random points is evaluated.
    """

    def __init__(self, domain, nsamples):
        """
        :param domain: Optimization :class:`~.domain.Domain`.
        :param nsamples: number of random points to use
        """
        Optimizer.__init__(domain)
        self._nsamples = nsamples
        # Clear the initial data points
        self.set_initial(np.empty((0, self.domain.size)))

    def domain(self, dom):
        self._domain = dom

    def _get_eval_points(self):
        r =  RandomDesigner(self.domain)
        return r.generate_experiments(self._nsamples)

    def _optimize(self, objective):
        points = self._get_eval_points()
        evaluations = objective(points)
        idx_best = np.argmin(evaluations, axis=0)

        return OptimizeResult(x=points[idx_best, :],
                              success=True,
                              fun=evaluations[idx_best, :],
                              nfev=points.shape[0],
                              message="OK")

    def set_initial(self, initial):
        initial = np.atleast_2d(initial)
        if initial.size > 0:
            warnings.warn("Initial points set in {0} are ignored.".format(self.__class__.__name__), UserWarning)
            return

        super(MCOptimizer, self).set_initial(initial)

class CandidateOptimizer(MCOptimizer):
    """
    Optimization of an objective function by evaluating a set of pre-defined candidate points.
    Returns the point with minimal objective value.
    """

    def __init__(self, domain, candidates):
        """
        :param domain: Optimization :class:`~.domain.Domain`.
        :param candidates: candidate points, should be within the optimization domain.
        """
        MCOptimizer.__init__(self, domain, candidates.shape[0])
        assert (candidates in domain)
        self.candidates = candidates

    def _get_eval_points(self):
        return self.candidates

    def domain(self, dom):
        t = self.domain >> dom
        super(CandidateOptimizer, self.__class__).domain.fset(self, dom)
        self.candidates = t.forward(self.candidates)