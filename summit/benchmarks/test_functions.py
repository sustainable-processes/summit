from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
import pandas as pd

class Himmelblau(Experiment):
    ''' Himmelblau function (2D) for testing optimization algorithms

    Virtual experiment corresponds to a function evaluation.
    
    Examples
    --------
    >>> b = Himmelblau()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)
    
    Notes
    -----
    This function is taken from http://benchmarkfcns.xyz/benchmarkfcns/himmelblaufcn.html.
    
    ''' 
    def __init__(self):
        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Input 1"
        domain += ContinuousVariable(name='x_1',
                                     description=des_1,
                                     bounds=[-4, 4])
        
        des_2 = "Input 2"
        domain += ContinuousVariable(name='x_2',
                                     description=des_2,
                                     bounds=[-6, 6])

        # Objectives
        des_3 = 'Function value'
        domain += ContinuousVariable(name='y',
                                     description=des_3,
                                     bounds=[-1000, 1000],
                                     is_objective=True,
                                     maximize=True)

        return domain  

    def _run(self, conditions, **kwargs):
        x_1 = float(conditions['x_1'])
        x_2 = float(conditions['x_2'])
        himmelblau_equ = -((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)
        conditions[('y', 'DATA')] = himmelblau_equ
        return conditions, None

    def _plot(self, evluated_points, **kwargs):

        pass


class Hartmann3D(Experiment):
    ''' Hartmann test function (3D) for testing optimization algorithms

    Virtual experiment corresponds to a function evaluation.

    Examples
    --------
    >>> b = Hartmann3D()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    Notes
    -----
    This function is taken from https://www.sfu.ca/~ssurjano/hart3.html.

    '''

    def __init__(self):
        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Input 1"
        domain += ContinuousVariable(name='x_1',
                                     description=des_1,
                                     bounds=[0, 1])

        des_2 = "Input 2"
        domain += ContinuousVariable(name='x_2',
                                     description=des_2,
                                     bounds=[0, 1])

        des_3 = "Input 3"
        domain += ContinuousVariable(name='x_3',
                                     description=des_3,
                                     bounds=[0, 1])

        # Objectives
        des_4 = 'Function value'
        domain += ContinuousVariable(name='y',
                                     description=des_4,
                                     bounds=[-1000, 1000],
                                     is_objective=True,
                                     maximize=True)

        return domain

    def _run(self, conditions, **kwargs):

        def function_evaluation(x_1,x_2,x_3):
            x_exp = np.asarray([x_1,x_2,x_3])
            A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
            P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*10**(-4)
            alpha = np.array([1,1.2,3.0,3.2])
            d = np.zeros((4, 1))
            for k in range(4):
                d[k] = np.sum(np.dot(A[k,:],(x_exp-P[k,:])**2))
            y = np.sum(np.dot(alpha,np.exp(-d))).T
            return y


        x_1 = float(conditions['x_1'])
        x_2 = float(conditions['x_2'])
        x_3 = float(conditions['x_3'])

        y = function_evaluation(x_1,x_2,x_3)
        conditions[('y', 'DATA')] = y
        return conditions, None

    def _plot(self, evluated_points, **kwargs):

        pass

 