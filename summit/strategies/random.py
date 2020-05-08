from .base import Strategy, Design, _closest_point_indices
from summit.domain import (Domain, Variable, ContinuousVariable, 
                           DiscreteVariable, DescriptorsVariable,
                           DomainError)
from summit.utils.dataset import DataSet

import numpy as np
import pandas as pd
from typing import Type, Tuple

class Random(Strategy):
    ''' Random strategy for experiment suggestion

    Parameters
    ---------- 
    domain: `summit.domain.Domain`
        A summit domain object
    random_state: `np.random.RandomState``
        A random state object to seed the random generator
    
    Attributes
    ----------
    domain
    
    Examples
    -------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import Random
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = Random(domain, random_state=np.random.RandomState(3))
    >>> strategy.suggest_experiments(5)
    NAME  temperature  flowrate_a  flowrate_b strategy
    0       77.539895    0.458517    0.111950   Random
    1       85.407391    0.150234    0.282733   Random
    2       64.545237    0.182897    0.359658   Random
    3       75.541380    0.120587    0.211395   Random
    4       94.647348    0.276324    0.370502   Random

    Notes
    -----
    Descriptors variables are selected randomly as if they were discrete variables instead of sampling evenly in the continuous space.
    
    ''' 
    def __init__(self, domain: Domain, random_state: np.random.RandomState=None):
        self.domain = domain
        self._rstate = random_state if random_state else np.random.RandomState()
    
    def suggest_experiments(self, num_experiments: int) -> DataSet:
        """ Suggest experiments for a random experimental design 
        
        Parameters
        ---------- 
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        
        Returns
        -------
        ds
            A `Dataset` object with the random design
        """
        design = Design(self.domain, num_experiments, 'random')

        for i, variable in enumerate(self.domain.variables):
            if variable.is_objective:
                continue
            if variable.variable_type == 'continuous':
                values = self._random_continuous(variable, num_experiments)
                indices = None
            elif variable.variable_type == 'discrete':
                indices, values = self._random_discrete(variable, num_experiments)
            elif variable.variable_type == 'descriptors':
                indices, values = self._random_descriptors(variable, num_experiments)
            else:
                raise DomainError(f"Variable {variable} is not one of the possible variable types (continuous, discrete or descriptors).")

            design.add_variable(variable.name, values, indices=indices)
        
        ds = design.to_dataset()
        ds[('strategy', 'METADATA')] = "Random"
        return ds

    def _random_continuous(self, variable: ContinuousVariable,
                           num_samples: int) -> np.ndarray:
        """Generate a random design for a given continuous variable"""
        sample = self._rstate.rand(num_samples, 1)
        b = variable.lower_bound*np.ones([num_samples, 1])
        values = b + sample*(variable.upper_bound-variable.lower_bound)
        return np.atleast_2d(values).T

    def _random_discrete(self, variable: DiscreteVariable,
                        num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random design for a given discrete variable"""
        indices = self._rstate.randint(0, variable.num_levels, size=num_samples)
        values = variable.levels[indices, :]
        values.shape = (num_samples, 1)
        indices.shape = (num_samples, 1)
        return indices, values

    def _random_descriptors(self, variable: DescriptorsVariable,
                            num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a design for a given descriptors variable"""
        indices = self._rstate.randint(0, variable.num_examples-1, size=num_samples)
        values = variable.ds.data_to_numpy()[indices, :]
        values.shape = (num_samples, variable.num_descriptors)
        indices.shape = (num_samples, 1)
        return indices, values

class LHS(Strategy):
    ''' Latin hypercube sampling (LHS) strategy for experiment suggestion

    Parameters
    ---------- 
    domain: `summit.domain.Domain`
        A summit domain object
    random_state: `np.random.RandomState``
        A random state object to seed the random generator

    Examples
    --------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import Random
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = LHS(domain, random_state=np.random.RandomState(3))
    >>> strategy.suggest_experiments(5)
    NAME  temperature  flowrate_a  flowrate_b strategy
    0            95.0        0.46        0.38      LHS
    1            65.0        0.14        0.14      LHS
    2            55.0        0.22        0.30      LHS
    3            85.0        0.30        0.46      LHS
    4            75.0        0.38        0.22      LHS

    ''' 
    def __init__(self, domain: Domain, random_state: np.random.RandomState=None):
        self.domain = domain
        self._rstate = random_state if random_state else np.random.RandomState()

    def suggest_experiments(self, num_experiments, 
                            criterion='center', unique=False,
                            exclude = []) -> DataSet:
        """ Generate latin hypercube intial design 
        
        Parameters
        ---------- 
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        criterion: str (optional, Default='center')
            The criterion used for the LHS.  Allowable values are "center" or "c", "maximin" or "m", 
            "centermaximin" or "cm", and "correlation" or "corr". 
        unique: bool (Default=True)
            Determines if all suggested experiments should be unique
        exclude: array like
            List of variable names that should be excluded
            from the design. 
        
        Returns
        -------
        ds
            A `Dataset` object with the random design
        """
        design = Design(self.domain, num_experiments, 'Latin design', exclude=exclude)
        
        #Instantiate the random design class to be used with discrete variables
        rdesigner = Random(self.domain, random_state=self._rstate)

        num_discrete = self.domain.num_discrete_variables()
        n = self.domain.num_continuous_dimensions()
        if num_discrete < n:
            samples = lhs(n, samples=num_experiments, criterion=criterion, 
                          random_state=self._rstate)
        
        design.lhs = samples
        k=0
        for variable in self.domain.variables:
            if variable.name in exclude:
                continue

            if variable.is_objective:
                continue
                
            #For continuous variable, use samples directly
            if variable.variable_type == 'continuous':
                b = variable.lower_bound*np.ones(num_experiments)
                values = b + samples[:, k]*(variable.upper_bound-variable.lower_bound)
                values = np.atleast_2d(values)
                indices = None
                k+=1

            #For discrete variable, randomly choose
            elif variable.variable_type == 'discrete':
                indices, values = rdesigner._random_discrete(variable, num_experiments)

            #For descriptors variable, choose closest point by euclidean distance
            elif variable.variable_type == 'descriptors':
                num_descriptors = variable.num_descriptors
                normal_arr = variable.ds.zero_to_one()
                indices = _closest_point_indices(samples[:, k:k+num_descriptors],
                                                 normal_arr, unique=unique)
               
                values = normal_arr[indices[:, 0], :]
                var_min = variable.ds.loc[:, variable.ds.data_columns].min(axis=0).to_numpy()
                var_min = np.atleast_2d(var_min)
                var_max = variable.ds.loc[:, variable.ds.data_columns].max(axis=0).to_numpy()
                var_max = np.atleast_2d(var_max)
                var_range = var_max-var_min
                values_scaled = var_min + values*var_range
                values= values_scaled
                values.shape = (num_experiments, num_descriptors)
                k+=num_descriptors

            else:
                raise DomainError(f"Variable {variable} is not one of the possible variable types (continuous, discrete or descriptors).")

            design.add_variable(variable.name, values, indices=indices)
        ds = design.to_dataset()
        ds[('strategy', 'METADATA')] = "LHS"
        return ds


"""
The lhs code was copied from pyDoE and was originally published by 
the following individuals for use with Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    
    website: forge.scilab.org/index.php/p/scidoe/sourcetree/master/macros
Much thanks goes to these individuals. It has been converted to Python by 
Abraham Lee.

"""
def lhs(n, samples=None, criterion=None, iterations=None, random_state=None):
    """
    Generate a latin-hypercube design
    
    Parameters
    ----------
    n : int
        The number of factors to generate samples for
    
    Optional
    --------
    samples : int
        The number of samples to generate for each factor (Default: n)
    criterion : str
        Allowable values are "center" or "c", "maximin" or "m", 
        "centermaximin" or "cm", and "correlation" or "corr". If no value 
        given, the design is simply randomized.
    iterations : int
        The number of iterations in the maximin and correlations algorithms
        (Default: 5).
    
    Returns
    -------
    H : 2d-array
        An n-by-samples design matrix that has been normalized so factor values
        are uniformly spaced between zero and one.
    
    Example
    -------
    >>> import numpy as np
    
    A 3-factor design (defaults to 3 samples)::
    
        >>> lhs(3, random_state=np.random.RandomState(3))
        array([[0.5036092 , 0.73574763, 0.6320977 ],
               [0.70852844, 0.63098232, 0.09696825],
               [0.1835993 , 0.23604927, 0.6838224 ]])
       
    A 4-factor design with 6 samples::
    
        >>> lhs(4, samples=6, random_state=np.random.RandomState(3))
        array([[0.3419112 , 0.54641455, 0.3383127 , 0.59847714],
               [0.88058751, 0.11802464, 0.61270915, 0.4094722 ],
               [0.09179965, 0.40680164, 0.18759755, 0.20120715],
               [0.67066365, 0.94885632, 0.90674229, 0.85947796],
               [0.60819067, 0.31604885, 0.04848412, 0.08513793],
               [0.31549116, 0.75980901, 0.70987541, 0.7358502 ]])
       
    A 2-factor design with 5 centered samples::
    
        >>> lhs(2, samples=5, criterion='center', random_state=np.random.RandomState(3))
        array([[0.7, 0.7],
               [0.1, 0.1],
               [0.5, 0.9],
               [0.3, 0.3],
               [0.9, 0.5]])
       
    A 3-factor design with 4 samples where the minimum distance between
    all samples has been maximized::
    
        >>> lhs(3, samples=4, criterion='maximin', random_state=np.random.RandomState(3))
        array([[0.07987376, 0.37639351, 0.92316265],
               [0.25650657, 0.7314332 , 0.12061145],
               [0.55174153, 0.00530644, 0.56933076],
               [0.79401553, 0.9975753 , 0.47950751]])
       
    A 4-factor design with 5 samples where the samples are as uncorrelated
    as possible (within 10 iterations)::
    
        >>> lhs(4, samples=5, criterion='correlation', iterations=10, random_state=np.random.RandomState(3))
        array([[0.72982881, 0.91177082, 0.73525098, 0.71817256],
               [0.37858939, 0.48816197, 0.40597524, 0.10216552],
               [0.80479638, 0.37925862, 0.85185049, 0.49136664],
               [0.11015958, 0.65569746, 0.22511706, 0.88302024],
               [0.41029344, 0.14162956, 0.05818095, 0.24144858]])
    """
    H = None
    random_state = random_state if random_state else np.random.RandomState()

    if samples is None:
        samples = n
    
    if criterion is not None:
        assert criterion.lower() in ('center', 'c', 'maximin', 'm', 
            'centermaximin', 'cm', 'correlation', 
            'corr'), 'Invalid value for "criterion": {}'.format(criterion)
    else:
        H = _lhsclassic(n, samples, random_state)

    if criterion is None:
        criterion = 'center'
    
    if iterations is None:
        iterations = 5
        
    if H is None:
        if criterion.lower() in ('center', 'c'):
            H = _lhscentered(n, samples, random_state)
        elif criterion.lower() in ('maximin', 'm'):
            H = _lhsmaximin(n, samples, iterations, 'maximin', random_state)
        elif criterion.lower() in ('centermaximin', 'cm'):
            H = _lhsmaximin(n, samples, iterations, 'centermaximin', random_state)
        elif criterion.lower() in ('correlation', 'corr'):
            H = _lhscorrelate(n, samples, iterations, random_state)
    
    return H

################################################################################

def _lhsclassic(n, samples, random_state):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = random_state.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a
    
    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = random_state.permutation(range(samples))
        H[:, j] = rdpoints[order, j]
    
    return H
    
################################################################################

def _lhscentered(n, samples, random_state):
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)    
    
    # Fill points uniformly in each interval
    u = random_state.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    _center = (a + b)/2
    
    # Make the random pairings
    H = np.zeros_like(u)
    for j in range(n):
        H[:, j] = random_state.permutation(_center)
    
    return H
    
################################################################################

def _lhsmaximin(n, samples, iterations, lhstype, 
                random_state):
    maxdist = 0
    
    # Maximize the minimum distance between points
    for i in range(iterations):
        if lhstype=='maximin':
            Hcandidate = _lhsclassic(n, samples, random_state)
        else:
            Hcandidate = _lhscentered(n, samples, random_state)
        
        d = _pdist(Hcandidate)
        if maxdist<np.min(d):
            maxdist = np.min(d)
            H = Hcandidate.copy()
    
    return H

################################################################################

def _lhscorrelate(n, samples, iterations,
                  random_state):
    mincorr = np.inf
    
    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples, random_state)
        R = np.corrcoef(Hcandidate)
        if np.max(np.abs(R[R!=1]))<mincorr:
            mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
            # print('new candidate solution found with max,abs corrcoef = {}'.format(mincorr))
            H = Hcandidate.copy()
    
    return H
    
################################################################################

def _pdist(x):
    """
    Calculate the pair-wise point distances of a matrix
    
    Parameters
    ----------
    x : 2d-array
        An m-by-n array of scalars, where there are m points in n dimensions.
    
    Returns
    -------
    d : array
        A 1-by-b array of scalars, where b = m*(m - 1)/2. This array contains
        all the pair-wise point distances, arranged in the order (1, 0), 
        (2, 0), ..., (m-1, 0), (2, 1), ..., (m-1, 1), ..., (m-1, m-2).
    
              
    """
    
    x = np.atleast_2d(x)
    assert len(x.shape)==2, 'Input array must be 2d-dimensional'
    
    m, n = x.shape
    if m<2:
        return []
    
    d = []
    for i in range(m - 1):
        for j in range(i + 1, m):
            d.append((sum((x[j, :] - x[i, :])**2))**0.5)
    return np.array(d)