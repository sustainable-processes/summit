from .base import Strategy, Design, _closest_point_indices
from summit.domain import (Domain, Variable, ContinuousVariable, 
                           DiscreteVariable, DescriptorsVariable,
                           DomainError)
from summit.utils.dataset import DataSet
from summit.utils.lhs import lhs
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


