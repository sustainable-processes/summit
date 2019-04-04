from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pyDOE import lhs

from abc import ABC, abstractmethod
from typing import Type, Tuple

class Designer(ABC):
    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def generate_experiments(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Design:
    """Representation of an experimental desgin
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the design
    num_samples: int
        Number of samples in the design  

    Examples
    --------
    >>> domain = Domain()
    >>> domain += ContinuousVariable('temperature', 'reaction temperature', [1, 100])
    >>> initial_design = Design()
    >>> initial_design.add_variable('temperature', 
                                    np.array([[100, 120, 150]]))

    """ 
    def __init__(self, domain: Domain, num_samples, design_type: str):
        self._variable_names = [variable.name for variable in domain.variables]
        self._indices = domain.num_variables * [0]
        self._values = domain.num_variables * [0]
        self.num_samples = num_samples
        self.design_type = design_type
        self._domain = domain

    def add_variable(self, variable_name: str, 
                     values: np.ndarray, indices: np.ndarray=None):
        """ Add a variable to a design 
        
        Parameters
        ---------- 
        variable_name: str
            Name of the variable to be added. Must already be in the domain.
        values: numpy.ndarray
            Values of the design points in the variable
        indices: numpy.ndarray, optional
            Indices of the design points in the variable
        
        Raises
        ------
        ValueError
            If indices or values are not a two-dimensional array.
        """
        variable_index = self._get_variable_index(variable_name)
        if values.ndim < 2:
            raise ValueError("Values must be 2 dimensional. Use np.atleast_2d.")
        if indices is not None:
            if indices.ndim < 2:
                raise ValueError("Indices must be 2 dimensional. Use np.atleast_2d.")
            self._indices[variable_index] = indices 
        self._values[variable_index] = values

    def get_indices(self, variable_name: str=None) -> np.ndarray:
        """ Get indices of designs points  
        
        Parameters
        ---------- 
        variable_name: str, optional
            Get only the indices for a specific variable name.
        
        Returns
        -------
        indices: numpy.ndarray
            Indices of the design pionts
        
        Raises
        ------
        ValueError
            If the variable name is not in the list of variables
        """ 
        if variable_name is not None:
            variable_index = self._get_variable_index(variable_name)
            indices = self._indices[variable_index]
        else:
            indices = np.concatenate(self._indices, axis=1)
        return indices

    def get_values(self, variable_name: str=None) -> np.ndarray:
        """ Get values of designs points  
        
        Parameters
        ---------- 
        variable_name: str, optional
            Get only the values for a specific variable name.
        
        Returns
        -------
        values: numpy.ndarray
            Values of the design pionts
        
        Raises
        ------
        ValueError
            If the variable name is not in the list of variables
        """  
        if variable_name is not None:
            variable_index = self._get_variable_index(variable_name)
            values = self._values[variable_index]
        else:
            values = np.concatenate(self._values, axis=1)

        return values

    def _get_variable_index(self, variable_name: str) -> int:
        '''Method for getting the internal index for a variable'''
        if not variable_name in self._variable_names:
            raise ValueError(f"Variable {variable_name} not in domain.")

        return self._variable_names.index(variable_name)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame([])
        for i, variable in enumerate(self._domain.variables):
            if variable.variable_type == 'descriptors':
                descriptors = variable.df.iloc[self.get_indices(variable.name)[:, 0], :]
                df = pd.concat([df, descriptors.index.to_frame(index=False)], axis=1)
            else:
                df.insert(i, variable.name, self.get_values(variable.name)[:, 0])
        return df

    def _repr_html_(self):
        return self.to_frame().to_html()


class RandomDesign(Designer):
    def __init__(self, domain: Domain, random_state: np.random.RandomState=None):
        self.domain = domain
        self._rstate = random_state if random_state else np.random.RandomState()
    
    def generate_experiments(self, num_experiments: int) -> Design:
        """ Generate a random experimental design 
        
        Parameters
        ---------- 
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        
        Returns
        -------
        design: `Design`
            A `Design` object with the random design
        """
        design = Design(self.domain, num_experiments, 'Random design')

        for i, variable in enumerate(self.domain.variables):
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
        
        return design

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
        values = variable.df.values[indices, :]
        values.shape = (num_samples, variable.num_descriptors)
        indices.shape = (num_samples, 1)
        return indices, values

class LatinDesign(Designer):
    def __init__(self, domain: Domain, random_state: np.random.RandomState=None):
        self.domain = domain
        self._rstate = random_state if random_state else np.random.RandomState()

    def generate_experiments(self, num_experiments, criterion='center') -> Design:
        """ Generate latin hypercube experimental design 
        
        Parameters
        ---------- 
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        
        Returns
        -------
        design: `Design`
            A `Design` object with the latin hypercube design
        """
        design = Design(self.domain, num_experiments, 'Latin design')
        
        #Instantiate the random design class to be used with discrete variables
        rdesigner = RandomDesign(self.domain, random_state=self._rstate)

        num_discrete = self.domain.num_discrete_variables
        n = self.domain.num_continuous_dimensions
        if num_discrete < n:
            samples = lhs(n, samples=num_experiments, criterion=criterion)
        
        k=0
        for variable in self.domain.variables:
            #For continuous variable, use samples directly
            if variable.variable_type == 'continuous':
                b = variable.lower_bound*np.ones(num_experiments)
                values = b + samples[:, k]*(variable.upper_bound-variable.lower_bound)
                values = np.atleast_2d(values).T
                indices = None
                k+=1

            #For discrete variable, randomly choose
            elif variable.variable_type == 'discrete':
                indices, values = rdesigner._random_discrete(variable, num_experiments)

            #For descriptors variable, choose closest point by euclidean distance
            elif variable.variable_type == 'descriptors':
                num_descriptors = variable.num_descriptors
                #TODO: to take into account subsetting
                indices = _closest_point_indices(samples[:, k:k+num_descriptors+1],
                                                 variable.normalized.values) 
                values = variable.df.values[indices[:, 0], :]
                values.shape = (num_experiments, num_descriptors)
                k+=num_descriptors-1

            else:
                raise DomainError(f"Variable {variable} is not one of the possible variable types (continuous, discrete or descriptors).")

            design.add_variable(variable.name, values, indices=indices)
        
        return design      

def _closest_point_indices(design_points, candidate_matrix):
    '''Return the indices of the closest point in the candidate matrix to each design point'''
    indices = [_closest_point_index(design_point, candidate_matrix)
                 for design_point in design_points]
    indices = np.array(indices)
    return np.atleast_2d(indices).T

def _closest_point_index(design_point, candidate_matrix):
    '''Return the index of the closest point in the candidate matrix'''
    distances = _design_distances(design_point, candidate_matrix)
    return np.argmin(np.atleast_2d(distances)) 

def _design_distances(design_point,candidate_matrix):
    ''' Return the distances between a design_point and all candidates'''
    diff = design_point - candidate_matrix
    squared = np.power(diff, 2)
    summed  = np.sum(squared, axis=1)
    root_square = np.sqrt(summed)
    return root_square

        