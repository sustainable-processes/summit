from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Type, Tuple

class Design:
    """Representation of an experimental desgin
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the design
    num_samples: int
        Number of samples in the design 
    design_type: str
        The name of the design type 

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

    def to_frame(self) -> pd.DataFrame:
        ''' Get design as a pandas dataframe 
        Returns
        -------
        df: pd.DataFrame
        ''' 
        df = pd.DataFrame([])
        for i, variable in enumerate(self._domain.variables):
            if variable.variable_type == 'descriptors':
                descriptors = variable.ds.iloc[self.get_indices(variable.name)[:, 0], :]
                df = pd.concat([df, descriptors.index.to_frame(index=False)], axis=1)
            else:
                df.insert(i, variable.name, self.get_values(variable.name)[:, 0])
        return df

    def _get_variable_index(self, variable_name: str) -> int:
        '''Method for getting the internal index for a variable'''
        if not variable_name in self._variable_names:
            raise ValueError(f"Variable {variable_name} not in domain.")
        return self._variable_names.index(variable_name)

    def _repr_html_(self):
        return self.to_frame().to_html()

class Designer(ABC):
    ''' Base class for designers

    All intial design strategies should inherit this base class.
    ''' 
    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def generate_experiments(self):
        raise NotImplementedError("Subclasses should implement this method.")

def _closest_point_indices(design_points, candidate_matrix, unique=False):
    '''Return the indices of the closest point in the candidate matrix to each design point'''
    if unique:
        mask = np.ones(candidate_matrix.shape[0], dtype=bool)
        indices = [0 for i in range(len(design_points))]
        for i, design_point in enumerate(design_points):
            point_index = _closest_point_index(design_point, candidate_matrix[mask, :])
            indices[i] = point_index
            mask[point_index] = False
    else:
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

        