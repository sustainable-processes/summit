from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Type, Tuple

class Design:
    """Representation of an experimental design
    
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
        self._indices = domain.num_variables() * [0]
        self._values = domain.num_variables() * [0]
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

    def get_indices(self, variable_name: str) -> np.ndarray:
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
        variable_index = self._get_variable_index(variable_name)
        indices = self._indices[variable_index]
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
            if variable.is_output:
                continue
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

    # def coverage(self, design_indices, search_matrix=None,
    #              metric=closest_point_distance):
    #     ''' Get coverage statistics for a design based 
    #     Arguments:
    #         design_indices: Indices in the search matrix of the design points
    #         search_matrix (optional): A matrix of descriptors used for calculating the coverage. By default, the 
    #                                   descriptor matrix in the instance of solvent select will be used as the search 
    #                                   matrix
    #         metric (optional): A function for calculating the coverage. By default this is the closest point. 
    #                            The function should take a design point as its first argument and a candidate matrix 
    #                            as its second argument. 
    #     Notes:
    #         Coverage statistics are calculated by finding the distance between each point in the search matrix 
    #         and the closest design point. The statistics are mean, standard deviation, median, maximum, and minimum
    #         of the distances. 
    #     Returns
    #         An instance of `DesignCoverage`
            
    #     '''
    #     if search_matrix is None:
    #         search_matrix = self.descriptor_df.values

    #     mask = np.ones(search_matrix.shape[0], dtype=bool)
    #     mask[design_indices] = False
    #     distances = [metric(row, search_matrix[design_indices, :])
    #                 for row in search_matrix[mask, ...]]
    #     mean = np.average(distances)
    #     std_dev = np.std(distances)
    #     median = np.median(distances)
    #     max = np.max(distances)
    #     min = np.min(distances)
    #     return DesignCoverage(
    #                     mean=mean,
    #                     std_dev=std_dev,
    #                     median=median,
    #                     max = max,
    #                     min = min
    #                     )

    def _repr_html_(self):
        return self.to_frame().to_html()

class DesignCoverage:
    properties = ['mean', 'std_dev', 'median', 'max', 'min']

    def __init__(self, mean=None, 
                       std_dev=None, 
                       median=None, 
                       max=None, 
                       min=None):
        self._mean = mean
        self._std_dev = std_dev
        self._median = median
        self._max= max
        self._min = min

    @property
    def mean(self):
        return self._mean

    @property
    def std_dev(self):
        return self._std_dev

    @property
    def median(self):
        return self._median

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    def __repr__(self):
        values = ''.join([f"{property}:{getattr(self, property)}, " for property in self.properties])
        return f'''DesignCoverage({values.rstrip(", ")})'''
    
    def get_dict(self):
        return {property: getattr(self, property) for property in self.properties}

    def get_array(self):
        return [getattr(self, property) for property in self.properties]

    @staticmethod
    def average_coverages(coverages):
        '''Average multiple design coverages
        
        Arguments:
            coverages: a list of `DesignCoverage` objects.
        '''
        #Check that argument is  a list of coverages
        for coverage in coverages:
            assert isinstance(coverage, DesignCoverage)

        avg_mean = np.average([coverage.mean for coverage in coverages])
        avg_std_dev = np.average([coverage.std_dev for coverage in coverages])
        avg_median = np.average([coverage.median for coverage in coverages])
        avg_max = np.average([coverage.max for coverage in coverages])
        avg_min = np.average([coverage.min for coverage in coverages])
        return DesignCoverage(
            mean = avg_mean,
            std_dev = avg_std_dev,
            median = avg_median,
            max=avg_max,
            min = avg_min
        )

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
            masked_candidates = candidate_matrix[mask, :]
            point_index = _closest_point_index(design_point, masked_candidates)
            actual_index = np.where(candidate_matrix==masked_candidates[point_index, :])[0][0]
            indices[i] = actual_index
            mask[actual_index] = False
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

        