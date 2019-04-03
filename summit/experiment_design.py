from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)
from abc import ABC, abstractmethod
from typing import Type, Tuple
import numpy as np

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
    def __init__(self, domain: Domain, num_samples):
        self._variable_names = [variable.name for variable in domain.variables]
        self._indices = domain.num_variables * [np.zeros((num_samples, 1))]
        self._values = domain.num_variables * [np.zeros((num_samples, 1))]
        self.num_samples = num_samples

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
        if indices is not None:
            if indices.ndim < 2 or values.ndim < 2:
                raise ValueError("Indices and values must be 2 dimensional. Use np.atleast_2d.")
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

class RandomDesign(Designer):
    def __init__(self, domain: Domain, random_state: np.random.RandomState=None):
        self.domain = domain
        self._rstate = random_state if random_state else np.random.RandomState()
    
    def generate_experiments(self, num_experiments: int) -> Design:
        """ Generate an experimental design 
        
        Parameters
        ---------- 
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        
        Returns
        -------
        design: `Design`
            A `Design` object with the random design
        """
        design = Design(self.domain, num_experiments)

        for i, variable in enumerate(self.domain.variables):
            if hasattr(variable, 'lower_bound') and hasattr(variable, 'upper_bound'):
                values = self._random_continuous(variable, num_experiments)
                indices = None
            elif hasattr(variable, 'levels') and hasattr(variable, 'num_levels'):
                indices, values = self._random_discrete(variable, num_experiments)
            elif hasattr(variable, 'df') and hasattr('num_examples'):
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
        return values

    def _random_discrete(self, variable: DiscreteVariable,
                        num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random design for a given discrete variable"""
        indices = self._rstate.randint(0, variable.num_levels-1, size=num_samples)
        values = variable.levels[indices]
        return indices, values

    def _random_descriptors(self, variable: DescriptorsVariable,
                            num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a design for a given descriptors variable"""
        indices = self._rstate.randint(0, variable.num_examples-1, size=num_samples)
        values = variable.df.values[indices, :]
        return indices, values

class ModifiedLatinDesign(Designer):
    def __init__(self, domain: Domain):
        Design.__init__(self, domain)

    def generate_experiments(self, n_experiments, criterion='maximin'):
        pass
        
        