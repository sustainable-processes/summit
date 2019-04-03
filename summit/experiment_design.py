from summit.domain import Domain, Variable, ContinuousVariable, DiscreteVariable, DescriptorsVariable
from abc import ABC, abstractmethod
from typing import Type
import numpy as np

class Designer(ABC):
    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def generate_experiments(self):
        raise NotImplementedError("Subclasses should implement this method.")

class Design:
    def __init__(self, domain: Domain, num_samples):
        self._variable_names = [variable.name for variable in domain.variables]
        self._indices = domain.num_variables * [np.empty((num_samples))]
        self._values = domain.num_variables * [np.empty(num_samples)]

    def add_variable(self, variable_name: str, 
                     indices: np.ndarray, values: np.ndarray):

        variable_index = self.get_variable_index(variable_name)
        self._indices[variable_index] = indices 
        self._values[variable_index] = values

    def get_indices(self, variable_name: str=None) -> np.ndarray:
        if variable_name is not None:
            variable_index = self.get_variable_index(variable_name)
            indices = self._indices[variable_index]
        else:
            indices = np.concatenate(self._indices, axis=1)

        return indices

    def get_values(self, variable_name: str=None) -> np.ndarray:
        if variable_name is not None:
            variable_index = self.get_variable_index(variable_name)
            values = self._values[variable_index]
        else:
            values = np.concatenate(self._values, axis=1)

        return values

    def get_variable_index(self, variable_name: str) -> int:
        if not variable_name in self._variable_names:
            raise ValueError(f"Variable {variable_name} not in domain.")

        return self._variable_names.index(variable_name)

class RandomDesign(Designer):
    def __init__(self, domain: Domain):
        self.domain = domain
    
    def generate_experiments(self, num_experiments: int):
        design 

        for i, variable in enumerate(domain.variables):
            if isinstance(variable, ContinuousVariable):
                values = random_continuous(variable)
                indices = None
            elif isinstance(variable, DiscreteVariable):
                indices, values = random_discrete(variable)
            elif isinstance(variable, DescriptorsVariable):
                indices, values = random_descriptors(variable)

            design[:, i] = values

class ModifiedLatinDesign(Designer):
    def __init__(self, domain: Domain):
        Design.__init__(self, domain)

    def generate_experiments(self, n_experiments, criterion='maximin'):
        pass
        
        