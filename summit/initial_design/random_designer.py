from .base import Designer, Design
from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)

import numpy as np
import pandas as pd
from typing import Type, Tuple

class RandomDesigner(Designer):
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
        values = variable.ds.descriptors_to_numpy()[indices, :]
        values.shape = (num_samples, variable.num_descriptors)
        indices.shape = (num_samples, 1)
        return indices, values
