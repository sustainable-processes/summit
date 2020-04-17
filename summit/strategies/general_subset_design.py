"""
Original GSD code Copyright (C) 2018 - Rickard Sjoegren
"""
from .base import Strategy, Design
from summit.domain import (Domain, Variable, ContinuousVariable, 
                          DiscreteVariable, DescriptorsVariable,
                          DomainError)
import itertools
import numpy as np
import pandas as pd
from typing import Type, Tuple

class GSDesigner(Strategy):
    def __init__(self, domain: Domain):
        self.domain = domain

    def suggest_experiments(self, reduction) -> Design:
        """
        Create a Generalized Subset Design (GSD).

        Parameters
        ----------
        reduction : int
            Reduction factor (bigger than 1). Larger `reduction` means fewer
            experiments in the design and more possible complementary designs.
        n : int
            Number of complementary GSD-designs (default 1). The complementary
            designs are balanced analogous to fold-over in two-level fractional
            factorial designs.

        Returns
        -------
        design: `Design`
            A `Design` object with the random design
        """
        num_var = self.domain.num_variables(include_outputs=False)
        levels = []
        for v in self.domain.variables:
            if v.variable_type == 'discrete' and not v.is_objective:
                levels += [len(v.levels)]
            elif v.variable_type == 'continuous' and not v.is_objective:
                levels += [2]
            else:
                raise DomainError(f"Variable {v} is not one of the possible variable types (continuous, discrete or descriptors).")
        num_experiments = np.prod(levels)/reduction
        design = Design(self.domain, num_experiments, 'gsd')
        samples = gsd(levels, reduction, n=1)
        
        return samples

def gsd(levels, reduction, n=1):

    try:
        assert all(isinstance(v, int) for v in levels), \
            'levels has to be sequence of integers'
        assert isinstance(reduction, int) and reduction > 1, \
            'reduction has to be integer larger than 1'
        assert isinstance(n, int) and n > 0, \
            'n has to be positive integer'
    except AssertionError as e:
        raise ValueError(e)

    partitions = _make_partitions(levels, reduction)
    latin_square = _make_latin_square(reduction)
    ortogonal_arrays = _make_orthogonal_arrays(latin_square, len(levels))

    try:
        designs = [_map_partitions_to_design(partitions, oa) - 1 for oa in
                   ortogonal_arrays]
    except ValueError:
        raise ValueError('reduction too large compared to factor levels')

    if n == 1:
        return designs[0]
    else:
        return designs[:n]


def _make_orthogonal_arrays(latin_square, n_cols):
    """
    Augment latin-square to the specified number of columns to produce
    an orthogonal array.
    """
    p = len(latin_square)

    first_row = latin_square[0]
    A_matrices = [np.array([[v]]) for v in first_row]

    while A_matrices[0].shape[1] < n_cols:
        new_A_matrices = list()

        for i, A_matrix in enumerate(A_matrices):
            sub_a = list()
            for constant, other_A in zip(first_row,
                                         np.array(A_matrices)[latin_square[i]]):
                constant_vec = np.repeat(constant, len(other_A))[:, np.newaxis]
                combined = np.hstack([constant_vec, other_A])
                sub_a.append(combined)

            new_A_matrices.append(np.vstack(sub_a))

        A_matrices = new_A_matrices

        if A_matrices[0].shape[1] == n_cols:
            break

    return A_matrices


def _map_partitions_to_design(partitions, ortogonal_array):
    """
    Map partitioned factor to final design using orthogonal-array produced
    by augmenting latin square.
    """
    assert len(
        partitions) == ortogonal_array.max() + 1 and ortogonal_array.min() == 0, \
        'Orthogonal array indexing does not match partition structure'

    mappings = list()
    for row in ortogonal_array:
        if any(not partitions[p][factor] for factor, p in enumerate(row)):
            continue

        partition_sets = [partitions[p][factor] for factor, p in enumerate(row)]
        mapping = list(itertools.product(*partition_sets))
        mappings.append(mapping)

    return np.vstack(mappings)


def _make_partitions(factor_levels, num_partitions):
    """
    Balanced partitioning of factors.
    """
    partitions = list()
    for partition_i in range(1, num_partitions + 1):
        partition = list()

        for num_levels in factor_levels:
            part = list()
            for level_i in range(1, num_levels):
                index = partition_i + (level_i - 1) * num_partitions
                if index <= num_levels:
                    part.append(index)

            partition.append(part)

        partitions.append(partition)

    return partitions


def _make_latin_square(n):
    numbers = np.arange(n)
    latin_square = np.vstack([np.roll(numbers, -i) for i in range(n)])
    return latin_square