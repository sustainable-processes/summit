import numpy as np
import pandas as pd
import GPyOpt
from GPyOpt.experiment_design.latin_design import LatinDesign


class Solvent:
    def __init__(self, metadata, descriptors):
        self._metadata = metadata
        self._descriptors = descriptors

        @property
        def metadata(self):
            return self._metadata

        @property
        def descriptors(self):
            return self._descriptors

class SolventSelect:
    '''
    Helper class for solvent selection
    
    Attributes:
        metadata_df: A pandas dataframe with metadata for the solvents
        descriptor_df: A pandas dataframe with molecular descriptors for the solvents
    '''
    def __init__(self, metadata_df, descriptors_df):
        self._metadata_df = metadata_df
        self._descriptor_df = descriptors_df

    @property
    def descriptor_df(self):
        return self._descriptor_df

    @property
    def metadata_df(self):
        return self._metadata_df

    def get_solvent_by_index(self, index):
        metadata = self._metadata_df.iloc[index, :]
        descriptors = self._descriptor_df.iloc[index, :]
        return Solvent(metadata=metadata, descriptors=descriptors)

    def get_closest_solvent(self, descriptors, filter=None, max_searches=10,
                            descriptor_matrix=None):
        '''Get the closest solvent to a provided set of descriptors
        
        Arguments:
            descriptors: numpy array of descriptors (same column order as descriptor_df)
            filter: a function that takes in a solvent and returns a boolean. Can be used to filter 
                    solvents based on, for example, available inventory
            max_searches: the maximum number of solvents to search if filter gives a value of False (default: 10)
        Returns:
            tuple of index of solvent and a solvent object for the solvent
        '''
        if descriptor_matrix is None:
            descriptor_matrix = self._descriptor_df.values

        index = closest_point_index(descriptors, descriptor_matrix)
        solvent = self.get_solvent_by_index(index)

        if not filter:
            pass
        elif not filter(solvent):
            mask = np.ones(descriptor_matrix.shape[0], dtype=bool)
            mask[index] = False
            for i in range(max_searches):
                index = closest_point_index(descriptors, descriptor_matrix[mask, ...])
                solvent = self.get_solvent_by_index(index)
                if not filter(solvent):
                    mask[index] = False
                else:
                    break

        return index, solvent

    def get_closest_solvents(self, descriptor_matrix, search_matrix=None,
                            filter=None, max_searches=10):
        '''Get the closest solvent to a provided set of descriptors
        
        Arguments:
            descriptor_matrix: numpy array of descriptors (same column order as descriptor_df) for multiple design points
            filter: a function that takes in a solvent and returns a boolean. Can be used to filter 
                    solvents based on, for example, available inventory
            max_searches: the maximum number of solvents to search if filter gives a value of False (default: 10)
        Returns:
            tuple of index of solvent and a solvent object for the solvent
        '''
        if search_matrix is None:
            search_matrix = self.descriptor_df.values
    
        results = descriptor_matrix.shape[0]*[0]
        for i, descriptors in enumerate(descriptor_matrix):
            index, solvent = self.get_closest_solvent(descriptors, descriptor_matrix=search_matrix)
            mask = np.ones(search_matrix.shape[0], dtype=bool)
            mask[index] = False
            search_matrix = search_matrix[mask, ...]
            results[i] = (index, solvent)
        return results
    
def construct_lhs_design(x, n_samples=10, criterion='center'):
    variables = [{'name': f'col_{i}', 
                  'type': 'continuous', 
                  'domain': (np.min(x[:, i]-1), np.max(x[:, i]+1))}
                 for i in range(x.shape[1])]
    space = GPyOpt.Design_space(space = variables)
    lhs = LatinDesign(space)
    return lhs.get_samples(n_samples, criterion=criterion)

def design_distances(design_point,candidate_matrix):
    ''' Return the distances between a design_point and all candidates'''
    diff = design_point - candidate_matrix
    squared = np.power(diff, 2)
    summed  = np.sum(squared, axis=1)
    root_square = np.sqrt(summed)
    return root_square

def closest_point_index(design_point, candidate_matrix):
    '''Return the index of the closest point in the candidate matrix'''
    distances = design_distances(design_point, candidate_matrix)
    return np.argmin(np.atleast_2d(distances))

def closest_point(design_point, candidate_matrix):
    index = closest_point_index(design_point, candidate_matrix)
    return candidate_matrix[index, :]
    
__all__ = ['construct_lhs_design', 'design_distances', 'closest_point_index', 'closest_point',
           'get_closest_available_solvent', 'SolventSelect']