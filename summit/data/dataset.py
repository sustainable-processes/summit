#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

def normalize_df(df: pd.DataFrame):
    s = StandardScaler()
    transformed_values = s.fit_transform(df.descriptors_to_numpy())
    return pd.DataFrame(transformed_values,
                        columns = df.columns,
                        index=df.index)

def zero_one_scale_df(df):
    values = df.descriptors_to_numpy()
    maxes = np.max(values, axis=0)
    mins = np.min(values, axis=0)
    ranges = maxes-mins
    scaled = (values-mins)/ranges
    return pd.DataFrame(scaled,
                        columns = df.columns,
                        index = df.index)

class DataSet(pd.core.frame.DataFrame):
    """ A represenation of a dataset

    This is basically a pandas dataframe with a set of "metadata" columns
    that will be removed when the dataframe is converted to a numpy array

    Notes
    ----
    Based on https://notes.mikejarrett.ca/storing-metadata-in-pandas-dataframes/
    """
    @property  
    def _constructor(self):
        return DataSet       
        
    def __getitem__(self, key):
        is_mi_columns = isinstance(self.columns, pd.MultiIndex)
        if is_mi_columns and 'NAME' in self.columns.names and type(key)==str:
            tupkey = [x for x in self.columns if x[0]==key]
            if len(tupkey) == 1:
                key = tupkey[0]
            elif len(tupkey) > 1:
                raise ValueError('NAME level column labels must be unique')
        return super().__getitem__(key)

    def __unicode__(self):
        is_mi_columns = isinstance(self.columns, pd.MultiIndex)
        if is_mi_columns and 'NAME' in self.columns.names:

            newdf = self.copy()
            newdf.columns = self.columns.get_level_values('NAME')
            return newdf.__unicode__()
        return super().__unicode__()
    
    def _repr_html_(self):
        is_mi_columns = isinstance(self.columns, pd.MultiIndex)
        if is_mi_columns and 'NAME' in self.columns.names:

            newdf = self.copy()
            columns = self.columns.get_level_values('NAME').to_numpy()
            newdf.columns = columns 
            return newdf._repr_html_()
        return super()._repr_html_() 

    def descriptors_to_numpy(self) -> int:
        '''Return dataframe with the metadata columns removed'''
        result = super().to_numpy()
        metadata_columns = []
        for i, column in enumerate(self.columns):
            if column[1] == 'METADATA':
                metadata_columns.append(i)
        mask = np.ones(len(self.columns), dtype=bool)
        mask[metadata_columns] = False
        return result[:, mask]

    @property
    def num_data_columns(self) -> int:
        col_types = self.columns.get_level_values('TYPE')
        values, counts = np.unique(col_types, return_counts=True)
        i= np.where(values=='DATA')
        return counts[i][0]

    @property
    def metadata_columns(self):
        return [column[0] for column in self.columns if column[1]=='METADATA']

    @staticmethod
    def from_df(df: pd.DataFrame, metadata_columns: List=[], 
                units: List = []):
        '''Create Dataset from a pandas dataframe
    
        Arguments
        ----------
        df: pandas.DataFrame
            Dataframe to be converted to a DataSet
        metadata_columns: list, optional
            names of the columns in the dataframe that are metadata columns
        units: list, optional 
            A list of objects representing the units of the columns
        '''
        column_names = df.columns.to_numpy()
        if metadata_columns:
            types = ['METADATA' if x in metadata_columns else 'DATA' for x in df.columns]
        else:
            types = ['DATA' for _ in range(len(column_names))]
        arrays = [column_names, types]
        levels = ['NAME', 'TYPE']
        if units:
            arrays.append(units)
            levels.append('UNITS')
        tuples=list(zip(*arrays))
        columns = pd.MultiIndex.from_tuples(tuples, names=levels)

        return DataSet(df.to_numpy(), columns=columns, index=df.index)

    def zero_to_one(self, small_tol=1.0e-5):
        ''' Scale the data columns between zero and one 

        Each of the data columns is scaled between zero and one 
        based on the maximum and minimum values of each column

        Arguments
        ---------
        remove_small: float
            The minimum value of any value in the final scaled array. 
            This is used to prevent very small values that will cause
            issues in later calcualtions

        Returns
        -------
        scaled: numpy.ndarray
            A numpy array with the scaled data columns

        ''' 
        values = self.descriptors_to_numpy()
        values = values.astype(np.float64)
        maxes = np.max(values, axis=0)
        mins = np.min(values, axis=0)
        ranges = maxes-mins
        scaled = (values-mins)/ranges
        scaled[abs(scaled) < small_tol] = 0.0
        return scaled

