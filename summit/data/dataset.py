#!/usr/bin/env python
import pandas as pd
import numpy as np
from typing import List

class DataSet(pd.core.frame.DataFrame):
    """ A represenation of a dataset

    This is basically a pandas dataframe with a set of "metadata" columns
    that will be removed when the dataframe is converted to a numpy array

    Notes
    ----
    Based on https://notes.mikejarrett.ca/storing-metadata-in-pandas-dataframes/
    """
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

    @staticmethod
    def read_csv(filepath_or_buffer, **kwargs):
        """Create a DataSet from a csv"""
        header = kwargs.get('header', [0,1])
        index_col = kwargs.get('index_col', 0)
        df = pd.read_csv(filepath_or_buffer, header=header, index_col=index_col)
        return DataSet(df.to_numpy(), columns=df.columns, index=df.index)

    def zero_to_one(self, small_tol=1.0e-5) -> np.ndarray:
        ''' Scale the data columns between zero and one 

        Each of the data columns is scaled between zero and one 
        based on the maximum and minimum values of each column

        Arguments
        ---------
        small_tol: float, optional
            The minimum value of any value in the final scaled array. 
            This is used to prevent very small values that will cause
            issues in later calcualtions. Defaults to 1e-5.

        Returns
        -------
        scaled: numpy.ndarray
            A numpy array with the scaled data columns

        Notes
        ----- 
        This method does not change the internal values of the data columns in place.

        ''' 
        values = self.descriptors_to_numpy()
        values = values.astype(np.float64)
        maxes = np.max(values, axis=0)
        mins = np.min(values, axis=0)
        ranges = maxes-mins
        scaled = (values-mins)/ranges
        scaled[abs(scaled) < small_tol] = 0.0
        return scaled

    def standardize(self, small_tol=1.0e-5) -> np.ndarray:
        """Standardize data columns by removing the mean and scaling to unit variance

        The standard score of each data column is calculated as:
            z = (x - u) / s
        where `u` is the mean of the columns and `s` is the standard deviation of 
        each data column
        
        Parameters
        ---------- 
        small_tol: float, optional
            The minimum value of any value in the final scaled array. 
            This is used to prevent very small values that will cause
            issues in later calcualtions. Defaults to 1e-5.
        
        Returns
        -------
        standard: np.ndarray
            Numpy array of the standardized data columns

        Notes
        ----- 
        This method does not change the internal values of the data columns in place.
        
        """
        values = self.descriptors_to_numpy()
        values = values.astype(np.float64)
        mean = np.mean(values, axis=0)
        sigma = np.std(values, axis=0)
        standard = (values-mean)/sigma
        standard[abs(standard) < small_tol] = 0.0
        return standard

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

    # @property
    # def num_data_columns(self) -> int:
    #     col_types = self.columns.get_level_values('TYPE')
    #     values, counts = np.unique(col_types, return_counts=True)
    #     i= np.where(values=='DATA')
    #     return counts[i][0]

    @property
    def metadata_columns(self):
        '''Names of the metadata columns'''
        return [column[0] for column in self.columns if column[1]=='METADATA']

    @property
    def data_columns(self):
        '''Names of the data columns'''
        return [column[0] for column in self.columns if column[1]=='DATA']
    
    def insert(self, loc, column, value, type='DATA', units=None, allow_duplicates=False):
        super().insert(loc, column, value, allow_duplicates)
        import ipdb; ipdb.set_trace()
        self.columns[loc][0] = column
        self.columns[loc][1] = type
        self.columns[loc][2] = units

class ResultSet(DataSet):
    data_column_types = ['input', 'output']
