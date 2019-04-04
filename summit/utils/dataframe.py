#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_df(df: pd.DataFrame):
    s = StandardScaler()
    transformed_values = s.fit_transform(df.values)
    return pd.DataFrame(transformed_values,
                        columns = df.columns,
                        index=df.index)

def zero_one_scale_df(df: pd.DataFrame):
    values = df.values
    maxes = np.max(values, axis=0)
    mins = np.min(values, axis=0)
    ranges = maxes-mins
    scaled = (values-mins)/ranges
    return pd.DataFrame(scaled,
                        columns = df.columns,
                        index = df.index)


class MetaDataFrame(pd.DataFrame):
    """ Pandas dataframes with column metadata

    Notes:
    Copied from https://notes.mikejarrett.ca/storing-metadata-in-pandas-dataframes/
    """
    @property  
    def _constructor(self):
        return MetaDataFrame        
        
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
            newdf.columns = self.columns.get_level_values('NAME')
            return newdf._repr_html_()
        return super()._repr_html_() 