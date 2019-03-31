#!/usr/bin/env python
""" Pandas dataframes with column metadata

Notes:
    Copied from https://notes.mikejarrett.ca/storing-metadata-in-pandas-dataframes/
"""

__author__ = "Mike Jarret"

import pandas as pd

class MetaDataFrame(pd.DataFrame):

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