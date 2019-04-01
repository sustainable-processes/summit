import summit
from summit.utils import MetaDataFrame
import pandas as pd

#Constants
DATA_PATH   =  summit.__path__[0] + '/data/'
SOLVENT_DESCRIPTOR_DATA_FILE =  DATA_PATH + 'solvent_descriptors.csv'
SOLVENT_METADATA_VARIABLES = ['stenutz_name', 'cosmo_name', 'cas_number', 'chemical_formula']
UCB_PHARMA_APPROVED_LIST = DATA_PATH + 'ucb_pharma_approved_list.csv'

def make_dataset(df, metadata_columns, copy=True):
    '''Create a pandas dataframe with the specified columns as indices
    
    Arguments:
        df: Pandas dataframe
        metadata_columsn: columns of the dataframe that should be treated as indices (i.e., metadata)
    '''
    index = pd.MultiIndex.from_frame(df.loc[:, metadata_columns])
    if copy:
        new_df = df.copy()
    else:
        new_df = df

    new_df = new_df.drop(metadata_columns, axis=1)
    return new_df.set_index(index)


#Load solvent descriptor dataset
_solvent_candidates = pd.read_csv(SOLVENT_DESCRIPTOR_DATA_FILE)
solvent_ds = make_dataset(_solvent_candidates, SOLVENT_METADATA_VARIABLES)

#Load UCB Pharma approved list
ucb_list = pd.read_csv(UCB_PHARMA_APPROVED_LIST)

