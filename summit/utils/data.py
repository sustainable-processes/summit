import summit
from .dataset import DataSet
import pandas as pd

#Constants
DATA_PATH   =  summit.__path__[0] + '/data/'
SOLVENT_DESCRIPTOR_DATA_FILE =  DATA_PATH + 'solvent_descriptors.csv'
SOLVENT_INDEX = 'cas_number'
SOLVENT_METADATA_VARIABLES = ['stenutz_name', 'cosmo_name', 'chemical_formula']
UCB_PHARMA_APPROVED_LIST = DATA_PATH + 'ucb_pharma_approved_list.csv'


#Load solvent descriptor dataset
_solvent_candidates = pd.read_csv(SOLVENT_DESCRIPTOR_DATA_FILE)
_solvent_candidates = _solvent_candidates.set_index(SOLVENT_INDEX)
solvent_ds = DataSet.from_df(_solvent_candidates, metadata_columns=SOLVENT_METADATA_VARIABLES)

#Load UCB Pharma approved list
ucb_list = pd.read_csv(UCB_PHARMA_APPROVED_LIST)
ucb_list = ucb_list.set_index('cas_number')
ucb_ds = DataSet.from_df(ucb_list, metadata_columns=['solvent_class', 'solvent_name'])

