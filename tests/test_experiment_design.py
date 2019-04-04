import pytest
import numpy as np
import pandas as pd
from summit.experiment_design import Design, RandomDesign, LatinDesign
from summit.domain import ContinuousVariable, DiscreteVariable, DescriptorsVariable
    
NUM_VARIABLES = 3

@pytest.fixture
def patch_variable_simple(mocker):
    names = [f"var_{i}" for i in range(NUM_VARIABLES)]
    MockVariable = mocker.MagicMock()
    type(MockVariable).name = mocker.PropertyMock(side_effect=names)
    return MockVariable

@pytest.fixture
def patch_domain_simple(mocker, patch_variable_simple):
    '''Mock the Variable and Domain classes'''
    MockDomain = mocker.MagicMock()
    variables = [patch_variable_simple for i in range(NUM_VARIABLES)]
    type(MockDomain).variables = mocker.PropertyMock(return_value=variables)
    type(MockDomain).num_variables = mocker.PropertyMock(return_value=NUM_VARIABLES)
    return MockDomain

def test_design(patch_domain_simple):
    #Test that the design instantiates correctly
    d = Design(patch_domain_simple , 3, 'test')

    #Test that 1d designs work properly
    indices =  np.array([[0, 2, 4]]).T
    values = np.array([[1, 2, 3]]).T
    d.add_variable('var_0', values, indices=indices)
    assert d.get_indices('var_0').all() == indices.all()
    assert d.get_values('var_0').all() == values.all()
        
    #Test that nd designs work properly (for bandit variables)
    indices =  np.array([[0, 2, 4]]).T
    values = np.array([[1,2],[2, 3],[3,4]])
    d.add_variable('var_1', values, indices=indices)
    assert d.get_indices('var_1').all() == indices.all()
    assert d.get_values('var_1').all() == values.all()

    #Test that designs without indices work
    values = np.array([[1, 2, 3]]).T
    d.add_variable('var_2', values)

    #Test getting back the whole design
    indices = np.array([[0, 2, 4, 0], [0, 2, 4, 0]]).T
    values = np.array([[1,1,2, 1],[2, 2, 3, 2],[3, 3,4, 3]])

    assert d.get_indices().all() == indices.all()
    assert d.get_values().all() == values.all()

@pytest.fixture
def patch_continuous_variable(mocker):
    MockVariable = mocker.MagicMock()
    type(MockVariable).variable_type =mocker.PropertyMock(return_value='continuous') 
    type(MockVariable).name = mocker.PropertyMock(return_value='var_continuous')
    type(MockVariable).lower_bound = mocker.PropertyMock(return_value=0)
    type(MockVariable).upper_bound = mocker.PropertyMock(return_value=10)
    return MockVariable

chemicals = np.array([f'chemical_{i}' for i in range(20)])
chemicals  = np.atleast_2d(chemicals).T

@pytest.fixture
def patch_discrete_variable(mocker):
    MockVariable = mocker.MagicMock()
    type(MockVariable).variable_type = mocker.PropertyMock(return_value='discrete')
    type(MockVariable).name = mocker.PropertyMock(return_value='var_discrete')
    type(MockVariable).levels = mocker.PropertyMock(return_value=chemicals)
    type(MockVariable).num_levels = mocker.PropertyMock(return_value=20)
    return MockVariable


solvent_df = pd.DataFrame(200*np.random.rand(100, 2), 
                          index=[f"solvent_{i}" for i in range(100)],
                          columns=['melting_point', 'boiling_point'])

@pytest.fixture
def patch_descriptors_variable(mocker):
    MockVariable = mocker.MagicMock()
    type(MockVariable).variable_type = mocker.PropertyMock(return_value='descriptors') 
    type(MockVariable).name = mocker.PropertyMock(return_value='var_descriptors')
    type(MockVariable).df = mocker.PropertyMock(return_value=solvent_df)
    type(MockVariable).select_subset = mocker.PropertyMock(return_value=None)
    type(MockVariable).num_descriptors = mocker.PropertyMock(return_value=2)
    type(MockVariable).num_examples = mocker.PropertyMock(return_value=2)
    return MockVariable

@pytest.fixture
def patch_domain_full(mocker, patch_continuous_variable,
                      patch_discrete_variable,
                      patch_descriptors_variable):
    MockDomain = mocker.MagicMock()
    variables = [patch_continuous_variable,
                 patch_discrete_variable,
                 patch_descriptors_variable]
    type(MockDomain).variables = mocker.PropertyMock(return_value=variables)
    type(MockDomain).num_variables = mocker.PropertyMock(return_value=3)
    type(MockDomain).num_discrete_variables = mocker.PropertyMock(return_value=1)
    type(MockDomain).num_continuous_dimensions = mocker.PropertyMock(return_value=3)
    return MockDomain

@pytest.mark.parametrize("Designer", [RandomDesign, LatinDesign])
def test_designers(mocker, patch_domain_full, Designer):
    seed=100
    num_samples = 10
    random_state = np.random.RandomState(seed=seed)
    d = Designer(patch_domain_full, random_state=random_state)
    
    design = d.generate_experiments(num_samples)

    #Check that a Design object is returned
    assert isinstance(design, Design)
    
    #Check that all values are within the proper ranges for the variables
    c_values = design.get_values('var_continuous')
    assert c_values.min() > 0
    assert c_values.max() < 10

    d_values = design.get_values('var_discrete')
    assert all([value in chemicals for value in d_values])

    dc_values = design.get_values('var_descriptors')
    descriptor_values = solvent_df.values
    assert all([dc_value in descriptor_values for dc_value in dc_values])

    #Check that all indices are within the proper ranges for the variables
    c_indices = design.get_indices('var_continuous')
    assert c_indices.all() == 0
    
    d_indices = design.get_indices('var_discrete')
    assert d_indices.all() < len(chemicals)-1

    dc_indices = design.get_indices('var_descriptors')
    assert dc_indices.all() < solvent_df.shape[0] -1

    
    