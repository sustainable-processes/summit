from pytest import fixture
import numpy as np
from summit.experiment_design import Design
    
NUM_VARIABLES = 5

@fixture
def patch_variable(mocker):
    names = [f"var_{i}" for i in range(NUM_VARIABLES)]
    MockVariable = mocker.MagicMock()
    type(MockVariable).name = mocker.PropertyMock(side_effect=names)
    return MockVariable

@fixture
def patch_domain(mocker, patch_variable):
    MockDomain = mocker.MagicMock()
    variables = [patch_variable for i in range(NUM_VARIABLES)]
    type(MockDomain).variables = mocker.PropertyMock(return_value=variables)
    type(MockDomain).num_variables = mocker.PropertyMock(return_value=NUM_VARIABLES)
    return MockDomain

def test_design(patch_domain):
    #Test that the design instantiates correctly
    d = Design(patch_domain, 3)

    indices =  np.array([0, 2, 4])
    values = np.array([1, 2, 3])
    d.add_variable('var_0', indices, values)

    assert all(d.get_indices('var_0') == indices)
    assert all(d.get_values('var_0') == values)
        