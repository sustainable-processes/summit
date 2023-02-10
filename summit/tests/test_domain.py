from summit.domain import *
from summit.utils.dataset import DataSet
import pandas as pd
import pytest


def test_continuous_variable():
    # Test input continuous variables
    bounds = [1, 100]
    var = ContinuousVariable(
        name="temperature", description="reaction temperature", bounds=bounds
    )
    assert isinstance(var, Variable)
    assert var.lower_bound == 1
    assert var.upper_bound == 100
    assert bounds in var.bounds
    assert var.is_objective == False

    # Test objective continuous variables
    var = ContinuousVariable(
        name="temperature",
        description="reaction temperature",
        bounds=[1, 100],
        is_objective=True,
        maximize=False,
    )
    assert var.maximize == False
    assert var.is_objective == True

    # Test serialization
    var = ContinuousVariable(
        name="temperature",
        description="reaction temperature",
        bounds=[1, 100],
        is_objective=True,
        maximize=False,
    )
    ser = var.to_dict()
    new_var = ContinuousVariable.from_dict(ser)
    assert isinstance(new_var, Variable)
    assert var.name == new_var.name
    assert var.description == new_var.description
    assert var.lower_bound == new_var.lower_bound
    assert var.upper_bound == new_var.upper_bound
    assert var.bounds.all() == new_var.bounds.all()
    assert var.is_objective == new_var.is_objective
    assert var.maximize == new_var.maximize


def test_categorical_variable():
    levels = ["benzene", "toluene", 1]
    var = CategoricalVariable(
        name="reactant", description="aromatic reactant", levels=levels
    )
    assert isinstance(var, Variable)
    assert var.is_objective == False
    assert var.name == "reactant"
    assert var.description == "aromatic reactant"
    assert levels == var.levels
    assert len(levels) == len(var.levels)
    assert len(levels) == var.num_levels

    # Make sure exception raised on non unique levels
    levels = ["benzene", "benzene"]
    with pytest.raises(ValueError):
        var = CategoricalVariable(name="nu", description="not_unique", levels=levels)

    # Make sure exception raise if a list is not passed
    levels = ("benzene", "toluene")
    with pytest.raises(TypeError):
        var = CategoricalVariable(name="nl", description="not_list", levels=levels)

    # Test serialization
    ser = var.to_dict()
    new_var = CategoricalVariable.from_dict(ser)
    assert isinstance(new_var, Variable)
    assert var.name == new_var.name
    assert var.description == new_var.description
    assert var.levels == new_var.levels

    # Test adding and removing a level
    var.add_level("new_level")
    assert "new_level" in var.levels
    with pytest.raises(ValueError):
        var.add_level("new_level")
    var.remove_level("new_level")
    assert "new_level" not in var.levels
    with pytest.raises(ValueError):
        var.remove_level("does_not_exist")

    # I should probably mock the dataset but I'm lazy
    solvent_ds = DataSet(
        [[5, 81], [-93, 111]],
        index=["benzene", "toluene"],
        columns=["melting_point", "boiling_point"],
    )
    var = CategoricalVariable("solvent", "solvent descriptors", descriptors=solvent_ds)
    assert isinstance(var, Variable)
    assert var.name == "solvent"
    assert var.description == "solvent descriptors"
    assert all(var.ds) == all(solvent_ds)
    assert var.num_levels == 2

    # Test serialization
    ser = var.to_dict()
    new_var = CategoricalVariable.from_dict(ser)
    assert var.name == new_var.name
    assert var.description == new_var.description
    assert all(var.ds) == all(new_var.ds)


def test_constraint():
    for i in ["<", "<=", "==", ">", ">="]:
        c = Constraint("x+y", constraint_type=i)

    with pytest.raises(ValueError):
        c = Constraint("x+y", "*=")


def test_domain():
    var1 = ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[30, 100],
    )
    var2 = ContinuousVariable(
        name="flowrate_a", description="flowrate of reactant a", bounds=[1, 100]
    )
    var3 = ContinuousVariable(
        name="flowrate_b", description="flowrate of reactant b", bounds=[1, 100]
    )
    var4 = CategoricalVariable(
        name="base",
        description="base additive",
        levels=["potassium_hydroxide", "sodium_hydroxide"],
    )
    solvent_ds = DataSet(
        [[5, 81], [-93, 111]],
        index=["benzene", "toluene"],
        columns=["melting_point", "boiling_point"],
    )
    var5 = CategoricalVariable("solvent", "solvent descriptors", descriptors=solvent_ds)
    var6 = ContinuousVariable(
        name="yld",
        description="yield of reaction",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    c = Constraint("flowrate_a+flowrate_b-100", "<=")

    all_combos = DataSet(
        [
            ["potassium_hydroxide", "benzene"],
            ["potassium_hydroxide", "toluene"],
            ["sodium_hydroxide", "benzene"],
            ["sodium_hydroxide", "toluene"],
        ],
        columns=["base", "solvent"],
    )

    domain = Domain(variables=[var1, var2, var3, var4, var5, var6], constraints=[c])
    input_variables = [var1, var2, var3, var4, var5]
    assert domain.input_variables == input_variables
    assert domain.output_variables == [var6]
    assert domain.num_variables() == 5  # not including outputs
    assert domain.num_variables(include_outputs=True) == 6
    assert (
        domain.num_continuous_dimensions() == 3
    )  # Not including outputs or descriptors
    assert domain.num_continuous_dimensions(include_outputs=True) == 4
    assert domain.num_continuous_dimensions(include_descriptors=True) == 5
    assert (
        domain.num_continuous_dimensions(include_outputs=True, include_descriptors=True)
        == 6
    )
    # Check that all combinations is correct()
    domain_all_combos = domain.get_categorical_combinations()
    assert pd.merge(all_combos, domain_all_combos, how="inner").equals(all_combos)
    assert domain.constraints == [c]

    # Test the adding feature
    domain = Domain()
    for var in input_variables:
        domain += var
    domain += var6
    domain += c
    assert domain.input_variables == input_variables
    assert domain.output_variables == [var6]
    assert domain.num_variables() == 5  # not including outputs
    assert domain.num_variables(include_outputs=True) == 6
    assert (
        domain.num_continuous_dimensions() == 3
    )  # Not including outputs or descriptors
    assert domain.num_continuous_dimensions(include_outputs=True) == 4
    assert domain.num_continuous_dimensions(include_descriptors=True) == 5
    assert (
        domain.num_continuous_dimensions(include_outputs=True, include_descriptors=True)
        == 6
    )
    assert domain.constraints == [c]

    # Test sending wrong type
    with pytest.raises(TypeError):
        domain = Domain(variables=["temperature", "flowrate_a"])
    with pytest.raises(TypeError):
        domain = Domain(constraints=["temperature", "flowrate_a"])

    # Test adding two variables with the same names
    with pytest.raises(ValueError):
        domain = Domain(variables=[var1, var1])

    # Test naming a variable a python keyword
    with pytest.raises(ValueError):
        ContinuousVariable(
            name="yield",
            description="yield of reaction",
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )

    # Test serialization
    domain = Domain(variables=[var1, var2, var3, var4, var5, var6], constraints=[c])
    ser = domain.to_dict()
    new_domain = Domain.from_dict(ser)
    assert domain.num_variables() == new_domain.num_variables()
    assert domain.num_variables(include_outputs=True) == new_domain.num_variables(
        include_outputs=True
    )
    assert domain.num_continuous_dimensions() == new_domain.num_continuous_dimensions()
    assert domain.num_continuous_dimensions(
        include_outputs=True
    ) == new_domain.num_continuous_dimensions(include_outputs=True)
