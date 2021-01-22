import pytest
from summit.strategies import Strategy
from summit.experiment import Experiment
from summit.domain import *
from summit.benchmarks import (
    SnarBenchmark,
    DTLZ2,
    ExperimentalEmulator,
    ANNRegressor,
    # Hartmann3D,
    # Himmelblau,
    # ThreeHumpCamel,
    # ReizmanSuzukiEmulator,
    # BaumgartnerCrossCouplingEmulator,
)
from summit.utils.dataset import DataSet
import numpy as np
from numpy.random import default_rng
import os


@pytest.mark.parametrize("noise_level", [0.0, 2.5])
def test_snar_benchmark(noise_level):
    """Test the SnAr benchmark"""
    b = SnarBenchmark(noise_level=noise_level)
    columns = [v.name for v in b.domain.variables]
    values = {
        ("tau", "DATA"): 1.5,  # minutes
        ("equiv_pldn", "DATA"): 0.5,
        ("conc_dfnb", "DATA"): 0.1,  # molar
        ("temperature", "DATA"): 30.0,  # degrees celsius
    }

    # Check that results are reasonable
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)
    assert float(results["tau"]) == values[("tau", "DATA")]
    assert float(results["equiv_pldn"]) == values[("equiv_pldn", "DATA")]
    assert float(results["conc_dfnb"]) == values[("conc_dfnb", "DATA")]
    assert float(results["temperature"]) == values[("temperature", "DATA")]
    if noise_level == 0.0:
        assert np.isclose(results["sty"].values[0], 168.958672)
        assert np.isclose(results["e_factor"].values[0], 191.260294)

    # Test serialization
    d = b.to_dict()
    new_b = SnarBenchmark.from_dict(d)
    assert b.noise_level == noise_level

    return results


def create_domain():
    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[30, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flowrate of reactant a", bounds=[1, 100]
    )

    domain += ContinuousVariable(
        name="flowrate_b", description="flowrate of reactant b", bounds=[1, 100]
    )

    domain += ContinuousVariable(
        name="yield",
        description="yield of reaction",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    return domain


def create_dataset(domain, n_samples=100, random_seed=100):
    rng = default_rng(random_seed)
    n_features = len(domain.input_variables)
    inputs = rng.standard_normal(size=(n_samples, n_features))
    inputs *= [-5, 6, 0.1]
    output = np.sum(inputs ** 2, axis=1)
    data = np.append(inputs, np.atleast_2d(output).T, axis=1)
    columns = [v.name for v in domain.input_variables] + [
        domain.output_variables[0].name
    ]
    return DataSet(data, columns=columns)


def test_experimental_emulator(n_samples=100, max_epochs=100):
    # Setup
    domain = create_domain()
    ds = create_dataset(domain, n_samples=n_samples)

    # Train emulator
    exp = ExperimentalEmulator("test_model", domain, dataset=ds, regressor=ANNRegressor)
    exp.train(max_epochs=max_epochs)

    # Evaluate emulator
    print(exp.test())


def test_baumgartner_CC_emulator():
    """ Test the Baumgartner Cross Coupling emulator"""
    b = BaumgartnerCrossCouplingEmulator()
    columns = [v.name for v in b.domain.variables]
    values = {
        ("catalyst", "DATA"): "tBuXPhos",
        ("base", "DATA"): "DBU",
        ("t_res", "DATA"): 328.717801570892,
        ("temperature", "DATA"): 30,
        ("base_equivalents", "DATA"): 2.18301549894049,
        ("yield", "DATA"): 0.19,
    }
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)

    assert str(results["catalyst", "DATA"].iloc[0]) == values["catalyst", "DATA"]
    assert str(results["base", "DATA"].iloc[0]) == values["base", "DATA"]
    assert float(results["t_res"]) == values["t_res", "DATA"]
    assert float(results["temperature"]) == values["temperature", "DATA"]
    assert np.isclose(float(results["yld"]), 0.173581)

    # Test serialization
    d = b.to_dict()
    exp = BaumgartnerCrossCouplingEmulator.from_dict(d)

    return results


@pytest.mark.parametrize("num_inputs", [6])
def test_dltz2_benchmark(num_inputs):
    """Test the DTLZ2 benchmark"""
    b = DTLZ2(num_inputs=num_inputs, num_objectives=2)
    values = {(f"x_{i}", "DATA"): [0.5] for i in range(num_inputs)}
    ds = DataSet(values)
    b.run_experiments(ds)
    data = b.data
    assert np.isclose(data["y_0"].iloc[0], 0.7071)
    assert np.isclose(data["y_1"].iloc[0], 0.7071)
