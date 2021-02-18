import pytest
from summit.benchmarks import *
from summit.utils.dataset import DataSet
import numpy as np
import os
import pathlib
import shutil
import pkg_resources

DATA_PATH = pathlib.Path(pkg_resources.resource_filename("summit", "benchmarks/data"))


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


def test_train_experimental_emulator():
    model_name = f"reizman_suzuki_case_1"
    domain = ReizmanSuzukiEmulator.setup_domain()
    ds = DataSet.read_csv(DATA_PATH / f"{model_name}.csv")
    exp = ExperimentalEmulator(model_name, domain, dataset=ds, regressor=ANNRegressor)

    # Test grid search cross validation and training
    params = {
        "regressor__net__max_epochs": [1, 1000],
    }
    res = exp.train(cv_folds=5, random_state=100, search_params=params, verbose=0)
    r2 = res["test_r2"].mean()
    assert r2 > 0.8

    # Test plotting
    fig, ax = exp.parity_plot(output_variables="yield", include_test=True)

    # Test saving/loading
    exp.save("test_ee")
    exp_2 = ExperimentalEmulator.load(model_name, "test_ee")
    shutil.rmtree("test_ee")


def test_reizman_emulator():
    pass


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
