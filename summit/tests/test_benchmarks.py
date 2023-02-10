import pytest
from summit.domain import *
from summit.benchmarks import *
from summit.utils.dataset import DataSet
import numpy as np
import pandas as pd
import os
import pathlib
import shutil
import pkg_resources
import matplotlib.pyplot as plt

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
    # params = {
    #     "regressor__net__max_epochs": [1, 1000],
    # }
    params = None
    exp.train(
        cv_folds=5, max_epochs=1000, random_state=100, search_params=params, verbose=0
    )

    # Testing
    res = exp.test()
    r2 = res["test_r2"].mean()
    assert r2 > 0.8

    # Test plotting
    fig, ax = exp.parity_plot(output_variables="yld", include_test=True)

    # Test saving/loading
    exp.save("test_ee")
    exp_2 = ExperimentalEmulator.load(model_name, "test_ee")
    assert all(exp.descriptors_features) == all(exp_2.descriptors_features)
    assert exp.n_examples == exp_2.n_examples
    assert all(exp.output_variable_names) == all(exp_2.output_variable_names)
    assert exp.clip == exp_2.clip
    exp_2.X_train, exp_2.y_train, exp_2.X_test, exp_2.y_test = (
        exp.X_train,
        exp.y_train,
        exp.X_test,
        exp.y_test,
    )
    res = exp_2.test(X_test=exp.X_test, y_test=exp.y_test)
    exp.parity_plot(output_variables="yld", include_test=True)
    r2 = res["test_r2"].mean()
    assert r2 > 0.8
    shutil.rmtree("test_ee")


def test_reizman_emulator(show_plots=False):
    b = get_pretrained_reizman_suzuki_emulator(case=1)
    b.parity_plot(include_test=True)
    if show_plots:
        plt.show()
    columns = [v.name for v in b.domain.variables]
    values = {
        "catalyst": ["P1-L3"],
        "t_res": [600],
        "temperature": [30],
        "catalyst_loading": [0.498],
    }
    conditions = pd.DataFrame(values)
    conditions = DataSet.from_df(conditions)
    results = b.run_experiments(conditions, return_std=True)

    for name, value in values.items():
        if type(value[0]) == str:
            assert str(results[name].iloc[0]) == value[0]
        else:
            assert float(results[name].iloc[0]) == value[0]
    assert np.isclose(float(results["yld"]), 0.6, atol=15)
    assert np.isclose(float(results["ton"]), 1.1, atol=15)

    # Test serialization
    d = b.to_dict()
    exp = ReizmanSuzukiEmulator.from_dict(d)
    return results


@pytest.mark.parametrize("use_descriptors", [True, False])
@pytest.mark.parametrize("include_cost", [True, False])
def test_baumgartner_CC_emulator(use_descriptors, include_cost, show_plots=False):
    """ Test the Baumgartner Cross Coupling emulator"""
    b = get_pretrained_baumgartner_cc_emulator(
        use_descriptors=use_descriptors, include_cost=include_cost
    )
    b.parity_plot(include_test=True)
    if show_plots:
        plt.show()
    columns = [v.name for v in b.domain.variables]
    values = {
        "catalyst": ["tBuXPhos"],
        "base": ["DBU"],
        "t_res": [328.7178016],
        "temperature": [30],
        "base_equivalents": [2.183015499],
    }
    conditions = pd.DataFrame(values)
    conditions = DataSet.from_df(conditions)
    results = b.run_experiments(conditions, return_std=True)

    assert str(results["catalyst"].iloc[0]) == values["catalyst"][0]
    assert str(results["base"].iloc[0]) == values["base"][0]
    assert float(results["t_res"].iloc[0]) == values["t_res"][0]
    assert float(results["temperature"].iloc[0]) == values["temperature"][0]
    assert np.isclose(results["yld"].iloc[0], 0.043, atol=0.2)

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


def test_no_objectives():
    domain = Domain()
    domain += ContinuousVariable("x", "", bounds=[0, 1])

    with pytest.raises(DomainError):
        ExperimentalEmulator("test", domain)