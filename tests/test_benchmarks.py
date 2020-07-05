import pytest
from summit.benchmarks import SnarBenchmark, Hartmann3D, Himmelblau, ThreeHumpCamel, ReizmanSuzukiEmulator
from summit.utils.dataset import DataSet
import numpy as np


def test_snar_benchmark():
    """Test the SnAr benchmark"""
    b = SnarBenchmark()
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
    assert np.isclose(float(results["sty"]), 168.958672)
    assert np.isclose(float(results["e_factor"]), 191.260294)

    return results


def test_reizman_suzuki_emulator():
    """ Test the Reizman Suzuki emulator"""
    b = ReizmanSuzukiEmulator(case=1)
    columns = [v.name for v in b.domain.variables]
    values = {
        ("catalyst", "DATA"): "P1-L2",
        ("t_res", "DATA"): 60,
        ("temperature", "DATA"): 110,
        ("catalyst_loading", "DATA"): 0.508,
    }
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)

    assert str(results["catalyst", "DATA"].iloc[0]) == values["catalyst", "DATA"]
    assert float(results["t_res"]) == values["t_res", "DATA"]
    assert float(results["temperature"]) == values["temperature", "DATA"]
    assert float(results["catalyst_loading"]) == values["catalyst_loading", "DATA"]
    assert np.isclose(float(results["ton"]), 16.513082)
    assert np.isclose(float(results["yield"]), 1.643731)

    return results