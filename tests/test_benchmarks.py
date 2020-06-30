import pytest
from summit.benchmarks import SnarBenchmark, Hartmann3D, Himmelblau, ThreeHumpCamel
from summit.utils.dataset import DataSet
import numpy as np
import matplotlib.pyplot as plt


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

def test_test_functions():
    b = ThreeHumpCamel()
    columns = [v.name for v in b.domain.variables]
    values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    values = np.array(values)
    values = np.atleast_2d(values)
    conditions = DataSet(values, columns=columns)
    results = b.run_experiments(conditions)
    fig, ax = b.plot()
    plt.show(fig)

test_test_functions()
