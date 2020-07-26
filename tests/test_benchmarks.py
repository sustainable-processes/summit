import pytest
from summit.strategies import Strategy
from summit.experiment import Experiment
from summit.benchmarks import SnarBenchmark,DTLZ2, Hartmann3D, Himmelblau, ThreeHumpCamel, ReizmanSuzukiEmulator, BaumgartnerCrossCouplingEmulator
from summit.utils.dataset import DataSet
import numpy as np
import os

def test_experiment():
    class MockStrategy(Strategy):
        def suggest_experiments(self, num_experiments, previous_results):
            inputs, outputs = self.transform.transform_inputs_outputs(previous_results)
            objectives = [v for v in self.domain.variables if v.is_objective]
            assert len(objectives) == 1
            assert objectives[0].name == "scalar_objective"
            assert outputs["scalar_objective"].iloc[0] == 70.0
            return self.transform.un_transform(inputs)


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

    d = b.to_dict()
    SnarBenchmark.from_dict(d)

    return results


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

    return results

@pytest.mark.parametrize('num_inputs', [6])
def test_dltz2_benchmark(num_inputs):
    """Test the DTLZ2 benchmark"""
    b = DTLZ2(num_inputs=num_inputs,
              num_objectives=2)
    values = {(f'x_{i}', 'DATA'): [0.5] for  i in range(num_inputs)}
    ds = DataSet(values)
    b.run_experiments(ds)
    data = b.data
    assert np.isclose(data['y_0'].iloc[0], 0.7071)
    assert np.isclose(data['y_1'].iloc[0], 0.7071)

