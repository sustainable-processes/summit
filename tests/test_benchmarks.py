import pytest
from summit.benchmarks import SnarBenchmark
from summit.utils.dataset import DataSet
import numpy as np

def test_snar_benchmark():
    """Test the SnAr benchmark"""
    b = SnarBenchmark()
    columns = [v.name for v in b.domain.variables]
    values  =   {('temperature', 'DATA'): 60.0,  #deg C,
                 ('q_dfnb', 'DATA'): 2.0,        #ml/min
                 ('q_pldn', 'DATA'): 2.0,        #ml/min
                 ('q_eth', 'DATA'): 0.5,         #ml/min
                  }

    # Check that results are reasonable
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)
    assert float(results['q_dfnb']) == 2.0
    assert float(results['q_pldn']) == 2.0
    assert float(results['q_eth']) == 0.5
    assert float(results['temperature']) == 60.0
    assert np.isclose(float(results['sty']), 4446.812424)
    assert np.isclose(float(results['e_factor']), 2.179134)

    # Check  total flowrate >10 raises error
    conditions['q_dfnb'].iloc[-1] = 12.0
    with pytest.raises(ValueError):
        b.run_experiment(conditions)

    return results
