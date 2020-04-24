import pytest
from summit.benchmarks import SnarBenchmark
from summit.utils.dataset import DataSet
import numpy as np

def test_snar_benchmark():
    """Test the SnAr benchmark"""
    b = SnarBenchmark()
    columns = [v.name for v in b.domain.variables]
    values = [v.bounds[0]+ 0.1*(v.bounds[1]-v.bounds[0])
              for v in b.domain.variables]
    values = np.array(values)
    values = np.atleast_2d(values)

    # Check that results are reasonable
    conditions = DataSet(values, columns=columns)
    results = b.run_experiment(conditions)
    assert float(results['q_dfnb']) == 1.0
    assert float(results['q_pldn']) == 1.0
    assert float(results['q_eth']) == 1.0
    assert float(results['temperature']) == 39.0
    assert np.isclose(float(results['sty']), 0.173095)
    assert np.isclose(float(results['e_factor']), 1.018594)


    # Check  total flowrate >10 raises error
    conditions['q_dfnb'].iloc[-1] = 12.0
    with pytest.raises(ValueError):
        b.run_experiment(conditions)

    return results
