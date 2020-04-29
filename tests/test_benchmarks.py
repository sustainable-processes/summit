import pytest
from summit.benchmarks import SnarBenchmark
from summit.utils.dataset import DataSet
import numpy as np

def test_snar_benchmark():
    """Test the SnAr benchmark"""
    b = SnarBenchmark()
    columns = [v.name for v in b.domain.variables]
    values  =   {('tau', 'DATA'): 0.5,  # minutes
                ('equiv_pldn', 'DATA'): 1.5,  
                ('conc_dfnb', 'DATA'): 0.1, #molar
                ('temperature', 'DATA'): 30.0, # degrees celsius
                }

    # Check that results are reasonable
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)
    assert float(results['tau']) == 0.5
    assert float(results['equiv_pldn']) == 1.5
    assert float(results['conc_dfnb']) == 0.1
    assert float(results['temperature']) == 30.0
    assert np.isclose(float(results['sty']), 1859.50836)
    assert np.isclose(float(results['e_factor']), 1.771402)

    return results
