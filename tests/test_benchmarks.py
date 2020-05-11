import pytest
from summit.benchmarks import SnarBenchmark, DTLZ2
from summit.utils.dataset import DataSet
import numpy as np

def test_snar_benchmark():
    """Test the SnAr benchmark"""
    b = SnarBenchmark()
    columns = [v.name for v in b.domain.variables]
    values  =   {('tau', 'DATA'): 1.5,  # minutes
                ('equiv_pldn', 'DATA'): 0.5,  
                ('conc_dfnb', 'DATA'): 0.1, #molar
                ('temperature', 'DATA'): 30.0, # degrees celsius
                }

    # Check that results are reasonable
    conditions = DataSet([values], columns=columns)
    results = b.run_experiments(conditions)
    assert float(results['tau']) == values[('tau', 'DATA')]
    assert float(results['equiv_pldn']) == values[('equiv_pldn', 'DATA')]
    assert float(results['conc_dfnb']) == values[('conc_dfnb', 'DATA')]
    assert float(results['temperature']) == values[('temperature', 'DATA')]
    assert np.isclose(float(results['sty']), 168.958672)
    assert np.isclose(float(results['e_factor']), 191.260294)

    return results

def test_dltz_benchmark():
    """Test the DTLZ2 benchmark"""
    b = DTLZ2()

    b 