import pytest

from slurm_runner import SlurmRunner
from summit import *
from summit.benchmarks import BaumgartnerCrossCouplingEmulator
from summit.strategies import *

import warnings
import logging
import os
logging.basicConfig(level=logging.INFO)
token = os.environ.get('NEPTUNE_API_TOKEN')
if token is None:
    raise ValueError("Neptune_API_TOKEN needs to be an environmental variable")

NUM_REPEATS=20
MAX_EXPERIMENTS=100
NEPTUNE_PROJECT="sustainable-processes/summit"

# Cross Coupling Benchmark by Emulator trained on Baumgartner et al. (2019) data
experiment = BaumgartnerCrossCouplingEmulator()

# Run experiments
@pytest.mark.parametrize('strategy', [Random, NelderMead, SNOBFIT, SOBO, GRYFFIN])
@pytest.mark.parametrize('batch_size', [1,24])
def test_cn_experiment(strategy, batch_size, transform_descriptors=False):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain, transform_descriptors=transform_descriptors)

        # Early stopping for local optimization strategies
        if strategy in [NelderMead]:
            f_tol = 1e-5
        else:
            f_tol = None

        r = NeptuneRunner(experiment=experiment, strategy=s, 
                          neptune_project=NEPTUNE_PROJECT,
                          neptune_experiment_name=f"cn_experiment_SO_{s.__class__.__name__}_with_descriptors_repeat_{i}",
                          neptune_tags=["cn_experiment_single_objective", s.__class__.__name__,transform.__class__.__name__],
                          files=["slrum_summit_cn_experiment.sh"],
                          max_iterations=MAX_EXPERIMENTS//batch_size,
                          batch_size=batch_size,
                          f_tol=f_tol)
        r.run(save_at_end=True)
