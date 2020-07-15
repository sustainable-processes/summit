import pytest

from summit import *
from summit.benchmarks import BaumgartnerCrossCouplingEmulator_Yield_Cost
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
# This emulator is modified to include costs as an output
experiment = BaumgartnerCrossCouplingEmulator_Yield_Cost()
print(experiment.domain.variables)

# Transforms from multi to single objective
hierarchies = [{'yld': {'hierarchy': 0, 'tolerance': 1},
                'cost': {'hierarchy': 1, 'tolerance': 1}},
               
               {'yld': {'hierarchy': 0, 'tolerance': 0.5},
                'cost': {'hierarchy': 1, 'tolerance': 0.5}},
               
               {'yld': {'hierarchy': 0, 'tolerance': 1.0},
                'cost': {'hierarchy': 1, 'tolerance': 0.5}},
               
               {'yld': {'hierarchy': 0, 'tolerance': 0.5},
                'cost': {'hierarchy': 1, 'tolerance': 1.0}}
              ]
transforms = [Chimera(experiment.domain, hierarchies[2]),
              MultitoSingleObjective(experiment.domain, 
                                     expression='-yld+(cost-1.001)/(2.999)',
                                     maximize=False),
              Chimera(experiment.domain, hierarchies[0]),
              Chimera(experiment.domain, hierarchies[1]),
              Chimera(experiment.domain, hierarchies[3]),
]

# Run experiments
@pytest.mark.parametrize('strategy', [Random, SNOBFIT, NelderMead, SOBO, GRYFFIN])
@pytest.mark.parametrize('transform', transforms)
@pytest.mark.parametrize('batch_size', [1,24])
def test_cn_experiment(strategy, transform, batch_size):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain, transform=transform, transform_descriptors=True)

        # Early stopping for local optimization strategies
        if strategy in [NelderMead]:
            f_tol = 1e-5
        else:
            f_tol = None

        r = NeptuneRunner(experiment=experiment, strategy=s, 
                          neptune_project=NEPTUNE_PROJECT,
                          neptune_experiment_name=f"cn_experiment_MO_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}",
                          files=["experimental_emulator"],
                          max_iterations=MAX_EXPERIMENTS//batch_size,
                          batch_size=batch_size,
                          f_tol=f_tol)
        r.run(save_at_end=True)
