import pytest

from summit import *
from summit.benchmarks import SnarBenchmark
from summit.strategies import *

import warnings
import logging
import os
logging.basicConfig(level=logging.INFO)
token = os.environ.get('NEPTUNE_API_TOKEN')
if token is None:
    raise ValueError("Neptune_API_TOKEN needs to be an environmental variable")
MAX_EXPERIMENTS=100
NEPTUNE_PROJECT="sustainable-processes/summit"

#SnAr benchmark with 2.5% experimental measurement noise
experiment = SnarBenchmark(noise_level_percent=2.5)

# Transforms from multi to single objective
hierarchies = [{'sty': {'hierarchy': 0, 'tolerance': 1}, 
                'e_factor': {'hierarchy': 1, 'tolerance': 1}},
               
               {'sty': {'hierarchy': 0, 'tolerance': 0.5}, 
                'e_factor': {'hierarchy': 1, 'tolerance': 0.5}},
               
               {'sty': {'hierarchy': 0, 'tolerance': 1.0}, 
                'e_factor': {'hierarchy': 1, 'tolerance': 0.5}},
               
               {'sty': {'hierarchy': 0, 'tolerance': 0.5}, 
                'e_factor': {'hierarchy': 1, 'tolerance': 1.0}}
              ]
transforms = [Chimera(experiment.domain, hierarchies[2]),
              MultitoSingleObjective(experiment.domain, 
                                     expression='-sty/1e4+e_factor/100', 
                                     maximize=False),
              Chimera(experiment.domain, hierarchies[0]),
              Chimera(experiment.domain, hierarchies[1]),
              Chimera(experiment.domain, hierarchies[3]),

]

# Run experiments
def test_snar_experiment(strategy, transform, batch_size, num_repeats=20):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for i in range(num_repeats):
        experiment.reset()
        s = strategy(experiment.domain, transform=transform)

        # Early stopping for local optimization strategies
        if strategy in [NelderMead]:
            f_tol = 1e-5
        else:
            f_tol = None

        r = NeptuneRunner(experiment=experiment, strategy=s, 
                          neptune_project=NEPTUNE_PROJECT,
                          neptune_experiment_name=f"snar_experiment_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}",
                          files=["snar_experiment.py"],
                          max_iterations=MAX_EXPERIMENTS//batch_size,
                          batch_size=batch_size,
                          f_tol=f_tol)
        r.run(save_at_end=True)

if __name__== "__main__":
    # Test Factorial DoE
    test_snar_experiment(strategy=FullFactorial, transform=None, batch_szie=1)

    # Test Gryffin
    for transform in transforms:
        test_snar_experiment(strategy=GRYFFIN, transform=transform, batch_size=1)
