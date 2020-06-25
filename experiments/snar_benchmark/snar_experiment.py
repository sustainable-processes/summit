import pytest

from summit import *
from summit.benchmarks import SnarBenchmark
from summit.strategies import *

import logging
logging.basicConfig(level=logging.INFO)

NUM_REPEATS=20
MAX_ITERATIONS=100
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
transforms = [MultitoSingleObjective(experiment.domain, 
                                     expression='-sty/1e4+e_factor/100', 
                                     maximize=False),
              Chimera(experiment.domain, hierarchies[0]),
              Chimera(experiment.domain, hierarchies[1]),
              Chimera(experiment.domain, hierarchies[2]),
              Chimera(experiment.domain, hierarchies[3]),
]

# Run experiments
@pytest.mark.parametrize('strategy', [NelderMead, SNOBFIT, Random, SOBO])
@pytest.mark.parametrize('transform', transforms)
@pytest.mark.parametrize('batch_size', [1,5,10,20])
def test_snar_experiment(strategy, transform, batch_size, num_repeats=20):
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain, transform=transform)

        # Early stopping for local optimization strategies
        if strategy in [NelderMead, SNOBFIT]:
            f_tol = 1e-5
        else:
            f_tol = None

        r = NeptuneRunner(experiment=experiment, strategy=s, 
                          neptune_project=NEPTUNE_PROJECT,
                          neptune_experiment_name=f"snar_experiment_{s.__class__.__name__}_repeat_{i}",
                          files=["snar_experiment.py"],
                          max_iterations=MAX_ITERATIONS,
                          batch_size=batch_size,
                          f_tol=f_tol)
        r.run(save_at_end=True)