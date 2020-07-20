#!/usr/bin/python

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

NUM_REPEATS=1
NEPTUNE_PROJECT="sustainable-processes/summit"
MAX_EXPERIMENTS=50
BATCH_SIZE=1

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
warnings.filterwarnings('ignore', category=RuntimeWarning)
for transform in transforms:
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = GRYFFIN(experiment.domain, transform=transform)

        exp_name=f"snar_experiment_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}"
        r = NeptuneRunner(experiment=experiment, strategy=s, 
                        neptune_project=NEPTUNE_PROJECT,
                        neptune_experiment_name=exp_name,
                        tags=["snar_experiment", s.__class__.__name__, transform.__class__.__name__],
                        files=["snar_experiment_gryffin.py"],
                        max_iterations=MAX_EXPERIMENTS//BATCH_SIZE,
                        batch_size=BATCH_SIZE,
                        num_initial_experiments=1,
                        hypervolume_ref=[-2957,10.7])
        r.run(save_at_end=True)
