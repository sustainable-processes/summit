from summit import *
from summit.benchmarks import SnarBenchmark
from summit.strategies import *

import warnings
import logging
import os
logging.basicConfig(level=logging.ERROR)
token = os.environ.get('NEPTUNE_API_TOKEN')
if token is None:
    raise ValueError("Neptune_API_TOKEN needs to be an environmental variable")

# Variables
NUM_REPEATS=20
NEPTUNE_PROJECT="sustainable-processes/summit"
MAX_EXPERIMENTS=50
BATCH_SIZE=1

#SnAr benchmark with 2.5% experimental measurement noise
experiment = SnarBenchmark(noise_level_percent=2.5)

# Run experiments
warnings.filterwarnings('ignore', category=RuntimeWarning)
for i in range(NUM_REPEATS):
    experiment.reset()
    s = TSEMO(experiment.domain)

    exp_name = f"snar_experiment_{s.__class__.__name__}_repeat_{i}"
    r = NeptuneRunner(experiment=experiment, strategy=s, 
                      neptune_project=NEPTUNE_PROJECT,
                      tags=["snar_experiment", s.__class__.__name__],
                      neptune_experiment_name=exp_name,
                      files=["snar_experiment_tsemo.py"],
                      max_iterations=MAX_EXPERIMENTS//BATCH_SIZE,
                      batch_size=BATCH_SIZE,
                      num_initial_experiments=1,
                      hypervolume_ref=[-2957,10.7])
    r.run(save_at_end=True)

