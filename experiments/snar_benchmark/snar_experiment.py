from summit import *
from summit.benchmarks import SnarBenchmark
from summit.strategies import *

import logging
logging.basicConfig(level=logging.INFO)

NUM_REPEATS=20
MAX_ITERATIONS=100
NEPTUNE_PROJECT="sustainable-processes/summit"

strategies = [NelderMead, SNOBFIT, Random, SOBO]
experiment = SnarBenchmark()

for strategy in strategies:
    for i in range(NUM_REPEATS):
        experiment.reset()
        transform = MultitoSingleObjective(experiment.domain, 
                                           expression='-sty/1e4+e_factor/100', 
                                           maximize=False)
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
                          batch_size=1,
                          f_tol=f_tol)
        r.run()