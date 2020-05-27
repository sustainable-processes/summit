from summit.strategies import TSEMO
from summit.benchmarks import DTLZ2
import matplotlib.pyplot as plt
import numpy as np


def dtlz2_test():
    #Run the DTLZ2 benchmark
    num_inputs=6
    num_objectives=2
    lab = DTLZ2(num_inputs=num_inputs, 
            num_objectives=num_objectives)
    strategy = TSEMO(lab.domain, random_rate=0.00)
    experiments = strategy.suggest_experiments(5*num_inputs)

    for i in range(100):
        # Run experiments
        lab.run_experiments(experiments)
        
        # Get suggestions
        nsga_options = dict(pop_size=100, iterations=100)
        experiments = strategy.suggest_experiments(1, lab.data,
                                                **nsga_options)

    return lab


if __name__ == '__main__':
    for i in range(20):
        lab = dtlz2_test()
        lab.save(f'data/python/experiment_{i}.csv')