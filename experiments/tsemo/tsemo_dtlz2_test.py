from summit.strategies import TSEMO
from summit.benchmarks import DTLZ2
from summit.utils.models import GPyModel
from GPy.kern import Exponential
import matplotlib.pyplot as plt
import numpy as np
import json

#Two tests were run; one with 1500 and one with 4000
N_SPECTRAL_POINTS = 1500
DATE="20200608"
tsemo_options = dict(pop_size=100, iterations=100,
                     n_spectral_points=N_SPECTRAL_POINTS,
                     num_restarts=200, parallel=True)
description="Description: Used exponential kernel instead of matern and increase num restarts"

def dtlz2_test():
    #Run the DTLZ2 benchmark
    num_inputs=6
    num_objectives=2
    lab = DTLZ2(num_inputs=num_inputs, 
                num_objectives=num_objectives)
    models = {f'y_{i}': GPyModel(Exponential(input_dim=num_inputs,ARD=True))
              for i in range(num_objectives)}
    strategy = TSEMO(lab.domain, models=models, random_rate=0.00)
    experiments = strategy.suggest_experiments(5*num_inputs)

    for i in range(100):
        # Run experiments
        lab.run_experiments(experiments)
        
        # Get suggestions
        experiments = strategy.suggest_experiments(1, lab.data,
                                                   **tsemo_options)

    return lab, tsemo_options


if __name__ == '__main__':
    tsemo_options.update({'description': description})
    json.dump(f'data/python/{DATE}/params.json', tsemo_options)
    for i in range(20):
        lab, options = dtlz2_test()
        lab.save(f'data/python/{DATE}/experiment_{i}.csv')
