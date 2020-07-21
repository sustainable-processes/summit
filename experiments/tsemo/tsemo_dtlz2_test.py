from summit.strategies import TSEMO
from summit.benchmarks import DTLZ2
from summit.utils.models import GPyModel
from GPy.kern import Exponential
import numpy as np
import json
import warnings
from fastprogress.fastprogress import master_bar, progress_bar


#Two tests were run; one with 1500 and one with 4000
N_SPECTRAL_POINTS = 4000
DATE="20200618"
tsemo_options = dict(pop_size=100,                          #population size for NSGAII
                     iterations=100,                        #iterations for NSGAII
                     n_spectral_points=N_SPECTRAL_POINTS,   #number of spectral points for spectral sampling
                     num_restarts=200,                      #number of restarts for GP optimizer (LBSG)
                     parallel=True)                         #operate GP optimizer in parallel
description="Description: Used exponential kernel instead of matern and increase num restarts. Also, catch and save errors."

def dtlz2_test():
    #Run the DTLZ2 benchmark
    errors = 0
    num_inputs=6
    num_objectives=2
    lab = DTLZ2(num_inputs=num_inputs, 
                num_objectives=num_objectives)
    models = {f'y_{i}': GPyModel(Exponential(input_dim=num_inputs,ARD=True))
              for i in range(num_objectives)}

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tsemo = TSEMO(lab.domain, models=models, random_rate=0.00)
    experiments = tsemo.suggest_experiments(5*num_inputs)

    mb = master_bar(range(1))
    for j in mb:
        mb.main_bar.comment = f'Repeats'
        for i in progress_bar(range(100), parent=mb):
            mb.child.comment = f'Iteration'
            # Run experiments
            experiments = lab.run_experiments(experiments)
            
            # Get suggestions
            try:
                experiments = tsemo.suggest_experiments(1, experiments,
                                                        **tsemo_options)
            except Exception as e:
                print(e)
                errors +=1

        tsemo.save(f'new_tsemo_params_{j}.json')


if __name__ == '__main__':
    tsemo_options.update({'description': description})
    with open(f'new_params.json', 'w') as f:
        json.dump(tsemo_options,f)
    dtlz2_test()
