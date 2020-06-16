from summit.strategies import Strategy
from summit.experiment import Experiment
from fastprogress.fastprogress import progress_bar

import os
import json

class Runner:
    """"  Run a closed-loop strategy and experiment cycle
    
    Parameters
    ---------- 
    strategy: `summit.strategies.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment: `summit.experiment.Experiment`, optional
        The experiment class to use for running experiments. If None,
        the ExternalExperiment class will be used, which assumes that
        data from each experimental run will be added as a keyword
        argument to the `run` method.
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is None.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments.

    Examples
    --------    
    
    """
    def __init__(self, strategy: Strategy, 
                 experiment: Experiment, 
                 max_iterations=100, batch_size=1):
        self.strategy = strategy
        self.experiment = experiment
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def run(self, **kwargs):
        """  Run the closed loop experiment cycle
        """
        prev_res = None
        i=0
        for i in progress_bar(range(self.max_iterations)):
            next_experiments = self.strategy.suggest_experiments(num_experiments=self.batch_size,
                                                                prev_res=prev_res)                                      
            prev_res = self.experiment.run_experiments(next_experiments)


    def to_dict(self,):
        runner_params = dict(max_iterations=self.max_iterations, 
                             batch_size=self.batch_size)

        return dict(runner=runner_params,
                    strategy=self.strategy.to_dict(),
                    experiment=self.experiment.to_dict())

    def save(self, filename):
        json.dump(filename, self.to_dict())