from summit.strategies import Strategy
from summit.experiment import Experiment, ExternalExperiment

import os
import json

class Runner:
    """"  Run an open- or closed-loop strategy and experiment cycle
    
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
                 experiment: Experiment=None, 
                 max_iterations=None, batch_size=1):
        self.strategy = strategy
        if experiment is not None:
            self.experiment = experiment
            self.max_iterations = max_iterations
            self.call_experiment = True
        if experiment is None:
            self.experiment = ExternalExperiment(self.strategy.transform.domain)
            self.max_iterations = 1
            self.call_experiment = False
        self.batch_size = batch_size

    def run(self, **kwargs):
        """  brief description 
        
        Parameters
        ---------- 
        new_data: summit.utils.data.DataSet, optional
            Dataset with data from most recent experiments. 
        new_extras: dict or list of dict
            Extras to include in the experiment.
        batch_size: int, optional
            The number experiments to request at each call of strategy.suggest_experiments.
            This overrides the batch_size in the initial call to Runner.
        """
        new_data = kwargs.get('new_data')
        new_extras = kwargs.get('new_extras')
        if new_data is not None or new_extras is not None:
            self.experiment.add_data(new_data, extras=new_extras)
            prev_res = self.experiment.data
        else:
            prev_res = None
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        i=0
        while True:
            next_experiments = self.strategy.suggest_experiments(num_experiments=batch_size,
                                                                 prev_res=prev_res)
            if self.call_experiment:                                        
                self.experiment.run_experiments(next_experiments)
            prev_res = self.experiment.data
            i += 1
            if self.max_iterations is not None:
                if i >= self.max_iterations:
                    break
        return next_experiments

    def to_dict(self,):
        runner_params = dict(max_iterations=self.max_iterations, 
                             batch_size=self.batch_size)

        return dict(runner=runner_params,
                    strategy=self.strategy.to_dict(),
                    experiment=self.experiment.to_dict())

    def save(self, filename):
        json.dump(filename, self.to_dict())