"""Command line entry point """
import os
import json
class Runner:
    def __init__(self, strategy, experiment, 
                 max_iterations=100, batch_size=1):
        self.experiment = experiment
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def run(self):
        prev_res=None
        for i in range(self.max_iterations):
            next_experiments = self.strategy.suggest_experiments(num_experiments=self.batch_size,
                                                                 prev_res=prev_res)
            self.experiment.run_experiments(next_experiments)
            prev_res = self.experiment.data

    def to_dict(self,):
        runner_params = dict(max_iterations=self.max_iterations, 
                             batch_size=self.batch_size)

        return dict(runner=runner_params,
                    strategy=self.strategy.to_dict(),
                    experiment=self.experiment.to_dict())

    def save(self, filename):
        json.dump(filename, self.to_dict())