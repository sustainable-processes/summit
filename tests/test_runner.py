import pytest
from summit import Runner, Strategy, Experiment
from summit.strategies import *
from summit.domain import *
from summit.utils.dataset import DataSet

import numpy as np
import os

def test_runner_unit():
    class MockStrategy(Strategy):
        iterations = 0
        def suggest_experiments(self, num_experiments=1, **kwargs):
            values = 0.5*np.ones([num_experiments,2])
            self.iterations += 1
            return DataSet(values, columns=['x_1', 'x_2']) 

    class MockExperiment(Experiment):
        def __init__(self):
            super().__init__(self.create_domain())

        def create_domain(self):
            domain = Domain()
            domain += ContinuousVariable('x_1',
                                         description='',
                                         bounds=[0,1])
            domain += ContinuousVariable('x_2',
                                         description='',
                                         bounds=[0,1])
            domain += ContinuousVariable('y_1',
                                         description='',
                                         bounds=[0,1],
                                         is_objective=True,
                                         maximize=True)
            return domain

        def _run(self, conditions, **kwargs):
            conditions[('y_1', 'DATA')] = 0.5
            return conditions, {}

    max_iterations = 10
    batch_size = 5
    exp = MockExperiment()
    r = Runner(strategy=MockStrategy(exp.domain),
               experiment=exp,
               max_iterations=max_iterations,
               batch_size=batch_size)
    r.run()

    # Check that correct number of iterations run
    assert r.strategy.iterations == max_iterations
    assert r.experiment.data.shape[0] == int(batch_size*max_iterations)
    
    # Check saving and loading functionality
    r.save('test_runner.json')
    r_2 = Runner.load('test_runner.json')


@pytest.mark.parametrize('strategy', [SNOBFIT, TSEMO2, NelderMead, Random, LHS])
def test_runner_integration(strategy):
    class MockExperiment(Experiment):
        def __init__(self):
            super().__init__(self.create_domain())

        def create_domain(self):
            domain = Domain()
            domain += ContinuousVariable('x_1',
                                         description='',
                                         bounds=[0,1])
            domain += ContinuousVariable('x_2',
                                         description='',
                                         bounds=[0,1])
            domain += ContinuousVariable('y_1',
                                         description='',
                                         bounds=[0,1],
                                         is_objective=True,
                                         maximize=True)
            return domain

        def _run(self, conditions, **kwargs):
            conditions[('y_1', 'DATA')] = 0.5
            return conditions, {}

    exp = MockExperiment()
    strategy = strategy(exp.domain)

    r = Runner(strategy=strategy,
               experiment=exp,
               max_iterations=1,
               batch_size=1)
    r.run()
    
