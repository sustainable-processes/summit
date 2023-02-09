import pytest
from summit import NeptuneRunner, Runner, Strategy, Experiment
from summit.strategies import *
from summit.benchmarks import *
from summit.domain import *
from summit.utils.dataset import DataSet

import numpy as np
import os


@pytest.mark.parametrize("max_iterations", [1, 10])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("max_same", [None, 5])
@pytest.mark.parametrize("max_restarts", [1, 5])
@pytest.mark.parametrize("runner", [Runner, NeptuneRunner])
def test_runner_unit(max_iterations, batch_size, max_same, max_restarts, runner):
    class MockStrategy(Strategy):
        iterations = 0

        def suggest_experiments(self, num_experiments=1, **kwargs):
            values = 0.5 * np.ones([num_experiments, 2])
            self.iterations += 1
            return DataSet(values, columns=["x_1", "x_2"])

        def reset(self):
            pass

    class MockExperiment(Experiment):
        def __init__(self):
            super().__init__(self.create_domain())

        def create_domain(self):
            domain = Domain()
            domain += ContinuousVariable("x_1", description="", bounds=[0, 1])
            domain += ContinuousVariable("x_2", description="", bounds=[0, 1])
            domain += ContinuousVariable(
                "y_1", description="", bounds=[0, 1], is_objective=True, maximize=True
            )
            return domain

        def _run(self, conditions, **kwargs):
            conditions[("y_1", "DATA")] = 0.5
            return conditions, {}

    class MockNeptuneExperiment:
        def send_metric(self, metric, value):
            pass

        def send_artifact(self, filename):
            pass

        def stop(self):
            pass

    exp = MockExperiment()
    strategy = MockStrategy(exp.domain)
    r = runner(
        strategy=strategy,
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
        max_same=max_same,
        max_restarts=max_restarts,
        neptune_project="sustainable-processes/summit",
        neptune_experiment_name="test_experiment",
        neptune_exp=MockNeptuneExperiment(),
    )
    r.run()

    # Check that correct number of iterations run
    if max_same is not None:
        iterations = (max_restarts + 1) * max_same
        iterations = iterations if iterations < max_iterations else max_iterations
    else:
        iterations = max_iterations

    assert r.strategy.iterations == iterations
    assert r.experiment.data.shape[0] == int(batch_size * iterations)

    # Check that reset works
    r.reset()
    assert r.experiment.data.shape[0] == 0

    # Check that using previous data works
    r.strategy.iterations = 0
    suggestions = r.strategy.suggest_experiments(num_experiments=10)
    results = exp.run_experiments(suggestions)
    r.run(prev_res=results)
    assert r.strategy.iterations == iterations + 1
    assert r.experiment.data.shape[0] == int(batch_size * iterations + 10)


@pytest.mark.parametrize("strategy", [SOBO, SNOBFIT, NelderMead, Random, LHS])
@pytest.mark.parametrize(
    "experiment",
    [
        Himmelblau,
        Hartmann3D,
        ThreeHumpCamel,
        get_pretrained_baumgartner_cc_emulator(include_cost=True),
    ],
)
def test_runner_so_integration(strategy, experiment):
    if not isinstance(experiment, ExperimentalEmulator):
        exp = experiment()
    else:
        exp = experiment

    s = strategy(exp.domain)

    r = Runner(strategy=s, experiment=exp, max_iterations=1, batch_size=1)
    r.run()

    # Try saving and loading
    r.save("test_save.json")
    r.load("test_save.json")
    os.remove("test_save.json")


@pytest.mark.parametrize("strategy", [SOBO, SNOBFIT, NelderMead, Random, LHS, TSEMO])
@pytest.mark.parametrize(
    "experiment",
    [
        get_pretrained_reizman_suzuki_emulator(),
        get_pretrained_baumgartner_cc_emulator(include_cost=True),
        DTLZ2,
        VLMOP2,
        SnarBenchmark,
    ],
)
def test_runner_mo_integration(strategy, experiment):
    """Test Runner with multiobjective optimization strategies and benchmarks"""
    if not isinstance(experiment, ExperimentalEmulator):
        exp = experiment()
    else:
        exp = experiment

    if experiment.__class__.__name__ == "ReizmanSuzukiEmulator" and strategy not in [
        SOBO,
        TSEMO,
    ]:
        # only run on strategies that work with categorical variables direclty
        return
    elif strategy == TSEMO:
        s = strategy(exp.domain)
        iterations = 10
    else:
        hierarchy = {
            v.name: {"hierarchy": i, "tolerance": 1}
            for i, v in enumerate(exp.domain.output_variables)
        }
        transform = Chimera(exp.domain, hierarchy)
        s = strategy(exp.domain, transform=transform)
        iterations = 3

    r = Runner(strategy=s, experiment=exp, num_initial_experiments=8, max_iterations=iterations, batch_size=1)
    r.run()

    # Try saving and loading
    # r.save("test_save.json")
    # r.load("test_save.json")
    # os.remove("test_save.json")
