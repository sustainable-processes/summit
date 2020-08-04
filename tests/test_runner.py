import pytest
from summit import NeptuneRunner, Runner, Strategy, Experiment
from summit.strategies import *
from summit.benchmarks import *
from summit.domain import *
from summit.utils.dataset import DataSet

import numpy as np
import os


def test_runner_unit():
    class MockStrategy(Strategy):
        iterations = 0

        def suggest_experiments(self, num_experiments=1, **kwargs):
            values = 0.5 * np.ones([num_experiments, 2])
            self.iterations += 1
            return DataSet(values, columns=["x_1", "x_2"])

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

    max_iterations = 10
    batch_size = 5
    exp = MockExperiment()
    r = Runner(
        strategy=MockStrategy(exp.domain),
        experiment=exp,
        max_iterations=max_iterations,
        batch_size=batch_size,
    )
    r.run()

    # Check that correct number of iterations run
    assert r.strategy.iterations == max_iterations
    assert r.experiment.data.shape[0] == int(batch_size * max_iterations)


@pytest.mark.parametrize(
    "strategy", [DRO, SOBO, SNOBFIT, GRYFFIN, NelderMead, Random, LHS]
)
@pytest.mark.parametrize(
    "experiment",
    [
        Himmelblau,
        Hartmann3D,
        ThreeHumpCamel,
        BaumgartnerCrossCouplingEmulator,
        BaumgartnerCrossCouplingDescriptorEmulator,
    ],
)
def test_runner_so_integration(strategy, experiment):
    """Runner integration test with single-objective benchmarks and strategies"""
    exp = experiment()
    s = strategy(exp.domain)

    if experiment in [
        BaumgartnerCrossCouplingEmulator,
        BaumgartnerCrossCouplingDescriptorEmulator,
    ]:
        return

    r = Runner(strategy=s, experiment=exp, max_iterations=1, batch_size=1)
    r.run()

    # Try saving and loading
    r.save("test_save.json")
    r.load("test_save.json")
    os.remove("test_save.json")


@pytest.mark.parametrize(
    "strategy", [DRO, SOBO, SNOBFIT, GRYFFIN, NelderMead, Random, LHS, TSEMO]
)
@pytest.mark.parametrize(
    "experiment",
    [
        SnarBenchmark,
        ReizmanSuzukiEmulator,
        BaumgartnerCrossCouplingEmulator_Yield_Cost,
        DTLZ2,
        VLMOP2,
    ],
)
def test_runner_mo_integration(strategy, experiment):
    """Runner integration test with multiobjective benchmarks and strategies"""
    exp = experiment()

    if (
        experiment
        in [ReizmanSuzukiEmulator, BaumgartnerCrossCouplingEmulator_Yield_Cost]
        and strategy == DRO
    ):
        return

    if experiment == ReizmanSuzukiEmulator and strategy not in [SOBO, GRYFFIN]:
        # only run on strategies that work with categorical variables deireclty
        return
    elif strategy == TSEMO:
        s = strategy(exp.domain)
    else:
        hierarchy = {
            v.name: {"hierarchy": 0, "tolerance": 1}
            for v in exp.domain.output_variables
        }
        transform = Chimera(exp.domain, hierarchy)
        s = strategy(exp.domain, transform=transform)

    r = Runner(strategy=s, experiment=exp, max_iterations=1, batch_size=1)
    r.run()

    # Try saving and loading
    r.save("test_save.json")
    r.load("test_save.json")
    os.remove("test_save.json")


# def test_neptune_runner_integration():
#     class MockExperiment(Experiment):
#         def __init__(self):
#             super().__init__(self.create_domain())

#         def create_domain(self):
#             domain = Domain()
#             domain += ContinuousVariable("x_1", description="", bounds=[0, 1])
#             domain += ContinuousVariable("x_2", description="", bounds=[0, 1])
#             domain += ContinuousVariable(
#                 "y_1", description="", bounds=[0, 1], is_objective=True, maximize=True
#             )
#             return domain

#         def _run(self, conditions, **kwargs):
#             conditions[("y_1", "DATA")] = 0.5
#             return conditions, {}

#     exp = MockExperiment()
#     strategy = Random(exp.domain)

#     r = NeptuneRunner(
#         neptune_project="sustainable-processes/summit",
#         neptune_experiment_name="test_experiment",
#         strategy=strategy,
#         experiment=exp,
#         max_iterations=1,
#         batch_size=1,
#     )
#     r.run()
