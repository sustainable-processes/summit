""" SnAr Experiment Base Script

Run this with pytest: `pytest test_snar_experiment.py``

You need to have your Neptune API token as an environmental variable,
SSH_USER and SSH_PASSWORD should be set if you want to use SlurmRunner.

"""
import pytest

from summit import *
from summit.benchmarks import SnarBenchmark
from summit.strategies import *
from .slurm_runner import SlurmRunner

import warnings
import logging
import os

logging.basicConfig(level=logging.INFO)
token = os.environ.get("NEPTUNE_API_TOKEN")
if token is None:
    raise ValueError("Neptune_API_TOKEN needs to be an environmental variable")

NUM_REPEATS = 20
MAX_EXPERIMENTS = 50
NEPTUNE_PROJECT = "sustainable-processes/summit"
BATCH_SIZE = 1

# SnAr benchmark with 2.5% experimental measurement noise
experiment = SnarBenchmark(noise_level_percent=2.5)

# Transforms from multi to single objective
hierarchies = [
    {
        "sty": {"hierarchy": 0, "tolerance": 1},
        "e_factor": {"hierarchy": 1, "tolerance": 1},
    },
    {
        "sty": {"hierarchy": 0, "tolerance": 0.5},
        "e_factor": {"hierarchy": 1, "tolerance": 0.5},
    },
    {
        "sty": {"hierarchy": 0, "tolerance": 1.0},
        "e_factor": {"hierarchy": 1, "tolerance": 0.5},
    },
    {
        "sty": {"hierarchy": 0, "tolerance": 0.5},
        "e_factor": {"hierarchy": 1, "tolerance": 1.0},
    },
]
transforms = [
    Chimera(experiment.domain, hierarchies[2]),
    MultitoSingleObjective(
        experiment.domain, expression="-sty/1e4+e_factor/100", maximize=False
    ),
    Chimera(experiment.domain, hierarchies[0]),
    Chimera(experiment.domain, hierarchies[1]),
    Chimera(experiment.domain, hierarchies[3]),
]


@pytest.mark.parametrize("strategy", [Random, TSEMO])
def test_no_transform(strategy):
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain)

        exp_name = f"snar_experiment_{s.__class__.__name__}__repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            neptune_project=NEPTUNE_PROJECT,
            docker_container="marcosfelt/summit:cn_benchmark",
            neptune_experiment_name=exp_name,
            neptune_files=["slurm_summit_snar_experiment.sh"],
            neptune_tags=["snar_experiment", s.__class__.__name__],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            num_initial_experiments=1,
            hypervolume_ref=[-2957, 10.7],
        )
        r.run(save_at_end=True)


# Run experiments
@pytest.mark.parametrize("strategy", [DRO, NelderMead, SNOBFIT, SOBO, GRYFFIN])
@pytest.mark.parametrize("transform", transforms)
def test_snar_experiment(strategy, transform):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain, transform=transform, sampling_strategies=1)

        # Special considerations for Neldermead
        if strategy == NelderMead:
            f_tol = 1e-5
            s.random_start = True
            max_restarts = 10
            s.adaptive = True
        else:
            f_tol = None
            max_restarts = 0

        container = "marcosfelt/summit:cn_benchmark"
        if strategy == DRO:
            container = "marcosfelt/summit:dro"
            strategy._model_size = "bigger"

        exp_name = f"snar_experiment_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            docker_container=container,
            neptune_project=NEPTUNE_PROJECT,
            neptune_experiment_name=exp_name,
            neptune_files=["slurm_summit_snar_experiment.sh"],
            neptune_tags=[
                "snar_experiment",
                s.__class__.__name__,
                transform.__class__.__name__,
                "bigger_model",
            ],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            f_tol=f_tol,
            max_restarts=max_restarts,
            num_initial_experiments=1,
            hypervolume_ref=[-2957, 10.7],
        )
        r.run(save_at_end=True)

