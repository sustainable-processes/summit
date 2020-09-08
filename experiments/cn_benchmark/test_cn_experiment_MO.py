import pytest

from .slurm_runner import SlurmRunner
from summit import *
from summit.benchmarks import BaumgartnerCrossCouplingEmulator_Yield_Cost
from summit.strategies import *

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
HYPERVOLUME_REF = [0, 1]

# Cross Coupling Benchmark by Emulator trained on Baumgartner et al. (2019) data
# This emulator is modified to include costs as an output
experiment = BaumgartnerCrossCouplingEmulator_Yield_Cost()

# Transforms from multi to single objective
hierarchies = [
    {"yld": {"hierarchy": 0, "tolerance": 1}, "cost": {"hierarchy": 1, "tolerance": 1}},
    {
        "yld": {"hierarchy": 0, "tolerance": 0.5},
        "cost": {"hierarchy": 1, "tolerance": 0.5},
    },
    {
        "yld": {"hierarchy": 0, "tolerance": 1.0},
        "cost": {"hierarchy": 1, "tolerance": 0.5},
    },
    {
        "yld": {"hierarchy": 0, "tolerance": 0.5},
        "cost": {"hierarchy": 1, "tolerance": 1.0},
    },
]
transforms = [
    Chimera(experiment.domain, hierarchies[2]),
    MultitoSingleObjective(experiment.domain, expression="-yld+cost", maximize=False),
    Chimera(experiment.domain, hierarchies[0]),
    Chimera(experiment.domain, hierarchies[1]),
    Chimera(experiment.domain, hierarchies[3]),
]

# Run experiments
@pytest.mark.parametrize("strategy", [Random])
def test_baselines(strategy):
    """Test Multiobjective CN Benchmark with baseline strategies (random, full factorial)"""
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(experiment.domain, transform_descriptors=True)

        name = f"cn_experiment_MO_baselines_{s.__class__.__name__}_repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            docker_container="marcosfelt/summit:cn_benchmark",
            neptune_project=NEPTUNE_PROJECT,
            neptune_experiment_name=name,
            neptune_tags=["cn_experiment_MO", s.__class__.__name__],
            neptune_files=["slurm_summit_cn_experiment.sh"],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            hypervolume_ref=HYPERVOLUME_REF,
        )
        r.run(save_at_end=True)


@pytest.mark.parametrize("strategy", [NelderMead, SNOBFIT, SOBO, GRYFFIN])
@pytest.mark.parametrize("transform", transforms)
def test_cn_experiment_descriptors(strategy, transform):
    """Test multiobjective CN benchmark with descriptors and multiobjective transforms"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for i in range(NUM_REPEATS):
        experiment.reset()

        s = strategy(
            experiment.domain,
            transform=transform,
            use_descriptors=True,
            auto_desc_gen=True,
            sampling_strategies=1,
        )

        # Early stopping for local optimization strategies
        if strategy == NelderMead:
            s.random_start = True
            max_same = 2
            max_restarts = 10
            s.adaptive = True
        else:
            max_same = None
            max_restarts = 0

        name = f"cn_experiment_MO_descriptors_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            neptune_project=NEPTUNE_PROJECT,
            docker_container="marcosfelt/summit:cn_benchmark",
            neptune_experiment_name=name,
            neptune_tags=[
                "cn_experiment_MO",
                "descriptors",
                s.__class__.__name__,
                transform.__class__.__name__,
            ],
            neptune_files=["slurm_summit_cn_experiment.sh"],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            max_same=max_same,
            max_restarts=max_restarts,
            hypervolume_ref=HYPERVOLUME_REF,
        )
        r.run(save_at_end=True)


def test_cn_experiment_tsemo():
    """Test multiobjective CN benchmark with descriptors and TSEMO (multiobjective strategy)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for i in range(1):
        experiment.reset()
        s = TSEMO(experiment.domain, n_spectral_points=4000)

        name = f"cn_experiment_MO_{s.__class__.__name__}_repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            neptune_project=NEPTUNE_PROJECT,
            docker_container="marcosfelt/summit:cn_benchmark",
            neptune_experiment_name=name,
            neptune_tags=["cn_experiment", "descriptors", s.__class__.__name__],
            neptune_files=["slurm_summit_cn_experiment.sh"],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            hypervolume_ref=HYPERVOLUME_REF,
        )
        r.run(save_at_end=True)


@pytest.mark.parametrize("strategy", [GRYFFIN, SOBO])
@pytest.mark.parametrize("transform", transforms)
def test_cn_experiment_no_descriptors(strategy, transform):
    """Test Multiobjective CN Benchmark with no descriptors"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for i in range(NUM_REPEATS):
        experiment.reset()
        s = strategy(
            experiment.domain,
            transform=transform,
            use_descriptors=False,
            auto_desc_gen=True,
            sampling_strategies=1,
        )

        name = f"cn_experiment_MO_no_descriptors_{s.__class__.__name__}_{transform.__class__.__name__}_repeat_{i}"
        r = SlurmRunner(
            experiment=experiment,
            strategy=s,
            neptune_project=NEPTUNE_PROJECT,
            docker_container="marcosfelt/summit:cn_benchmark",
            neptune_experiment_name=name,
            neptune_tags=[
                "cn_experiment_MO",
                "no_descriptors",
                s.__class__.__name__,
                transform.__class__.__name__,
            ],
            neptune_files=["slurm_summit_cn_experiment.sh"],
            max_iterations=MAX_EXPERIMENTS // BATCH_SIZE,
            batch_size=BATCH_SIZE,
            hypervolume_ref=HYPERVOLUME_REF,
        )
        r.run(save_at_end=True)

