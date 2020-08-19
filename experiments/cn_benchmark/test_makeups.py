""" SnAr Experiment Base Script

Run this with pytest: `pytest test_snar_experiment.py``

You need to have your Neptune API token as an environmental variable,
SSH_USER and SSH_PASSWORD should be set if you want to use SlurmRunner.

"""
import pytest

from summit import *
from summit.benchmarks import BaumgartnerCrossCouplingEmulator_Yield_Cost
from summit.strategies import *
from .slurm_runner import SlurmRunner

import warnings
import logging
import os

logging.basicConfig(level=logging.INFO)
token = os.environ.get("NEPTUNE_API_TOKEN")
if token is None:
    raise ValueError("Neptune_API_TOKEN needs to be an environmental variable")

MAX_EXPERIMENTS = 50
NEPTUNE_PROJECT = "sustainable-processes/summit"
BATCH_SIZE = 1
HYPERVOLUME_REF = [0, 1]

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


@pytest.mark.parametrize("transform", transforms)
@pytest.mark.parametrize("use_descriptors", [False, True])
def test_gryffin(transform, use_descriptors):
    experiment.reset()
    s = GRYFFIN(
        experiment.domain,
        transform=transform,
        sampling_strategies=4,
        use_descriptors=use_descriptors,
        auto_desc_gen=True,
    )

    name = f"cn_experiment_MO_baselines_{s.__class__.__name__}"
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


@pytest.mark.parametrize(
    "transform",
    12 * [transforms[0]]
    + 17 * [transforms[1]]
    + 12 * [transforms[2]]
    + 12 * [transforms[3]]
    + 20 * [transforms[4]],
)
def test_sobo(transform):
    experiment.reset()
    s = SOBO(experiment.domain, transform=transform)

    name = f"cn_experiment_MO_baselines_{s.__class__.__name__}"
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

