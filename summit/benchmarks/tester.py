from summit.benchmarks import ReizmanSuzukiEmulator

import numpy as np

from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.benchmarks.experiment_emulator import experimental_datasets

b = ReizmanSuzukiEmulator()
columns = [v.name for v in b.domain.variables]
values = [v.bounds[0]+0.6*(v.bounds[1]-v.bounds[0]) if v.variable_type == 'continuous' else v.levels[-1] for v in b.domain.variables]
values = np.array(values)
values = np.atleast_2d(values)
conditions = DataSet(values, columns=columns)
results = b.run_experiments(conditions)

