from summit.benchmarks import ReizmanSuzukiEmulator, ExperimentalEmulator
from summit.utils.dataset import DataSet

import numpy as np

b = ReizmanSuzukiEmulator()
columns = [v.name for v in b.domain.variables]
values = {
    ("catalyst", "DATA"): ["P1-L2", "P1-L7", "P1-L3"],
    ("t_res", "DATA"): [60, 120, 110],
    ("temperature", "DATA"): [110, 170, 250],
    ("catalyst_loading", "DATA"): [0.508, 0.6, 1.4],
    ("yield", "DATA"): [20, 40, 60],
    ("ton", "DATA"): [33, 34, 21]
}
dataset = DataSet(values, columns=columns)
print(dataset)
print(b.domain, dataset)
#e = ExperimentalEmulator(b.domain, dataset, train=True, validate=False, epochs=300)
e = ExperimentalEmulator(b.domain, dataset, train=True, validate=False)
#e.train_model(dataset=dataset, epochs=300)
values = {
    ("catalyst", "DATA"): ["P1-L2", "P1-L3"],
    ("t_res", "DATA"): [60, 120],
    ("temperature", "DATA"): [130, 143],
    ("catalyst_loading", "DATA"): [0.54, 0.7],
}
conditions = DataSet(values, columns=columns)
results = e.run_experiments(conditions)
print(results)
values = {
    ("temperature", "DATA"): [130, 143],
    ("t_res", "DATA"): [60, 120],
    ("catalyst", "DATA"): ["P1-L2", "P1-L3"],
    ("catalyst_loading", "DATA"): [0.54, 0.7],
}
conditions = DataSet(values, columns=columns)
results = e.run_experiments(conditions)
print(results)