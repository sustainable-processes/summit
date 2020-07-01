import os
import os.path as osp

from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.benchmarks.experiment_emulator import experimental_datasets

import numpy as np
from scipy.integrate import solve_ivp

from summit.benchmarks.experiment_emulator.bnn_regressor import BNNEmulator
from summit.benchmarks import ReizmanSuzukiEmulator
from summit.utils.dataset import DataSet


class ExperimentalEmulator(Experiment):
    """ Reizman Suzuki Emulator

    Virtual experiments representing the Suzuki-Miyaura Cross-Coupling reaction
    similar to Reizman et al. (2016). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Reizman et al. 
    
    Examples
    --------
    >>> b = ReizmanSuzukiEmulator()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = {
    >>>     ("catalyst", "DATA"): ["P1-L2", "P1-L7", "P1-L3"],
    >>>     ("t_res", "DATA"): [60, 120, 110],
    >>>     ("temperature", "DATA"): [110, 170, 250],
    >>>     ("catalyst_loading", "DATA"): [0.508, 0.6, 1.4],
    >>>     ("yield", "DATA"): [20, 40, 60],
    >>>     ("ton", "DATA"): [33, 34, 21]
    >>> }
    >>> b = Emulator(domain, dataset)
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.6*(v.bounds[1]-v.bounds[0]) if v.variable_type == 'continuous' else v.levels[-1] for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    
    Notes
    -----
    This benchmark is based on Reizman et al. React. Chem. Eng. 2016. 
    https://doi.org/10.1039/C6RE00153J
    
    """

    def __init__(self, domain, dataset=None, train=False, validate=False, regressor_type="BNN", **kwargs):
        super().__init__(domain)

        if regressor_type == "BNN":
            self.emulator = BNNEmulator(domain=domain, dataset=dataset, model_name="TEST", train=train, validate=validate, kwargs=kwargs)
        else:
            raise NotImplementedError("Regressor type <{}> not implemented yet".format(str(regressor_type)))


    def _run(self, conditions, **kwargs):
        condition = DataSet.from_df(conditions.to_frame().T)
        infer_dict = self.emulator.infer_model(condition)
        for k, v in infer_dict.items():
            conditions[(k, "DATA")] = v
        return conditions, None

    def train_model(self, dataset, **kwargs):
        self.emulator.set_training_hyperparameters(kwargs)
        self.emulator.train_model(dataset=dataset)


