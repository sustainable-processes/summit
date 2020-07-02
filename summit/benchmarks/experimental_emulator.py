import os
import os.path as osp

from summit.experiment import Experiment

import numpy as np

from summit.benchmarks.experiment_emulator.bnn_regressor import BNNEmulator
from summit.utils.dataset import DataSet
from summit.domain import *


class ExperimentalEmulator(Experiment):
    """ Experimental Emulator
    
    Examples
    --------
    >>> test_domain = ExperimentalEmulator()
    >>> e = ExperimentalEmulator(test_domain)
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = {
    >>>     ("catalyst", "DATA"): ["P1-L2", "P1-L7", "P1-L3"],
    >>>     ("t_res", "DATA"): [60, 120, 110],
    >>>     ("temperature", "DATA"): [110, 170, 250],
    >>>     ("catalyst_loading", "DATA"): [0.508, 0.6, 1.4],
    >>>     ("yield", "DATA"): [20, 40, 60],
    >>>     ("ton", "DATA"): [33, 34, 21]
    >>> }
    >>> e.train()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.6*(v.bounds[1]-v.bounds[0]) if v.variable_type == 'continuous' else v.levels[-1] for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = e.run_experiments(conditions)


    Notes
    -----

    
    """

    def __init__(self, domain, dataset=None, csv_dataset=None, model_name="dataset_name_emulator_bnn", regressor_type="BNN", **kwargs):
        super().__init__(domain)

        dataset = self._check_datasets(dataset, csv_dataset)

        if regressor_type == "BNN":
            self.emulator = BNNEmulator(domain=domain, dataset=dataset, model_name=model_name, kwargs=kwargs)
            try:
                self.extras = [self.emulator._load_model(model_name)]
            except:
                print("No trained model for {}. Please train this model with ExperimentalEmulator.train().".format(self.emulator.model_name))
        else:
            raise NotImplementedError("Regressor type <{}> not implemented yet".format(str(regressor_type)))


    def _run(self, conditions, **kwargs):
        condition = DataSet.from_df(conditions.to_frame().T)
        infer_dict = self.emulator.infer_model(dataset=condition)
        for k, v in infer_dict.items():
            conditions[(k, "DATA")] = v
        return conditions, None

    def train(self, dataset=None, csv_dataset=None, **kwargs):
        dataset = self._check_datasets(dataset, csv_dataset)
        self.emulator.set_training_hyperparameters(kwargs=kwargs)
        self.emulator.train_model(dataset=dataset, kwargs=kwargs)
        self.extras = [self.emulator.output_models]

    def validate(self, dataset=None, csv_dataset=None, **kwargs):
        dataset = self._check_datasets(dataset, csv_dataset)
        return self.emulator.validate_model(dataset=dataset, kwargs=kwargs)

    def _check_datasets(self, dataset=None, csv_dataset=None):
        if csv_dataset:
            if dataset:
                print("Dataset and csv.dataset are given, hence dataset will be overwritten by csv.data.")
            dataset=DataSet.read_csv(csv_dataset, index_col=None)
        return dataset


class ReizmanSuzukiEmulator(ExperimentalEmulator):
    """ Reizman Suzuki Emulator

    Virtual experiments representing the Suzuki-Miyaura Cross-Coupling reaction
    similar to Reizman et al. (2016). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Reizman et al.

    Parameters
    ----------
    case: int, optional, default=1
        Reizman et al. (2016) reported experimental data for 4 different
        cases. The case number refers to the cases they reported.
        Please see their paper for more information on the cases.

    Examples
    --------
    >>> reizman_emulator = ReizmanSuzukiEmulator(case=1)

    Notes
    -----
    This benchmark is based on Reizman et al. React. Chem. Eng. 2016.
    https://doi.org/10.1039/C6RE00153J

    """

    def __init__(self, case=1, **kwargs):
        domain = self.setup_domain()
        dataset_file = osp.join(osp.dirname(osp.realpath(__file__)), "experiment_emulator/data/reizman_suzuki_case" + str(case)+ "_train_test.csv")
        super().__init__(domain=domain, csv_dataset=dataset_file, model_name="reizman_suzuki")

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Catalyst type - different ligands"
        domain += DiscreteVariable(
            name="catalyst", description=des_1, levels=["P1-L1", "P2-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7"])

        des_2 = "Residence time in seconds (s)"
        domain += ContinuousVariable(
            name="t_res", description=des_2, bounds=[60, 600]
        )

        des_3 = "Reactor temperature in degrees Celsius (ºC)"
        domain += ContinuousVariable(
            name="temperature", description=des_3, bounds=[30, 110]
        )

        des_4 = "Catalyst loading in mol%"
        domain += ContinuousVariable(
            name="catalyst_loading", description=des_4, bounds=[0.5, 2.5]
        )

        # Objectives
        des_5 = "Turnover number - moles product generated divided by moles catalyst used"
        domain += ContinuousVariable(
            name="ton",
            description=des_5,
            bounds=[0, 200],   # TODO: not sure about bounds, maybe redefine
            is_objective=True,
            maximize=True,
        )

        des_6 = "Yield"
        domain += ContinuousVariable(
            name="yield",
            description=des_6,
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )

        return domain

