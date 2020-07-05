import os
import os.path as osp

from summit.experiment import Experiment

import numpy as np

from summit.benchmarks.experiment_emulator.bnn_regressor import BNNEmulator
from summit.utils.dataset import DataSet
from summit.domain import *

import matplotlib.pyplot as plt


class ExperimentalEmulator(Experiment):
    """ Experimental Emulator
    
    Examples
    --------
    >>> test_domain = ReizmanSuzukiEmulator().domain
    >>> e = ExperimentalEmulator(domain=test_domain, model_name="Pytest")
    No trained model for Pytest. Train this model with ExperimentalEmulator.train() in order to use this Emulator as an virtual Experiment.
    >>> columns = [v.name for v in e.domain.variables]
    >>> train_values = {("catalyst", "DATA"): ["P1-L2", "P1-L7", "P1-L3"], ("t_res", "DATA"): [60, 120, 110], ("temperature", "DATA"): [110, 170, 250], ("catalyst_loading", "DATA"): [0.508, 0.6, 1.4], ("yield", "DATA"): [20, 40, 60], ("ton", "DATA"): [33, 34, 21]}
    >>> train_dataset = DataSet(train_values, columns=columns)
    >>> e.train(train_dataset, verbose=False, test_size=0.1)
    >>> columns = [v.name for v in e.domain.variables]
    >>> values = [float(v.bounds[0] + 0.6 * (v.bounds[1] - v.bounds[0])) if v.variable_type == 'continuous' else v.levels[-1] for v in e.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = e.run_experiments(conditions)

    Notes
    -----
    """

# =======================================================================

    def __init__(self, domain, dataset=None, csv_dataset=None, model_name="dataset_name_emulator_bnn", regressor_type="BNN", **kwargs):
        super().__init__(domain)

        dataset = self._check_datasets(dataset, csv_dataset)

        if regressor_type == "BNN":
            self.emulator = BNNEmulator(domain=domain, dataset=dataset, model_name=model_name, kwargs=kwargs)
            try:
                self.extras = [self.emulator._load_model(model_name)]
            except:
                print("No trained model for {}. Train this model with ExperimentalEmulator.train() in order to use this Emulator as an virtual Experiment.".format(self.emulator.model_name))
        else:
            raise NotImplementedError("Regressor type <{}> not implemented yet".format(str(regressor_type)))

# =======================================================================

    def _run(self, conditions, **kwargs):
        condition = DataSet.from_df(conditions.to_frame().T)
        infer_dict = self.emulator.infer_model(dataset=condition)
        for k, v in infer_dict.items():
            conditions[(k, "DATA")] = v
        return conditions, None

# =======================================================================

    def train(self, dataset=None, csv_dataset=None, verbose=True, **kwargs):
        dataset = self._check_datasets(dataset, csv_dataset)
        self.emulator.set_training_hyperparameters(kwargs=kwargs)
        self.emulator.train_model(dataset=dataset, verbose=verbose, kwargs=kwargs)
        self.extras = [self.emulator.output_models]

# =======================================================================

    def validate(self, dataset=None, csv_dataset=None, parity_plots=False, **kwargs):
        dataset = self._check_datasets(dataset, csv_dataset)
        if dataset is not None:
            return self.emulator.validate_model(dataset=dataset, parity_plots=parity_plots, kwargs=kwargs)
        else:
            #try:
            print("Evaluation based on training and test set.")
            return self.emulator.validate_model(parity_plots=parity_plots)
            #except:
            #    raise ValueError("No dataset to evaluate.")

# =======================================================================

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

# =======================================================================

    def __init__(self, case=1, **kwargs):
        model_name = "reizman_suzuki_case" + str(case)
        domain = self.setup_domain()
        dataset_file = osp.join(osp.dirname(osp.realpath(__file__)), "experiment_emulator/data/" + model_name + "_train_test.csv")
        super().__init__(domain=domain, csv_dataset=dataset_file, model_name=model_name)

# =======================================================================

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

