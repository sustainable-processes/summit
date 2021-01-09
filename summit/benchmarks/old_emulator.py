import os
import os.path as osp

from summit.experiment import Experiment

import numpy as np

from summit.benchmarks.experiment_emulator.bnn_emulator import BNNEmulator
from summit.utils.dataset import DataSet
from summit.domain import *
from summit.utils import jsonify_dict, unjsonify_dict


class ExperimentalEmulator(Experiment):
    """Experimental Emulator

    Parameters
    ---------
    domain: summit.domain.Domain
        The domain of the experiment
    dataset: class:~summit.utils.dataset.DataSet, optional
        A DataSet with data for training where the data columns correspond to the domain and the data rows correspond to the training points.
        By default: None
    csv_dataset: string, optional
        Path to csv_file with data for training where columns correspond to the domain and the rows correspond to the training points.
        Note that the first row should exactly match the variable names of the domain and the second row should only have "DATA" as entry.
        By default: None
    model_name: string, optional
        Name of the model that is used for saving model parameters. Should be unique.
        By default: "dataset_emulator_model_name"
    regressor_type: string, optional
        Type of the regressor that is used within the emulator (available: "BNN").
        By default: "BNN"
    cat_to_descr: Boolean, optional
        If True, transform categorical variable to one or more continuous variable(s)
        corresponding to the descriptors of the categorical variable (else do nothing).
        By default: False

    Examples
    --------
    >>> test_domain = ReizmanSuzukiEmulator().domain
    >>> e = ExperimentalEmulator(domain=test_domain, model_name="Pytest")
    No trained model for Pytest. Train this model with ExperimentalEmulator.train() in order to use this Emulator as an virtual Experiment.
    >>> columns = [v.name for v in e.domain.variables]
    >>> train_values = {("catalyst", "DATA"): ["P1-L2", "P1-L7", "P1-L3", "P1-L3"], ("t_res", "DATA"): [60, 120, 110, 250], ("temperature", "DATA"): [110, 30, 70, 80], ("catalyst_loading", "DATA"): [0.508, 0.6, 1.4, 1.3], ("yield", "DATA"): [20, 40, 60, 34], ("ton", "DATA"): [33, 34, 21, 22]}
    >>> train_dataset = DataSet(train_values, columns=columns)
    >>> e.train(train_dataset, verbose=False, cv_fold=2, test_size=0.25)
    >>> columns = [v.name for v in e.domain.variables]
    >>> values = [float(v.bounds[0] + 0.6 * (v.bounds[1] - v.bounds[0])) if v.variable_type == 'continuous' else v.levels[-1] for v in e.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = e.run_experiments(conditions)

    """

    def __init__(
        self,
        domain,
        dataset=None,
        csv_dataset=None,
        model_name="dataset_name_emulator_bnn",
        regressor_type="BNN",
        cat_to_descr=False,
        **kwargs
    ):
        super().__init__(domain)
        dataset = self._check_datasets(dataset, csv_dataset)

        kwargs["cat_to_descr"] = cat_to_descr

        if regressor_type == "BNN":
            self.emulator = BNNEmulator(
                domain=domain, dataset=dataset, model_name=model_name, kwargs=kwargs
            )
            try:
                self.extras = [self.emulator._load_model(model_name)]
            except:
                print(
                    "No trained model for {}. Train this model with ExperimentalEmulator.train() in order to use this Emulator as an virtual Experiment.".format(
                        self.emulator.model_name
                    )
                )
        else:
            raise NotImplementedError(
                "Regressor type <{}> not implemented yet".format(str(regressor_type))
            )

    def _run(self, conditions, **kwargs):
        condition = DataSet.from_df(conditions.to_frame().T)
        infer_dict = self.emulator.infer_model(dataset=condition)
        for k, v in infer_dict.items():
            conditions[(k, "DATA")] = v
        return conditions, None

    def train(self, dataset=None, csv_dataset=None, verbose=True, **kwargs):
        dataset = self._check_datasets(dataset, csv_dataset)
        self.emulator.set_training_hyperparameters(kwargs=kwargs)
        self.emulator.train_model(dataset=dataset, verbose=verbose, kwargs=kwargs)
        self.extras = [self.emulator.output_models]

    def validate(
        self,
        dataset=None,
        csv_dataset=None,
        parity_plots=False,
        get_pred=False,
        **kwargs
    ):
        dataset = self._check_datasets(dataset, csv_dataset)
        if dataset is not None:
            return self.emulator.validate_model(
                dataset=dataset,
                parity_plots=parity_plots,
                get_pred=get_pred,
                kwargs=kwargs,
            )
        else:
            try:
                print("Evaluation based on training and test set.")
                return self.emulator.validate_model(parity_plots=parity_plots)
            except:
                raise ValueError("No dataset to evaluate.")

    def _check_datasets(self, dataset=None, csv_dataset=None):
        if csv_dataset:
            if dataset:
                print(
                    "Dataset and csv.dataset are given, hence dataset will be overwritten by csv.data."
                )
            dataset = DataSet.read_csv(csv_dataset, index_col=None)
        return dataset

    def to_dict(self, **kwargs):
        """Serialize the class to a dictionary

        Subclasses can add a experiment_params dictionary
        key with custom parameters for the experiment
        """
        kwargs.update(
            dict(
                model_name=self.emulator.model_name,
                dataset=self.emulator._dataset.to_dict()
                if self.emulator._dataset is not None
                else None,
                output_models=self.emulator.output_models,
            )
        )
        return super().to_dict(**kwargs)

    @classmethod
    def from_dict(cls, d):
        dataset = d["experiment_params"]["dataset"]
        d["experiment_params"]["dataset"] = DataSet.from_dict(dataset)
        exp = super().from_dict(d)
        exp.emulator.output_models = d["experiment_params"]["output_models"]
        return exp


class ReizmanSuzukiEmulator(ExperimentalEmulator):
    """Reizman Suzuki Emulator

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
    This benchmark is based on data from [Reizman]_ et al.

    References
    ----------
    .. [Reizman] B. J. Reizman et al., React. Chem. Eng., 2016, 1, 658–666.
       DOI: `10.1039/C6RE00153J <https://doi.org/10.1039/C6RE00153J>`_.

    """

    def __init__(self, case=1, **kwargs):
        model_name = "reizman_suzuki_case" + str(case)
        domain = self.setup_domain()
        dataset_file = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "experiment_emulator/data/" + model_name + "_train_test.csv",
        )
        super().__init__(domain=domain, model_name=model_name)

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Catalyst type - different ligands"
        domain += CategoricalVariable(
            name="catalyst",
            description=des_1,
            levels=[
                "P1-L1",
                "P2-L1",
                "P1-L2",
                "P1-L3",
                "P1-L4",
                "P1-L5",
                "P1-L6",
                "P1-L7",
            ],
        )

        des_2 = "Residence time in seconds (s)"
        domain += ContinuousVariable(name="t_res", description=des_2, bounds=[60, 600])

        des_3 = "Reactor temperature in degrees Celsius (ºC)"
        domain += ContinuousVariable(
            name="temperature", description=des_3, bounds=[30, 110]
        )

        des_4 = "Catalyst loading in mol%"
        domain += ContinuousVariable(
            name="catalyst_loading", description=des_4, bounds=[0.5, 2.5]
        )

        # Objectives
        des_5 = (
            "Turnover number - moles product generated divided by moles catalyst used"
        )
        domain += ContinuousVariable(
            name="ton",
            description=des_5,
            bounds=[0, 200],  # TODO: not sure about bounds, maybe redefine
            is_objective=True,
            maximize=False,
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

    def to_dict(self):
        """Serialize the class to a dictionary"""
        experiment_params = dict(
            case=self.emulator.model_name[-1],
        )
        return super().to_dict(**experiment_params)


class BaumgartnerCrossCouplingEmulator(ExperimentalEmulator):
    """Baumgartner Cross Coupling Emulator

    Virtual experiments representing the Aniline Cross-Coupling reaction
    similar to Baumgartner et al. (2019). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Baumgartner et al.

    This is a five dimensional optimisation of temperature, residence time, base equivalents,
    catalyst and base.

    The categorical variables (catalyst and base) contain descriptors
    calculated using COSMO-RS. Specifically, the descriptors are the first two sigma moments.

    Examples
    --------
    >>> bemul = BaumgartnerCrossCouplingDescriptorEmulator()

    Notes
    -----
    This benchmark is based on data from [Baumgartner]_ et al.

    References
    ----------

    .. [Baumgartner] L. M. Baumgartner et al., Org. Process Res. Dev., 2019, 23, 1594–1601
       DOI: `10.1021/acs.oprd.9b00236 <https://`doi.org/10.1021/acs.oprd.9b00236>`_

    """

    def __init__(self, **kwargs):
        model_name = kwargs.get("model_name", "baumgartner_aniline_cn_crosscoupling")
        dataset_file = kwargs.get(
            "dataset_file", "baumgartner_aniline_cn_crosscoupling.csv"
        )
        domain = self.setup_domain()
        dataset_file = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "experiment_emulator/data/" + dataset_file,
        )
        super().__init__(domain=domain, csv_dataset=dataset_file, model_name=model_name)

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Catalyst type"
        catalyst_df = DataSet(
            [
                [460.7543, 67.2057],  # 30.8413, 2.3043, 0], #, 424.64, 421.25040226],
                [518.8408, 89.8738],  # 39.4424, 2.5548, 0], #, 487.7, 781.11247064],
                [819.933, 129.0808],  # 83.2017, 4.2959, 0], #, 815.06, 880.74916884],
            ],
            index=["tBuXPhos", "tBuBrettPhos", "AlPhos"],
            columns=[
                "area_cat",
                "M2_cat",
            ],  # , 'M3_cat', 'Macc3_cat', 'Mdon3_cat'] #,'mol_weight', 'sol']
        )
        domain += CategoricalVariable(
            name="catalyst",
            description=des_1,
            levels=["tBuXPhos", "tBuBrettPhos", "AlPhos"],
            descriptors=catalyst_df,
        )

        des_2 = "Base"
        base_df = DataSet(
            [
                [162.2992, 25.8165],  # 40.9469, 3.0278, 0], #101.19, 642.2973283],
                [
                    165.5447,
                    81.4847,
                ],  # 107.0287, 10.215, 0.0169], # 115.18, 534.01544123],
                [227.3523, 30.554],  # 14.3676, 1.1196, 0.0127], # 171.28, 839.81215],
                [192.4693, 59.8367],  # 82.0661, 7.42, 0], # 152.24, 1055.82799],
            ],
            index=["TEA", "TMG", "BTMG", "DBU"],
            columns=["area", "M2"],  # , 'M3', 'Macc3', 'Mdon3'], # 'mol_weight', 'sol']
        )
        domain += CategoricalVariable(
            name="base",
            description=des_2,
            levels=["DBU", "BTMG", "TMG", "TEA"],
            descriptors=base_df,
        )

        des_3 = "Base equivalents"
        domain += ContinuousVariable(
            name="base_equivalents", description=des_3, bounds=[1.0, 2.5]
        )

        des_4 = "Temperature in degrees Celsius (ºC)"
        domain += ContinuousVariable(
            name="temperature", description=des_4, bounds=[30, 100]
        )

        des_5 = "residence time in seconds (s)"
        domain += ContinuousVariable(name="t_res", description=des_5, bounds=[60, 1800])

        des_6 = "Yield"
        domain += ContinuousVariable(
            name="yld",
            description=des_6,
            bounds=[0.0, 1.0],
            is_objective=True,
            maximize=True,
        )

        return domain


class BaumgartnerCrossCouplingDescriptorEmulator(ExperimentalEmulator):
    """Baumgartner Cross Coupling Emulator

    Virtual experiments representing the Aniline Cross-Coupling reaction
    similar to Baumgartner et al. (2019). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Baumgartner et al.

    The difference with this model is that it uses descriptors for the catalyst and base
    instead of one-hot encoding the options. The descriptors are the first two
    sigma moments from COSMO-RS.


    Parameters
    ----------

    Examples
    --------
    >>> bemul = BaumgartnerCrossCouplingDescriptorEmulator()

    Notes
    -----
    This benchmark is based on data from [Baumgartner]_ et al.

    References
    ----------

    .. [Baumgartner] L. M. Baumgartner et al., Org. Process Res. Dev., 2019, 23, 1594–1601
       DOI: `10.1021/acs.oprd.9b00236 <https://doi.org/10.1021/acs.oprd.9b00236>`_

    """

    def __init__(self, **kwargs):
        model_name = kwargs.get(
            "model_name", "baumgartner_aniline_cn_crosscoupling_descriptors"
        )
        dataset_file = kwargs.get(
            "dataset_file", "baumgartner_aniline_cn_crosscoupling_descriptors.csv"
        )
        domain = self.setup_domain()
        dataset_file = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "experiment_emulator/data/" + dataset_file,
        )
        super().__init__(domain=domain, csv_dataset=dataset_file, model_name=model_name)

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Catalyst type with descriptors"
        catalyst_df = DataSet(
            [
                [460.7543, 67.2057, 30.8413, 2.3043, 0],  # , 424.64, 421.25040226],
                [518.8408, 89.8738, 39.4424, 2.5548, 0],  # , 487.7, 781.11247064],
                [819.933, 129.0808, 83.2017, 4.2959, 0],  # , 815.06, 880.74916884],
            ],
            index=["tBuXPhos", "tBuBrettPhos", "AlPhos"],
            columns=[
                "area_cat",
                "M2_cat",
                "M3_cat",
                "Macc3_cat",
                "Mdon3_cat",
            ],  # ,'mol_weight', 'sol']
        )
        domain += CategoricalVariable(
            name="catalyst", description=des_1, descriptors=catalyst_df
        )

        des_2 = "Base type with descriptors"
        base_df = DataSet(
            [
                [162.2992, 25.8165, 40.9469, 3.0278, 0],  # 101.19, 642.2973283],
                [165.5447, 81.4847, 107.0287, 10.215, 0.0169],  # 115.18, 534.01544123],
                [227.3523, 30.554, 14.3676, 1.1196, 0.0127],  # 171.28, 839.81215],
                [192.4693, 59.8367, 82.0661, 7.42, 0],  # 152.24, 1055.82799],
            ],
            index=["TEA", "TMG", "BTMG", "DBU"],
            columns=["area", "M2", "M3", "Macc3", "Mdon3"],  # 'mol_weight', 'sol']
        )
        domain += CategoricalVariable(
            name="base", description=des_2, descriptors=base_df
        )

        des_3 = "Base equivalents"
        domain += ContinuousVariable(
            name="base_equivalents", description=des_3, bounds=[1.0, 2.5]
        )

        des_4 = "Temperature in degrees Celsius (ºC)"
        domain += ContinuousVariable(
            name="temperature", description=des_4, bounds=[30, 100]
        )

        des_5 = "residence time in seconds (s)"
        domain += ContinuousVariable(name="t_res", description=des_5, bounds=[60, 1800])

        des_6 = "Yield"
        domain += ContinuousVariable(
            name="yield",
            description=des_6,
            bounds=[0.0, 1.0],
            is_objective=True,
            maximize=True,
        )

        return domain


class BaumgartnerCrossCouplingEmulator_Yield_Cost(BaumgartnerCrossCouplingEmulator):
    """Baumgartner Cross Coupling Emulator

    Virtual experiments representing the Aniline Cross-Coupling reaction
    similar to Baumgartner et al. (2019). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Baumgartner et al.

    This is a multiobjective version for optimizing yield and cost simultaneously.

    Parameters
    ----------

    Examples
    --------
    >>> bemul = BaumgartnerCrossCouplingDescriptorEmulator()

    Notes
    -----
    This benchmark is based on data from [Baumgartner]_ et al.

    References
    ----------

    .. [Baumgartner] L. M. Baumgartner et al., Org. Process Res. Dev., 2019, 23, 1594–1601
       DOI: `10.1021/acs.oprd.9b00236 <https://doi.org/10.1021/acs.oprd.9b00236>`_

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_domain = self._domain
        self.mod_domain = self._domain + ContinuousVariable(
            name="cost",
            description="cost in USD of 40 uL reaction",
            bounds=[0.0, 1.0],
            is_objective=True,
            maximize=False,
        )
        self._domain = self.mod_domain

    def _run(self, conditions, **kwargs):
        # Change to original domain for running predictive model
        self._domain = self.init_domain
        conditions, _ = super()._run(conditions=conditions, **kwargs)

        # Calculate costs
        costs = self._calculate_costs(conditions)
        conditions[("cost", "DATA")] = costs

        # Change back to modified domain
        self._domain = self.mod_domain
        return conditions, {}

    @classmethod
    def _calculate_costs(cls, conditions):
        catalyst = conditions["catalyst"].values
        base = conditions["base"].values
        base_equiv = conditions["base_equivalents"].values

        # Calculate amounts
        droplet_vol = 40 * 1e-3  # mL
        mmol_triflate = 0.91 * droplet_vol
        mmol_anniline = 1.6 * mmol_triflate
        catalyst_equiv = {
            "tBuXPhos": 0.0095,
            "tBuBrettPhos": 0.0094,
            "AlPhos": 0.0094,
        }
        mmol_catalyst = [catalyst_equiv[c] * mmol_triflate for c in catalyst]
        mmol_base = base_equiv * mmol_triflate

        # Calculate costs
        cost_triflate = mmol_triflate * 5.91  # triflate is $5.91/mmol
        cost_anniline = mmol_anniline * 0.01  # anniline is $0.01/mmol
        cost_catalyst = np.array(
            [cls._get_catalyst_cost(c, m) for c, m in zip(catalyst, mmol_catalyst)]
        )
        cost_base = np.array(
            [cls._get_base_cost(b, m) for b, m in zip(base, mmol_base)]
        )
        tot_cost = cost_triflate + cost_anniline + cost_catalyst + cost_base
        if len(tot_cost) == 1:
            tot_cost = tot_cost[0]
        return tot_cost

    @staticmethod
    def _get_catalyst_cost(catalyst, catalyst_mmol):
        catalyst_prices = {
            "tBuXPhos": 94.08,
            "tBuBrettPhos": 182.85,
            "AlPhos": 594.18,
        }
        return float(catalyst_prices[catalyst] * catalyst_mmol)

    @staticmethod
    def _get_base_cost(base, mmol_base):
        # prices in $/mmol
        base_prices = {
            "DBU": 0.03,
            "BTMG": 1.2,
            "TMG": 0.001,
            "TEA": 0.01,
        }
        return float(base_prices[base] * mmol_base)
