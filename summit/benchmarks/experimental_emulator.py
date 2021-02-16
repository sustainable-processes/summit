from summit.utils.dataset import DataSet
from summit.domain import *
from summit.experiment import Experiment
from summit import get_summit_config_path

import torch
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.utils import to_device

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score

import pathlib
import numpy as np
import pandas as pd
from copy import deepcopy


class ExperimentalEmulator(Experiment):
    """Experimental Emulator

    Train a machine learning model based on experimental data.
    The model acts a benchmark for testing optimisation strategies.

    Parameters
    ----------
    model_name : str
        Model name used for identification. Must be unique from other models stored on the system.
    domain : :class:`~summit.domain.Domain`
        The domain of the emulator
    dataset : :class:`~summit.dataset.Dataset`
        Dataset used for training/validation
    regressor : :classs:`pl.LightningModule`, optional
        Pytorch LightningModule class. Defaults to the BayesianRegressor
    model_dir : :class:`pathlib.Path` or str, optional
        Directory where models are saved. Defaults to `~/.summit/ExperimentEmulator"
    load_checkpoint : bool, optional
        Whether to load any previously trained models on disk. By default previous models are not loaded.
    normalize : bool, optional
        Normalize continuous input variables. Default is True.
    test_size : float, optional
        Fraction of data used for test. Default is 0.25
    random_state : float, optional
        A random initialization value. Use to make results more reproducible.

    """

    def __init__(self, model_name, domain, **kwargs):
        super().__init__(domain, **kwargs)

        # Metadata
        self.model_name = model_name
        self.model_dir = kwargs.get(
            "model_dir", get_summit_config_path() / "ExperimentalEmulator"
        )
        self.model_dir = pathlib.Path(self.model_dir)
        self.checkpoint_path = (
            self.model_dir / self.model_name / f"{self.model_name}.ckpt"
        )

        # Training related parameters
        self.standardize_inputs = kwargs.get("standardize_inputs", True)
        self.standardize_outputs = kwargs.get("standardize_outputs", True)
        load_checkpoint = kwargs.get("load_checkpoint", False)

        # Create the regressor
        self.regressor = kwargs.get("regressor", BNNRegressor)
        #  Check if the regressor exists
        # if self.checkpoint_path.exists() and load_checkpoint:
        #     # TODO: Put skorch model loading code
        #     self.logger.info(f"{model_name} loaded from disk.")
        # # If it doesn't, log at INFO level that the regressor needs to be trained before being used.
        # else:
        #     self.logger.info("The regressor must be trained before use.")

    def _run(self, conditions, **kwargs):

        if self.datamodule.normalize:
            y = self.datamodule.output_scaler.inverse_transform(y)
        for i, v in enumerate(self.domain.output_variables):
            conditions[v.name] = y[:, i]
        return conditions

    def train(self, dataset, **kwargs):
        """Train the model on the dataset

        Parameters
        ----------
        standardize_inputs : bool, optional
            Standardize all input continuous variables. Default is True.
        standardize_outputs : bool, optional
            Standardize all output continuous variables. Default is True.
        output_variables : str or list, optional
            The variables that should be trained by the predictor.
            Defaults to all objectives in the domain.
        test_size : float, optional
            The size of the test as a fraction of the total dataset. Defaults to 0.1.
        cv_folds : int, optional
            The number of cross validation folds. Defaults to 5.
        max_epochs : int, optional
            The max number of epochs for each CV fold. Defaults to 100.
        scoring : str or list, optional
            A list of scoring functions or names of them. Defaults to R2 and MSE.
            See here for more https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        regressor_kwargs : dict, optional
            You can pass extra arguments to the regressor here.
        callbacks : None, "disable" or list of Callbacks
            Skorch callbacks passed to skorch.net. See: https://skorch.readthedocs.io/en/latest/net.html

        Returns
        -------
        A dictionary containing the results of the training.
        """
        # Preprocessors
        output_variables = kwargs.get(
            "output_variables", [v.name for v in self.domain.output_variables]
        )
        X_preprocessor = self._create_input_preprocessor()
        y_preprocessor = self._create_output_preprocessor(output_variables)

        # Create network
        output_variables = kwargs.get(
            "output_variables", [v.name for v in self.domain.output_variables]
        )
        regressor_kwargs = kwargs.get("regressor_kwargs", {})
        regressor_kwargs.update(
            dict(
                input_dim=11,
                output_dim=len(output_variables),
                n_examples=dataset.shape[0],
            )
        )
        net = EmulatorNet(
            self.regressor,
            regressor_kwargs=regressor_kwargs,
            train_split=None,
            max_epochs=kwargs.get("max_epochs", 100),
            callbacks=kwargs.get("callbacks"),
        )

        # Create predictor
        # TODO: also create an inverse function
        ds_to_tensor = FunctionTransformer(dataset_to_tensor)
        pipe = Pipeline(
            steps=[
                ("preprocessor", X_preprocessor),
                ("dst", ds_to_tensor),
                ("net", net),
            ]
        )
        self.predictor = TransformedTargetRegressor(pipe, transformer=StandardScaler())

        # Get data
        input_columns = [v.name for v in self.domain.input_variables]
        X = dataset[input_columns].to_numpy()
        y = dataset[output_variables].to_numpy().astype(float)
        # Sklearn columntransformer expects a pandas dataframe not a dataset
        X = pd.DataFrame(X, columns=input_columns)

        # Train-test split
        test_size = kwargs.get("test_size", 0.1)
        random_state = kwargs.get("random_state")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        y_train, y_test = torch.tensor(y_train).float(), torch.tensor(y_test).float()

        # Run training
        scoring = kwargs.get("scoring", ["r2", "neg_root_mean_squared_error"])
        folds = kwargs.get("cv_folds", 5)
        return cross_validate(
            self.predictor, X_train, y_train, scoring=scoring, cv=folds
        )

    def caclulate_input_dimensions(self):
        pass

    def _create_input_preprocessor(self):
        """Create feature preprocessors """
        transformers = []
        # Numeric transforms
        numeric_features = [
            v.name
            for v in self.domain.input_variables
            if v.variable_type == "continuous"
        ]
        if len(numeric_features) > 0:
            transformers.append(("num", StandardScaler(), numeric_features))

        # Categorical transforms
        categorical_features = [
            v.name
            for v in self.domain.input_variables
            if v.variable_type == "categorical"
        ]
        if len(categorical_features) > 0:
            transformers.append(("cat", OneHotEncoder(), categorical_features))

        # Create preprocessor
        if len(numeric_features) == 0 and len(categorical_features) > 0:
            raise DomainError(
                "With only categorical features, you can do a simple lookup."
            )
        elif len(numeric_features) > 0 or len(categorical_features) > 0:
            preprocessor = ColumnTransformer(transformers=transformers)
        else:
            raise DomainError(
                "No continuous or categorical features were found in the dataset."
            )
        return preprocessor

    def _create_output_preprocessor(self, output_variables):
        """"Create target preprocessors"""
        transformers = [
            ("scale", StandardScaler(), output_variables),
            ("dst", FunctionTransformer(dataset_to_tensor), output_variables),
        ]
        return ColumnTransformer(transformers=transformers)

    def to_dict(self, **kwargs):
        kwargs.update(
            dict(
                dataset=self.datamodule.ds,
                model_name=self.model_name,
                model_dir=self.model_dir,
                regressor_name=self.regressor.__class__.__name__,
                n_examples=self.n_examples,
            )
        )
        return super().to_dict(**kwargs)

    @classmethod
    def from_dict(cls, d):
        regressor = registry[d["experiment_params"]["regressor_name"]]
        d["experiment_params"]["regressor"] = regressor
        return super().from_dict(d)

    def parity_plot(self):
        """ Produce a parity plot based on the test data"""
        import matplotlib.pyplot as plt

        X_test, y_test = self.datamodule.X_test, self.datamodule.y_test
        with torch.no_grad():
            Y_test_pred = self.regressor(X_test)
        fig, ax = plt.subplots(1)
        ax.scatter(y_test[:, 0], Y_test_pred[:, 0])
        # Parity line
        min = np.min(np.concatenate([y_test[:, 0], Y_test_pred[:, 0]]))
        max = np.max(np.concatenate([y_test[:, 0], Y_test_pred[:, 0]]))
        ax.plot([min, max], [min, max])
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        return fig, ax


class EmulatorNet(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        self.regressor_kwargs = kwargs.pop("regressor_kwargs", {})
        super().__init__(*args, **kwargs)

    def initialize_module(self):
        kwargs = self.regressor_kwargs
        module = self.module
        is_initialized = isinstance(module, torch.nn.Module)
        if kwargs or not is_initialized:
            if is_initialized:
                module = type(module)

            if (is_initialized or self.initialized_) and self.verbose:
                msg = self._format_reinit_msg("module", kwargs)
                print(msg)

            module = module(**kwargs)
        self.module_ = to_device(module, self.device)
        return self


def dataset_to_tensor(X):
    """Convert datasets into """
    return torch.tensor(X).float()


@variational_estimator
class BNNRegressor(torch.nn.Module):
    """A Bayesian Neural Network pytorch lightining module"""

    val_str = "CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}"

    def __init__(
        self, input_dim, output_dim, n_examples=100, hidden_units=512, **kwargs
    ):
        super().__init__()
        # self.n_layers = kwargs.get("n_hidden", 3)
        # self.layers = [BayesianLinear(input_dim, hidden_units)]
        # self.layers += [
        #     BayesianLinear(hidden_units, hidden_units) for _ in range(self.n_layers)
        # ]
        # self.layers.append(BayesianLinear(hidden_units, output_dim))
        self.blinear1 = BayesianLinear(input_dim, hidden_units)
        self.blinear2 = BayesianLinear(hidden_units, output_dim)
        self.n_examples = n_examples
        self.n_samples = kwargs.get("n_samples", 50)
        self.save_hyperparameters("input_dim", "output_dim", "n_examples")
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        # for layer in self.layers[:-1]:
        #     x = layer(x)
        #     x = F.relu(x)
        # return self.layers[-1](x)
        x = self.blinear1(x)
        x = F.relu(x)
        return self.blinear2(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.sample_elbo(
            inputs=X,
            labels=y,
            criterion=self.criterion,
            sample_nbr=3,
            complexity_cost_weight=1 / self.n_examples,
        )
        self.log("train_mse", loss)
        return loss

    def test_step(self, batch, batch_idx, **kwargs):
        X, y = batch
        y_hats = torch.stack([self(X) for _ in range(self.n_samples)])
        mean_y_hat = y_hats.mean(axis=0)
        std_y_hat = y_hats.std(axis=0)
        loss = self.criterion(mean_y_hat, y)
        self.log("test_mse", loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     ic_acc, over_ci_lower, under_ci_upper = self.evaluate_regression(batch)

    #     self.log("val_loss", ic_acc)
    #     self.log("under_ci_upper", under_ci_upper)
    #     self.log("over_ci_lower", over_ci_lower)

    #     return ic_acc

    def evaluate_regression(self, batch, samples=100, std_multiplier=1.96):
        """Evaluate Bayesian Neural Network

        This answers the question "How many correction predictions
        are in the confidence interval (CI)?" It also spits out the CI.

        Parameters
        ----------
        batch : tuple
            The batch being evaluatd
        samples : int, optional
            The number of samples of the BNN for calculating the CI
        std_multiplier : float, optional
            The Z-score corresponding with the desired CI. Default is
            1.96, which corresponds with a 95% CI.

        Returns
        -------
        tuple of ic_acc, over_ci_lower, under_ci_upper

        icc_acc is the percentage within the CI.

        """

        X, y = batch

        # Sample
        preds = torch.tensor([self(X) for i in range(samples)])
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)

        # Calculate CI
        ci_upper, ci_lower = self._calc_ci(means, stds, std_multiplier)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()

        under_ci_upper = (ci_upper >= y).float().mean()
        over_ci_lower = (ci_lower <= y).float().mean()

        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()

        return ic_acc, over_ci_lower, under_ci_upper

    def _calc_ci(self, means, stds, std_multiplier=1.96):
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        return ci_lower, ci_upper

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer


class ANNRegressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=512, **kwargs):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, output_dim)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, **kwargs):
        x_ = self.linear1(x)
        x_ = F.relu(x_)
        return self.linear2(x_)

    def training_step(self, batch, batch_idx, **kwargs):
        X, y = batch
        y_hat = self(X)
        return self.evaluate_loss(y, y_hat, "train")

    def validation_step(self, batch, batch_idx, **kwargs):
        return self.calculate_loss(batch, "validation")

    def test_step(self, batch, batch_idx, **kwargs):
        return self.calculate_loss(batch, "test")

    def calculate_loss(self, batch, step, **kwargs):
        X, y = batch
        y_hat = self(X)
        return self.evaluate_loss(y, y_hat, "val")

    def test_step(self, batch, batch_idx, **kwargs):
        X, y = batch
        y_hat = self(X)
        return self.evaluate_loss(y, y_hat, "test")

    def evaluate_loss(self, y_true, y_hat, step):
        loss = F.mse_loss(y_hat, y_true)
        self.log(f"_mse", loss)
        # r2 = r2_score(y_true.detach().numpy(), y_hat.detach().numpy())
        # self.log(f"{step}_r2_score", r2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer


class RegressorRegistry:
    """Registry for Regressors

    Models registered using the register method
    are saved as the class name.

    """

    regressors = {}

    def __getitem__(self, key):
        reg = self.regressors.get(key)
        if reg is not None:
            return reg
        else:
            raise KeyError(
                f"{key} is not in the registry. Register using the register method."
            )

    def __setitem__(self, key, value):
        reg = self.regressors.get(key)
        if reg is not None:
            self.regressors[key] = value

    def register(self, regressor):
        key = regressor.__name__
        self.regressors[key] = regressor


# Create global registry
registry = RegressorRegistry()


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
        super().__init__(domain=domain, model_name=model_name)

    @staticmethod
    def setup_domain():
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
