import ipdb
from summit.utils.dataset import DataSet
from summit.domain import *
from summit.experiment import Experiment
from summit import get_summit_config_path
from summit.strategies import Transform

import torch
import pytorch_lightning as pl

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pathlib
import numpy as np
from numpy.random import default_rng


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


    Raises
    ------
    ValueError
        description

    Examples
    --------
    >>>
    Notes
    -----

    """

    def __init__(self, model_name, domain, dataset=None, **kwargs):
        super().__init__(domain, **kwargs)

        # Save locations
        self.model_name = model_name
        self.model_dir = kwargs.get(
            "model_dir", get_summit_config_path() / "ExprimentalEmulator"
        )
        self.model_dir = pathlib.Path(self.model_dir)
        self.checkpoint_path = (
            self.model_dir / self.model_name / f"{self.model_name}.ckpt"
        )

        # Create the datamoedule
        if dataset is not None:
            self.datamodule = EmulatorDataModule(self.domain, dataset, **kwargs)
            train_loader = self.datamodule.train_dataloader()
            self.n_examples = len(train_loader.dataset)
            self.n_features = train_loader.dataset[0][0].shape[0]
            self.n_targets = train_loader.dataset[0][1].shape[0]

        # Create the regressor
        Reg = kwargs.get("regressor", BNNRegressor)
        load_checkpoint = kwargs.get("load_checkpoint", False)
        if self.checkpoint_path.exists() and load_checkpoint:
            # Load checkpoint if it exists
            self.regressor = Reg.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path
            )
            print("Model Loaded from disk")
        elif self.datamodule is not None:
            # Create new regressor
            hparams = dict(n_examples=self.n_examples)
            self.regressor = Reg(self.n_features, self.n_targets, **hparams)
        elif self.datamodule is None:
            raise ValueError(
                "Dataset cannot be None when there is not pretrained model."
            )

    def _run(self, conditions, **kwargs):
        X = conditions[[v.name for v in self.domain.input_variables]]
        if self.datamodule.normalize:
            X = self.datamodule.input_scaler.transform(X)
        y = self.regressor(X)
        if self.datamodule.normalize:
            y = self.datamodule.output_scaler.inverse_transform(y)
        for i, v in enumerate(self.domain.output_variables):
            conditions[v.name] = y[:, i]
        return conditions

    def train(self, **kwargs):
        """Train the model on the dataset

        For kwargs, see pytorch-lightining documentation:
        https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#init
        """
        logger = kwargs.get("logger")
        # version = kwargs.get("version", 0)
        if logger is None:
            kwargs["logger"] = pl.loggers.TensorBoardLogger(
                name=self.model_name,
                save_dir=self.model_dir,
                version=version,
            )
        # kwargs["checkpoint_callback"] = kwargs.get("checkpoint_callback", False)

        # Use pytorch lightining for training and saving
        trainer = pl.Trainer(**kwargs)
        trainer.fit(model=self.regressor, datamodule=self.datamodule)
        trainer.save_checkpoint(self.checkpoint_path)

    def test(self, **kwargs):
        trainer = pl.Trainer(**kwargs)
        return trainer.test(model=self.regressor, datamodule=self.datamodule)

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

    def __init__(self, case=1, dataset=None, **kwargs):
        self.case = case
        model_name = f"reizman_suzuki_case_{case}"
        domain = self.setup_domain()
        kwargs.update(dict(domain=domain, model_name=model_name, dataset=dataset))
        super().__init__(**kwargs)

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
        experiment_params = {"case": self.case}
        return super().to_dict(**experiment_params)


class EmulatorDataModule(pl.LightningDataModule):
    """Convert Summit DataSet and Domain into a Pytorch Dataloader

    Parameters
    ----------
    domain : :class:`~summit.domain.Domain`
        Domain of the optimization
    dataset : :class:`~summit.dataset.Dataset`
        Dataset used for training/validation
    normalize : bool, optional
        Normalize continuous input and output variables. Default is True.
    test_size : float, optional
        Fraction of data used for test. Default is 0.25
    random_state : float, optional
        A random initialization value. Use to make results more reproducible.

    Returns
    -------
    result: `bool`
        description

    Raises
    ------
    ValueError
        description

    Examples
    --------


    Notes
    -----


    """

    def __init__(self, domain: Domain, dataset: DataSet, **kwargs):
        super().__init__()
        self.domain = domain
        self.ds = dataset
        self.normalize = kwargs.get("normalize", True)
        self.test_size = kwargs.get("test_size", 0.25)
        self.shuffle = kwargs.get("shuffle", True)
        self.batch_size = kwargs.get("train_batch_size", 4)
        self.random_state = kwargs.get("random_state")
        self.transform = kwargs.get("transform", Transform(self.domain))
        self.categorical_method = kwargs.get("categorical_method", "one-hot")

        # Run initial setup
        self.initial_setup()

    def initial_setup(self):
        # Get data
        X, y = self.transform.transform_inputs_outputs(
            self.ds, categorical_method=self.categorical_method
        )

        # Scaling
        self.input_scaler, self.output_scaler = self._create_scalers(X, y)
        X = self.input_scaler.transform(X)
        y = self.output_scaler.transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Convert to tensors
        self.X_train = torch.tensor(X_train).float()
        self.y_train = torch.tensor(y_train).float()
        self.X_test = torch.tensor(X_test).float()
        self.y_test = torch.tensor(y_test).float()

    def train_dataloader(self):
        ds_train = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        return torch.utils.data.DataLoader(
            ds_train, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def test_dataloader(self):
        ds_test = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        return torch.utils.data.DataLoader(ds_test)

    @classmethod
    def from_csv(cls, csv_file, domain, ds, model_dir, **kwargs):
        """Create a Summit Data Module from a csv file"""
        ds = DataSet.read_csv(csv_file)
        return cls(domain, ds, model_dir, **kwargs)

    @staticmethod
    def _create_scalers(X, y):
        input_scaler = StandardScaler().fit(X)
        output_scaler = StandardScaler().fit(y)
        return input_scaler, output_scaler


@variational_estimator
class BNNRegressor(pl.LightningModule):
    """A Bayesian Neural Network pytorch lightining module"""

    val_str = "CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}"

    def __init__(
        self, input_dim, output_dim, n_examples=100, hidden_units=512, **kwargs
    ):
        super().__init__()

        self.blinear1 = BayesianLinear(input_dim, hidden_units)
        self.blinear4 = BayesianLinear(hidden_units, output_dim)
        self.n_examples = n_examples
        self.n_samples = kwargs.get("n_samples", 50)
        self.save_hyperparameters("input_dim", "output_dim", "n_examples")
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x_ = torch.nn.functional.relu(self.blinear1(x))
        x_ = self.blinear4(x_)
        return x_

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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer


class ANNRegressor(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_units=512, **kwargs):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, output_dim)
        self.save_hyperparameters("input_dim", "output_dim")
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, **kwargs):
        x_ = self.linear1(x)
        x_ = torch.nn.functional.relu(x_)
        return self.linear2(x_)

    def training_step(self, batch, batch_idx, **kwargs):
        return self.calculate_loss(batch, "train")

    def validation_step(self, batch, batch_idx, **kwargs):
        return self.calculate_loss(batch, "validation")

    def test_step(self, batch, batch_idx, **kwargs):
        return self.calculate_loss(batch, "test")

    def calculate_loss(self, batch, step, **kwargs):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log(f"{step}_mse", loss)
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