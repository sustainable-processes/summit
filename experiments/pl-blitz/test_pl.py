import ipdb
from summit.utils.dataset import DataSet
from summit.domain import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
from numpy.random import default_rng

import pytorch_lightning as pl

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# TODO:
# Figure out how to create a dataloader from summit data
# Set location for logs to a reasonable place


def get_data(test_size=0.25, random_state=42):
    # Load data
    X, y = load_boston(return_X_y=True)

    # Normalize
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(np.expand_dims(y, -1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convert to tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # Create dataloaders
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=16, shuffle=True
    )

    return dataloader_train, X_test, y_test


class EmulatorDataModule(pl.LightningDataModule):
    def __init__(self, domain: Domain, ds: DataSet, model_dir: str, **kwargs):
        super().__init__()
        self.domain = domain
        self.ds = ds
        self.model_dir = model_dir
        self.normalize = kwargs.get("normalize", True)
        self.test_size = kwargs.get("test_size", 0.25)
        self.random_state = kwargs.get("random_state")
        self.batch_size = kwargs.get("train_batch_size", 4)
        self.shuffle = kwargs.get("shuffle", False)

        # Run initial setup
        self.initial_setup()

    def initial_setup(self):
        # Get data
        X, y = self.split_data(self.domain, self.ds)

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
    def split_data(domain, ds):
        X = ds[[v.name for v in domain.input_variables]].to_numpy()
        y = ds[[v.name for v in domain.output_variables]].to_numpy()
        return X, y

    @staticmethod
    def _create_scalers(X, y):
        input_scaler = StandardScaler().fit(X)
        output_scaler = StandardScaler().fit(y)
        return input_scaler, output_scaler


def create_parity_plot(regressor, datamodule):
    X_test, y_test = datamodule.X_test, datamodule.y_test
    with torch.no_grad():
        Y_test_pred = regressor(X_test)
    fig, ax = plt.subplots(1)
    ax.scatter(Y_test_pred[:, 0], y_test[:, 0])
    ax.plot([-4, 4], [-4, 4])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig, ax


@variational_estimator
class BayesianRegressor(pl.LightningModule):
    """A Bayesian Neural Network pytorch lightining module"""

    val_str = "CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}"

    def __init__(self, input_dim, output_dim, train_len, hidden_units=512):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, hidden_units)
        self.blinear4 = BayesianLinear(hidden_units, output_dim)

        self.criterion = torch.nn.MSELoss()
        self.train_len = train_len

    def forward(self, x):
        x_ = F.relu(self.blinear1(x))
        x_ = self.blinear4(x_)
        return x_

    def training_step(self, batch, batch_idx):
        X, y = batch

        loss = self.sample_elbo(
            inputs=X,
            labels=y,
            criterion=self.criterion,
            sample_nbr=3,
            complexity_cost_weight=1 / self.train_len,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ic_acc, over_ci_lower, under_ci_upper = self.evaluate_regression(batch)

        self.log("ic_acc", ic_acc)
        self.log("under_ci_upper", under_ci_upper)
        self.log("over_ci_lower", over_ci_lower)

        return ic_acc

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
        preds = [self(X) for i in range(samples)]
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
        optimizer = optim.SGD(self.parameters(), lr=0.001)
        return optimizer

    def test_step(self, batch, batch_idx):
        pass


def create_domain():
    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[30, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flowrate of reactant a", bounds=[1, 100]
    )

    domain += ContinuousVariable(
        name="flowrate_b", description="flowrate of reactant b", bounds=[1, 100]
    )

    domain += ContinuousVariable(
        name="yield",
        description="yield of reaction",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    return domain


def create_dataset(domain, n_samples=100, random_seed=100):
    rng = default_rng(random_seed)
    n_features = len(domain.input_variables)
    inputs = rng.standard_normal(size=(n_samples, n_features))
    inputs *= [-5, 6, 0.1]
    output = np.sum(inputs ** 2, axis=1)
    data = np.append(inputs, np.atleast_2d(output).T, axis=1)
    columns = [v.name for v in domain.input_variables] + [
        domain.output_variables[0].name
    ]
    return DataSet(data, columns=columns)


def main():
    # Get data
    domain = create_domain()
    dataset = create_dataset(domain, n_samples=500)
    summit_dm = EmulatorDataModule(
        domain, dataset, model_dir="", random_state=42, batch_size=16, shuffle=True
    )

    train_loader = summit_dm.train_dataloader()
    n_examples = len(train_loader.dataset)
    n_features = train_loader.dataset[0][0].shape[0]
    n_targets = train_loader.dataset[0][1].shape[0]
    regressor = BayesianRegressor(n_features, n_targets, n_examples)

    fig, axes = plt.subplots(3)
    X = summit_dm.X_train
    for i in range(3):
        axes[i].scatter(X[:, i], summit_dm.y_train)
    plt.show()

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(regressor, summit_dm)
    create_parity_plot(regressor, summit_dm)
    plt.show()


if __name__ == "__main__":
    main()