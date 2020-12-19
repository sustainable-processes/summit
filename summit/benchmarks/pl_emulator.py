import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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


class SummitDataModule(pl.LightningModule):
    def __init__(self, domain, dataset):
        self.domain = domain
        self.dataset = dataset


@torch.no_grad()
def create_parity_plot(regressor, X_test, y_test):
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

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = nn.BayesianLinear(input_dim, 512)
        self.blinear2 = nn.BayesianLinear(512, output_dim)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)

    def training_step(self, batch, batch_idx):
        datapoints, labels = batch

        loss = self.sample_elbo(
            inputs=datapoints, labels=labels, criterion=self.criterion, sample_nbr=3
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


def main():
    regressor = BayesianRegressor(13, 1)
    train_loader, X_test, y_test = get_data()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(regressor, train_loader)
    create_parity_plot(regressor, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()