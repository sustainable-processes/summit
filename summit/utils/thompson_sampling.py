from summit.utils.dataset import DataSet
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
)
import pyrff
import torch
import numpy as np
from summit import get_summit_config_path
import logging
import os


class ThompsonSampledModel:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.input_columns_ordered = None
        self.output_columns_ordered = None
        self.logger = logging.getLogger(__name__)

    def fit(self, X: DataSet, y: DataSet, **kwargs):
        """Train model and take spectral samples"""

        self.input_columns_ordered = [col[0] for col in X.columns]

        # Convert to tensors
        X_np = X.to_numpy().astype(float)
        y_np = y.to_numpy().astype(float)
        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)

        # Train the GP model
        self.model = SingleTaskGP(X, y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

        # self.logger.info model hyperparameters
        if self.model_name is None:
            self.model_name = self.output_columns_ordered[0]
        self.lengthscales_ = self.model.covar_module.base_kernel.lengthscale.detach()[
            0
        ].numpy()
        self.outputscale_ = self.model.covar_module.outputscale.detach().numpy()
        self.noise_ = self.model.likelihood.noise_covar.noise.detach().numpy()[0]
        self.logger.debug(f"Model {self.model_name} lengthscales: {self.lengthscales_}")
        self.logger.debug(f"Model {self.model_name} variance: {self.outputscale_}")
        self.logger.debug(f"Model {self.model_name} noise: {self.noise_}")

        # Spectral sampling
        n_spectral_points = kwargs.get("n_spectral_points", 1500)
        n_retries = kwargs.get("n_retries", 10)
        self.logger.debug(
            f"Spectral sampling {self.model_name} with {n_spectral_points} spectral points."
        )
        self.rff = None
        nu = self.model.covar_module.base_kernel.nu
        for _ in range(n_retries):
            try:
                self.rff = pyrff.sample_rff(
                    lengthscales=self.lengthscales_,
                    scaling=np.sqrt(self.outputscale_),
                    noise=self.noise_,
                    kernel_nu=nu,
                    X=X_np,
                    Y=y_np[:, 0],
                    M=n_spectral_points,
                )
                break
            except np.linalg.LinAlgError as e:
                self.logger.error(e)
            except ValueError as e:
                self.logger.error(e)
        if self.rff is None:
            raise RuntimeError(f"Spectral sampling failed after {n_retries} retries.")

        return dict(
            name=self.model_name,
            rff=self.rff,
            lengthscales=self.lengthscales_,
            outputscale=self.outputscale_,
            noise=self.noise_,
        )

    def predict(self, X: DataSet, **kwargs):
        """Predict the values of a"""
        X = X[self.input_columns_ordered].to_numpy()
        return self.rff(X)

    def save(self, filepath=None):
        if filepath is None:
            filepath = get_summit_config_path() / "tsemo" / str(self.uuid_val)
            os.makedirs(filepath, exist_ok=True)
            filepath = filepath / "models.h5"
        pyrff.save_rffs([self.rff], filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = get_summit_config_path() / "tsemo" / str(self.uuid_val)
            os.makedirs(filepath, exist_ok=True)
            filepath = filepath / "models.h5"
        self.rff = pyrff.load_rffs(filepath)[0]
