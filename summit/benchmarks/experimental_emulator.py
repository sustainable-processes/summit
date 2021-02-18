from summit.utils.dataset import DataSet
from summit.domain import *
from summit.experiment import Experiment
from summit import get_summit_config_path
from summit.utils import jsonify_dict, unjsonify_dict

import torch
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.utils import to_device

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    ParameterGrid,
)
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid
from sklearn.base import BaseEstimator, RegressorMixin, is_classifier, clone
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score,
    _score,
    _aggregate_score_dicts,
)
from sklearn.metrics import r2_score
from sklearn.utils.validation import (
    _deprecate_positional_args,
    indexable,
    check_is_fitted,
    _check_fit_params,
)
from sklearn.utils import check_array, _safe_indexing
from sklearn.utils.fixes import delayed
from sklearn.metrics._scorer import _check_multimetric_scoring

from tqdm.auto import tqdm
from joblib import Parallel
import pathlib
import numpy as np
from numpy.random import default_rng
import pandas as pd
from copy import deepcopy
from itertools import product
from collections import defaultdict
from copy import deepcopy
import pkg_resources
import time
import json
import types
import warnings

__all__ = [
    "ExperimentalEmulator",
    "ANNRegressor",
    "get_bnn",
    "RegressorRegistry",
    "registry",
    "get_pretrained_reizman_suzuki_emulator",
    "get_pretrained_baumgartner_cc_emulator",
    "ReizmanSuzukiEmulator",
    "BaumgartnerCrossCouplingEmulator",
]


class ExperimentalEmulator(Experiment):
    """Experimental Emulator

    Train a machine learning model based on experimental data.
    The model acts a benchmark for testing optimisation strategies.

    Parameters
    ----------
    model_name : str
        Name of the model, ideally with no spaces
    domain : :class:`~summit.domain.Domain`
        The domain of the emulator
    dataset : :class:`~summit.dataset.Dataset`, optional
        Dataset used for training/validation
    regressor : :classs:`torch.nn.Module`, optional
        Pytorch LightningModule class. Defaults to the ANNRegressor
    output_variable_names : str or list, optional
        The names of the variables that should be trained by the predictor.
        Defaults to all objectives in the domain.
    clip : bool or list
        Whether to clip predictions to the limits of
        the objectives in the domain. True (default) means
        clipping is activated for all outputs and False means
        it is not activated at all. A list of specific outputs to clip
        can also be passed.

    """

    def __init__(self, model_name, domain, **kwargs):
        super().__init__(domain, **kwargs)
        self.model_name = model_name

        # Data
        self.ds = kwargs.get("dataset")
        if self.ds is not None:
            self.n_features = self._caclulate_input_dimensions(self.domain)
            self.n_examples = self.ds.shape[0]

        self.output_variable_names = kwargs.get(
            "output_variable_names",
            [v.name for v in self.domain.output_variables],
        )

        # Create the regressor
        self.regressor = kwargs.get("regressor", ANNRegressor)
        self.predictors = kwargs.get("predictors")
        self.clip = kwargs.get("clip", True)

    def _run(self, conditions, **kwargs):
        input_columns = [v.name for v in self.domain.input_variables]
        X = conditions[input_columns].to_numpy()
        if X.shape[0] == len(input_columns):
            X = X[np.newaxis, :]
        X = pd.DataFrame(X, columns=input_columns)
        y_pred, y_pred_std = self._predict(X)
        return_std = kwargs.get("return_std", False)
        for i, name in enumerate(self.output_variable_names):
            if type(conditions) == pd.Series:
                y = y_pred[0, i]
                y_std = y_pred_std[0, i]
            else:
                y = y_pred[:, i]
                y_std = y_pred_std[:, i]
            conditions.at[(name, "DATA")] = y
            if return_std:
                conditions.at[(f"{name}_std", "METADATA")] = y_std
        return conditions, {}

    def _predict(self, X, **kwargs):
        """Get a prediction

        Parameters
        ----------
        X : pd.DataFrame
            A pandas dataframe with inputs to the predictor

        Returns
        -------
        mean, std
        Numpy arrays with the average and standard deviation of the ensemble

        """
        y_pred = np.array(
            [estimator.predict(X, **kwargs) for estimator in self.predictors]
        )
        if self.clip:
            for i, v in enumerate(self.domain.output_variables):
                if type(self.clip) == list:
                    if v.name not in self.clip:
                        continue
                y_pred[:, :, i] = np.clip(y_pred[:, :, i], v.lower_bound, v.upper_bound)

        return y_pred.mean(axis=0), y_pred.std(axis=0)

    def train(self, **kwargs):
        """Train the model on the dataset

        Parameters
        ---------
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
        verbose : int
            0 for no logging, 1 for logging

        Notes
        ------
        If predictor was set in the initialization, it will not be overwritten.

        Returns
        -------
        A dictionary containing the results of the training.
        """
        if self.ds is None:
            raise ValueError("Dataset is required for training.")

        # Create predictor
        predictor = self._create_predictor(
            self.regressor,
            self.domain,
            self.n_features,
            self.n_examples,
            output_variable_names=self.output_variable_names,
            **kwargs,
        )

        # Get data
        input_columns = [v.name for v in self.domain.input_variables]
        X = self.ds[input_columns].to_numpy()
        y = self.ds[self.output_variable_names].to_numpy().astype(float)
        # Sklearn columntransformer expects a pandas dataframe not a dataset
        X = pd.DataFrame(X, columns=input_columns)

        # Train-test split
        test_size = kwargs.get("test_size", 0.1)
        random_state = kwargs.get("random_state")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        y_train, y_test = (
            torch.tensor(self.y_train).float(),
            torch.tensor(self.y_test).float(),
        )

        # Training
        scoring = kwargs.get("scoring", ["r2", "neg_root_mean_squared_error"])
        folds = kwargs.get("cv_folds", 5)
        search_params = kwargs.get("search_params", {})
        # Run grid search if requested
        if search_params:
            self.logger.info("Starting grid search.")
            gs = ProgressGridSearchCV(
                predictor, search_params, refit="r2", cv=folds, scoring=scoring
            )
            gs.fit(self.X_train, y_train)
            best_params = gs.best_params_
            params = {}
            for param in search_params.keys():
                params[param] = best_params[param]
            predictor.set_params(**params)

        # Run final training using cross validation
        initializing = kwargs.get("initializing", False)

        if not initializing:
            self.logger.info("Starting training.")
        res = cross_validate(
            predictor,
            self.X_train,
            y_train,
            scoring=scoring,
            cv=folds,
            return_estimator=True,
        )

        self.predictors = res.pop("estimator")
        # Rename from test to validation
        for name in scoring:
            scores = res.pop(f"test_{name}")
            res[f"val_{name}"] = scores
        return res

    def test(self, **kwargs):
        scoring = kwargs.get("scoring", ["r2", "neg_root_mean_squared_error"])
        scores_list = []
        for predictor in self.predictors:
            if callable(scoring):
                scorers = scoring
            elif scoring is None or isinstance(scoring, str):
                scorers = check_scoring(predictor, scoring)
            else:
                scorers = _check_multimetric_scoring(predictor, scoring)
            scores_list.append(_score(predictor, self.X_test, self.y_test, scorers))
        scores_dict = _aggregate_score_dicts(scores_list)
        for name in scoring:
            scores = scores_dict.pop(name)
            scores_dict[f"test_{name}"] = scores
        return scores_dict

    @classmethod
    def _create_predictor(
        cls,
        regressor,
        domain,
        input_dimensions,
        num_examples,
        output_variable_names,
        **kwargs,
    ):
        # Preprocessors
        output_variable_names = kwargs.get(
            "output_variable_names", [v.name for v in domain.output_variables]
        )
        X_preprocessor = cls._create_input_preprocessor(domain)
        y_preprocessor = cls._create_output_preprocessor(output_variable_names)

        # Create network
        regressor_kwargs = kwargs.get("regressor_kwargs", {})
        regressor_kwargs.update(
            dict(
                module__input_dim=input_dimensions,
                module__output_dim=len(output_variable_names),
                module__n_examples=num_examples,
            )
        )
        verbose = kwargs.get("verbose", 0)
        net = NeuralNetRegressor(
            regressor,
            train_split=None,
            max_epochs=kwargs.get("max_epochs", 100),
            callbacks=kwargs.get("callbacks"),
            verbose=verbose,
            **regressor_kwargs,
        )

        # Create predictor
        # TODO: also create an inverse function
        ds_to_tensor = FunctionTransformer(numpy_to_tensor, check_inverse=False)
        pipe = Pipeline(
            steps=[
                ("preprocessor", X_preprocessor),
                ("dst", ds_to_tensor),
                ("net", net),
            ]
        )

        # output_pipeline = Pipeline(
        #     steps=[("scaler", StandardScaler()), ("dst", ds_to_tensor)]
        # )

        return UpdatedTransformedTargetRegressor(
            regressor=pipe, transformer=StandardScaler(), check_inverse=False
        )

    @staticmethod
    def _caclulate_input_dimensions(domain):
        num_dimensions = 0
        for v in domain.input_variables:
            if v.variable_type == "continuous":
                num_dimensions += 1
            elif v.variable_type == "categorical":
                num_dimensions += len(v.levels)
        return num_dimensions

    @staticmethod
    def _create_input_preprocessor(domain):
        """Create feature preprocessors """
        transformers = []
        # Numeric transforms
        numeric_features = [
            v.name for v in domain.input_variables if v.variable_type == "continuous"
        ]
        if len(numeric_features) > 0:
            transformers.append(("num", StandardScaler(), numeric_features))

        # Categorical transforms
        categorical_features = [
            v.name for v in domain.input_variables if v.variable_type == "categorical"
        ]
        categories = [
            v.levels for v in domain.input_variables if v.variable_type == "categorical"
        ]
        if len(categorical_features) > 0:
            transformers.append(
                ("cat", OneHotEncoder(categories=categories), categorical_features)
            )

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

    @staticmethod
    def _create_output_preprocessor(output_variable_names):
        """"Create target preprocessors"""
        transformers = [
            ("scale", StandardScaler(), output_variable_names),
            ("dst", FunctionTransformer(numpy_to_tensor), output_variable_names),
        ]
        return ColumnTransformer(transformers=transformers)

    def to_dict(self, **experiment_params):
        """Convert emulator parameters to dictionary

        Notes
        ------
        This does not save the weights and biases of the regressor.
        You need to use save_regressor method.

        """
        # Predictors
        predictors = [
            self._create_predictor_dict(predictor) for predictor in self.predictors
        ]

        # Update experiment_params
        experiment_params.update(
            {
                "model_name": self.model_name,
                "regressor_name": str(self.regressor.__name__),
                "n_features": self.n_features,
                "n_examples": self.n_examples,
                "output_variable_names": self.output_variable_names,
                "predictors": predictors,
            }
        )
        return super().to_dict(**experiment_params)

    @staticmethod
    def _create_predictor_dict(predictor):
        num = predictor.regressor_.named_steps.preprocessor.named_transformers_.num
        cat = predictor.regressor_.named_steps.preprocessor.named_transformers_.cat
        input_preprocessor = {
            # Numerical
            "num": {
                "mean_": num.mean_,
                "var_": num.var_,
                "scale_": num.scale_,
                "n_samples_seen_": num.n_samples_seen_,
            }
            # Categorical is automatic from the domain
        }
        out = predictor.transformer_
        output_preprocessor = {
            "mean_": out.mean_,
            "var_": out.var_,
            "scale_": out.scale_,
            "n_samples_seen_": out.n_samples_seen_,
        }
        return jsonify_dict(
            {
                "input_preprocessor": input_preprocessor,
                "output_preprocessor": output_preprocessor,
            }
        )

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Create ExperimentalEmulator from a dictionary

        Notes
        -----
        This does not load the regressor weights and biases.
        After calling from_dict, call load_regressor to load the
        weights and biases.

        """
        params = d["experiment_params"]
        domain = Domain.from_dict(d["domain"])

        # Load regressor
        regressor = registry[params["regressor_name"]]
        d["experiment_params"]["regressor"] = regressor

        # Load predictors
        predictors_params = params["predictors"]
        predictors = [
            cls._create_predictor(
                regressor,
                domain,
                params["n_features"],
                params["n_examples"],
                output_variable_names=params["output_variable_names"],
            )
            for predictor_params in predictors_params
        ]
        d["experiment_params"]["predictor"] = predictors

        # Dataset
        dataset = kwargs.get("dataset")
        d["experiment_params"]["dataset"] = dataset

        # Instantiate the class
        exp = super().from_dict(d)

        # Set runtime parameters
        exp.n_features = params["n_features"]
        exp.n_examples = params["n_examples"]

        # One round of training to initialize all variables
        if exp.ds is None:
            exp.ds = generate_data(domain, params["n_features"] + 1)
        exp.train(max_epochs=1, verbose=0, initializing=True)

        # Set parameters on predictors
        for predictor, predictor_params in zip(exp.predictors, predictors_params):
            exp.set_predictor_params(predictor, unjsonify_dict(predictor_params))

        return exp

    @staticmethod
    def set_predictor_params(predictor, predictor_params):
        # Input transforms
        num = predictor.regressor_.named_steps.preprocessor.named_transformers_.num
        cat = predictor.regressor_.named_steps.preprocessor.named_transformers_.cat
        input_preprocessor = RecursiveNamespace(
            **predictor_params["input_preprocessor"]
        )
        num.mean_ = input_preprocessor.num.mean_
        num.var_ = input_preprocessor.num.var_
        num.scale_ = input_preprocessor.num.scale_
        num.n_samples_seen_ = input_preprocessor.num.n_samples_seen_

        # Output transforms
        out = predictor.transformer_
        output_preprocessor = RecursiveNamespace(
            **predictor_params["output_preprocessor"]
        )
        out.mean_ = output_preprocessor.mean_
        out.var_ = output_preprocessor.var_
        out.scale_ = output_preprocessor.scale_
        out.n_samples_seen_ = output_preprocessor.n_samples_seen_

    def save_regressor(self, save_dir):
        """Save the weights and biases of the regressor to disk

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory used for saving emulator files.

        """
        save_dir = pathlib.Path(save_dir)
        if self.predictors is None:
            raise ValueError(
                "No predictors available. First, run training using the train method."
            )
        for i, predictor in enumerate(self.predictors):
            predictor.regressor_.named_steps.net.save_params(
                f_params=save_dir / f"{self.model_name}_predictor_{i}.pt"
            )

    def load_regressor(self, save_dir):
        """Load the weights and biases of the regressor from disk

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory used for saving emulator files.

        """
        save_dir = pathlib.Path(save_dir)
        for i, predictor in enumerate(self.predictors):
            net = predictor.regressor_.named_steps.net
            net.initialize()
            net.load_params(f_params=save_dir / f"{self.model_name}_predictor_{i}.pt")

    def save(self, save_dir):
        """Save all the essential parameters of the ExperimentalEmulator to disk

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory used for saving emulator files.

        """
        save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir()
        with open(save_dir / f"{self.model_name}.json", "w") as f:
            json.dump(self.to_dict(), f)
        self.save_regressor(save_dir)

    @classmethod
    def load(cls, model_name, save_dir, **kwargs):
        """Load all the essential parameters of the ExperimentalEmulator to disk

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory from which to load emulator files.

        """
        save_dir = pathlib.Path(save_dir)
        with open(save_dir / f"{model_name}.json", "r") as f:
            d = json.load(f)
        exp = ExperimentalEmulator.from_dict(d, **kwargs)
        exp.load_regressor(save_dir)
        return exp

    def parity_plot(self, **kwargs):
        """Produce a parity plot based for the trained model using matplotlib

        Parameters
        ---------
        output_variable_names : str or list, optional
            The output variables to plot. Defaults to all.
        include_test : bool, optional
            Include the performance of the model on the test set.
            Defaults to False.
        train_color : str, optional
            Hex string for the train points. Defaults to "#6f3666"
        test_color : str, optional
            Hex string for the train points. Defaults to "#3c328c"

        """
        import matplotlib.pyplot as plt

        include_test = kwargs.get("include_test", False)
        train_color = kwargs.get("train_color", "#6f3666")
        test_color = kwargs.get("test_color", "#3c328c")
        clip = kwargs.get("clip")
        vars = kwargs.get("output_variable_names", self.output_variable_names)
        if type(vars) == str:
            vars = [vars]

        fig, axes = plt.subplots(1, len(vars))
        if len(vars) > 1:
            fig.subplots_adjust(wspace=0.2)
        if type(axes) != np.ndarray:
            axes = np.array([axes])

        # Do predictions
        with torch.no_grad():
            y_train_pred, y_train_pred_std = self._predict(self.X_train)
            if include_test:
                y_test_pred, y_train_pred_std = self._predict(self.X_test)

        plots = 0
        for i, v in enumerate(self.output_variable_names):
            if v in vars:
                if include_test:
                    kwargs = dict(
                        y_test=self.y_test[:, i], y_test_pred=y_test_pred[:, i]
                    )
                else:
                    kwargs = {}
                make_parity_plot(
                    self.y_train[:, i],
                    y_train_pred[:, i],
                    ax=axes[plots],
                    train_color=train_color,
                    test_color=test_color,
                    title=v,
                    **kwargs,
                )
                plots += 1

        return fig, axes


def generate_data(domain, n_examples, random_state=None):
    data = {}
    random = default_rng(random_state)
    for v in domain.input_variables:
        if v.variable_type == "continuous":
            data[v.name] = random.normal(size=n_examples)
        elif v.variable_type == "categorical":
            data[v.name] = random.choice(v.levels, size=n_examples)
    for v in domain.output_variables:
        if v.variable_type == "continuous":
            data[v.name] = random.normal(size=n_examples)
    return pd.DataFrame(data)


def make_parity_plot(
    y_train,
    y_train_pred,
    y_test=None,
    y_test_pred=None,
    ax=None,
    train_color="#6f3666",
    test_color="#3c328c",
    title=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if ax is None:
        fig, ax = plt.subplots(1)
    ax.scatter(y_train, y_train_pred, color=train_color, alpha=0.5)
    # Test
    if y_test is not None:
        ax.scatter(y_test, y_test_pred, color=test_color, alpha=0.5)

    # Parity line
    min = np.min(np.concatenate([y_train, y_train_pred]))
    max = np.max(np.concatenate([y_train, y_train_pred]))
    ax.plot([min, max], [min, max], c="#747378")
    # Scores
    handles = []
    r2_train = r2_score(y_train, y_train_pred)
    r2_train_patch = mpatches.Patch(
        label=f"Train R2 = {r2_train:.2f}", color=train_color
    )
    handles.append(r2_train_patch)
    if y_test is not None:
        r2_test = r2_score(y_test, y_test_pred)
        r2_test_patch = mpatches.Patch(
            label=f"Test R2 = {r2_test:.2f}", color=test_color
        )
        handles.append(r2_test_patch)

    # Formatting
    ax.legend(handles=handles, fontsize=12)
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    if title is not None:
        ax.set_title(title)
    ax.tick_params(direction="in")
    return ax


def numpy_to_tensor(X):
    """Convert datasets into """
    return torch.tensor(X).float()


class UpdatedTransformedTargetRegressor(TransformedTargetRegressor):
    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Parameters passed to the ``fit`` method of the underlying
            regressor.
        Returns
        -------
        self : object
        """
        y = check_array(
            y,
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype="numeric",
        )

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # Remove this stupid line
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        # if y_trans.ndim == 2 and y_trans.shape[1] == 1:
        #     y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from ..linear_model import LinearRegression

            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        return self


class RecursiveNamespace(types.SimpleNamespace):
    # def __init__(self, /, **kwargs):  # better, but Python 3.8+
    def __init__(self, **kwargs):
        """Create a SimpleNamespace recursively"""
        self.__dict__.update({k: self.__elt(v) for k, v in kwargs.items()})

    def __elt(self, elt):
        """Recurse into elt to create leaf namepace objects"""
        if type(elt) is dict:
            return type(self)(**elt)
        if type(elt) in (list, tuple):
            return [self.__elt(i) for i in elt]
        return elt


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, val):
        self._total = val

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class ProgressGridSearchCV(BaseSearchCV):
    @_deprecate_positional_args
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    @_deprecate_positional_args
    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            # self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = ProgressParallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )
                runs = product(
                    enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                )
                parallel.total = len(list(deepcopy(runs)))
                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in runs
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError("best_index_ returned is not an integer")
                if self.best_index_ < 0 or self.best_index_ >= len(results["params"]):
                    raise IndexError("best_index_ index out of range")
            else:
                self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _check_refit_for_multimetric(self, scores):
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if (
            self.refit is not False
            and not valid_refit_dict
            and not callable(self.refit)
        ):
            raise ValueError(multimetric_refit_msg)


def get_bnn():
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator

    @variational_estimator
    class BNNRegressor(torch.nn.Module):
        """A Bayesian Neural Network pytorch lightining module"""

        val_str = "CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}"

        def __init__(
            self, input_dim, output_dim, n_examples=100, hidden_units=512, **kwargs
        ):
            super().__init__()
            self.blinear1 = BayesianLinear(input_dim, hidden_units)
            self.blinear2 = BayesianLinear(hidden_units, output_dim)
            self.n_examples = n_examples
            self.n_samples = kwargs.get("n_samples", 50)
            self.criterion = torch.nn.MSELoss()

        def forward(self, x):
            # for layer in self.layers[:-1]:
            #     x = layer(x)
            #     x = F.relu(x)
            # return self.layers[-1](x)
            x = self.blinear1(x)
            x = F.relu(x)
            return self.blinear2(x)

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


class ANNRegressor(torch.nn.Module):
    """Artificial Neural Network Regressor

    Parameters
    -----------
    input_dim : int
        The number of features in the input
    output_dim : int
        The number of outputs in the targets
    hidden_units : int, optional
        The number of hidden units. Default is 512.

    """

    def __init__(self, input_dim, output_dim, hidden_units=512, **kwargs):
        super().__init__()

        self.num_hidden_layers = 1
        self.input_layer = torch.nn.Linear(input_dim, hidden_units)
        self.output_layer = torch.nn.Linear(hidden_units, output_dim)

    def forward(self, x, **kwargs):
        x_ = F.relu(self.input_layer(x))
        if self.num_hidden_layers > 1:
            x_ = self.hidden_layers(x_)
            x_ = F.relu(x_)
        return self.output_layer(x_)


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
                f"{key} is not in the §. Register using the register method."
            )

    def __setitem__(self, key, value):
        reg = self.regressors.get(key)
        if reg is not None:
            self.regressors[key] = value

    def register(self, regressor):
        key = regressor.__name__
        self.regressors[key] = regressor


# Create global regressor registry
registry = RegressorRegistry()
registry.register(ANNRegressor)


def get_data_path():
    return pathlib.Path(pkg_resources.resource_filename("summit", "benchmarks/data"))


def get_model_path():
    return pathlib.Path(pkg_resources.resource_filename("summit", "benchmarks/models"))


def get_pretrained_reizman_suzuki_emulator(case=1):
    model_name = f"reizman_suzuki_case_{case}"
    model_path = get_model_path() / model_name
    if not model_path.exists():
        raise NotADirectoryError("Could not initialize from expected path.")
    exp = ReizmanSuzukiEmulator.load(model_path, case=case)
    data_path = get_data_path()
    exp.ds = DataSet.read_csv(data_path / f"{model_name}.csv")
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
        cases. Each case was has a different set of substrates but the
        same possible catalysts. Please see their paper for more information on the cases.

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
        # Initialization
        model_name = kwargs.get("model_name", f"reizman_suzuki_case_{case}")
        domain = kwargs.pop("domain", self.setup_domain())
        data_path = get_data_path()
        ds = DataSet.read_csv(data_path / f"{model_name}.csv")
        super().__init__(model_name, domain, dataset=ds, **kwargs)

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

    @classmethod
    def load(cls, save_dir, case=1):
        model_name = f"reizman_suzuki_case_{case}"
        return super().load(model_name, save_dir)

    @classmethod
    def to_dict(self):
        """Serialize the class to a dictionary"""
        experiment_params = dict(
            case=self.model_name[-1],
        )
        return super().to_dict(**experiment_params)


def get_pretrained_baumgartner_cc_emulator(include_cost=False):
    model_name = "baumgartner_aniline_cn_crosscoupling"
    model_path = get_model_path() / model_name
    if not model_path.exists():
        raise NotADirectoryError("Could not initialize from expected path.")
    data_path = get_data_path()
    ds = DataSet.read_csv(data_path / f"{model_name}.csv")
    exp = BaumgartnerCrossCouplingEmulator.load(model_path, dataset=ds)

    return exp


class BaumgartnerCrossCouplingEmulator(ExperimentalEmulator):
    """Baumgartner Cross Coupling Emulator

    Virtual experiments representing the Aniline Cross-Coupling reaction
    similar to Baumgartner et al. (2019). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Baumgartner et al.

    This is a five dimensional optimisation of temperature, residence time, base equivalents,
    catalyst and base.

    The categorical variables (catalyst and base) contain descriptors
    calculated using COSMO-RS. Specifically, the descriptors are the first two sigma moments.

    Parameters
    ----------
    include_cost : bool, optional
        Include minimization of cost as an extra objective. Cost is calculated
        as a deterministic function of the inputs (i.e., no model is trained).
        Defaults to False.

    Examples
    --------
    >>> bemul = BaumgartnerCrossCouplingEmulator()

    Notes
    -----
    This benchmark is based on data from [Baumgartner]_ et al.

    References
    ----------

    .. [Baumgartner] L. M. Baumgartner et al., Org. Process Res. Dev., 2019, 23, 1594–1601
       DOI: `10.1021/acs.oprd.9b00236 <https://`doi.org/10.1021/acs.oprd.9b00236>`_

    """

    def __init__(self, include_cost=False, **kwargs):
        # TODO: make it possible to select model based on one-hot encoding or descriptors
        model_name = kwargs.pop("model_name", "baumgartner_aniline_cn_crosscoupling")
        self.include_cost = include_cost
        domain = kwargs.pop("domain", self.setup_domain(self.include_cost))
        data_path = get_data_path()
        super().__init__(model_name, domain, **kwargs)

    @staticmethod
    def setup_domain(include_cost=False):
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
            name="yield",
            description=des_6,
            bounds=[0.0, 1.0],
            is_objective=True,
            maximize=True,
        )

        if include_cost:
            domain += ContinuousVariable(
                name="cost",
                description="cost in USD of 40 uL reaction",
                bounds=[0.0, 1.0],
                is_objective=True,
                maximize=False,
            )

        return domain

    @classmethod
    def load(cls, save_dir, **kwargs):
        """Load all the essential parameters of the BaumgartnerCrossCouplingEmulator
        from disc

        Parameters
        ----------
        save_dir : str or pathlib.Path
            The directory from which to load emulator files.

        """
        model_name = "baumgartner_aniline_cn_crosscoupling"
        return super().load(model_name, save_dir, **kwargs)

    def _run(self, conditions, **kwargs):
        conditions, _ = super()._run(conditions=conditions, **kwargs)

        # Calculate costs
        if self.include_cost:
            costs = self._calculate_costs(conditions)
            conditions[("cost", "DATA")] = costs

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
