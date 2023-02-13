from .base import Strategy
from .random import LHS
from .factorial_doe import fullfact
from summit.domain import *
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit.utils.dataset import DataSet
from summit import get_summit_config_path

from pymoo.core.problem import Problem

from fastprogress.fastprogress import progress_bar
from scipy.sparse import issparse
import pathlib
import os
import numpy as np
import pandas as pd
import uuid
import logging


class TSEMO(Strategy):
    """Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO)

    TSEMO is a multiobjective Bayesian optimisation strategy. It is designed
    to find optimal values in as few iterations as possible. This comes at the price
    of higher computational time.

    Parameters
    ----------

    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    use_descriptors : bool, optional
        Whether to use descriptors of categorical variables. Defaults to False.
    kernel : :class:`~GPy.kern.Kern`, optional
        A GPy kernel class (not instantiated). Must be Exponential,
        Matern32, Matern52 or RBF. Default Exponential.
    n_spectral_points : int, optional
        Number of spectral points used in spectral sampling.
        Default is 1500. Note that the Matlab TSEMO version uses 4000
        which will improve accuracy but significantly slow down optimisation speed.
    n_retries : int, optional
        Number of retries to use for spectral sampling iF the singular value decomposition
        fails. Retrying chooses a new Monte Carlo sampling which usually fixes the problem.
        Defualt is 10.
    generations : int, optional
        Number of generations used in the internal optimisation with NSGAII.
        Default is 100.
    pop_size : int, optional
        Population size used in the internal optimisation with NSGAII.
        Default is 100.

    Attributes
    ----------



    Examples
    --------

    >>> from summit.domain import *
    >>> from summit.strategies import TSEMO
    >>> from summit.utils.dataset import DataSet
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> columns = [v.name for v in domain.variables]
    >>> values = {("temperature", "DATA"): 60,("flowrate_a", "DATA"): 0.5,("flowrate_b", "DATA"): 0.5,("yield_", "DATA"): 50,("de", "DATA"): 90}
    >>> previous_results = DataSet([values], columns=columns)
    >>> strategy = TSEMO(domain)
    >>> result = strategy.suggest_experiments(5, prev_res=previous_results)

    Notes
    -----

    TSEMO trains a gaussian process (GP) to model each objective. Internally, we use
    `GPy <https://github.com/SheffieldML/GPy>`_ for GPs, and we accept any kernel in the Matérn family, including the
    exponential and squared exponential kernel. See [Rasmussen]_ for more information about GPs.

    A deterministic function is sampled from each of the trained GPs. We use spectral sampling available in `pyrff <https://github.com/michaelosthege/pyrff>`_.
    These sampled functions are optimised using NSGAII (via `pymoo <https://pymoo.org/>`_) to find a selection of potential conditions.
    Each of these conditions are evaluated using the hypervolume improvement (HVI) criterion, and the one(s) that offer the best
    HVI are suggested as the next experiments. More details about TSEMO can be found in the original paper [Bradford]_.

    The number of spectral points is the parameter that most affects TSEMO performance. By default, it's set at 1500,
    but increase it to around 4000 to get the best performance at the cost of longer computational times.  You can change it using the n_spectral_points keyword argument.

    The other two parameters are the number of generations and population size used in NSGA-II. Increasing their values can improve
    performance in some cases.


    References
    ----------

    .. [Rasmussen] C. E. Rasmussen et al.
       Gaussian Processes for Machine Learning, MIT Press, 2006.

    .. [Bradford] E. Bradford et al.
       "Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm."
       J. Glob. Optim., 2018, 71, 407–438.

    """

    def __init__(self, domain, transform=None, **kwargs):
        Strategy.__init__(self, domain, transform, **kwargs)

        # Categorical variable options
        self.use_descriptors = kwargs.get("use_descriptors", False)
        n_categoricals = len(
            [v for v in self.domain.input_variables if v.variable_type == "categorical"]
        )
        if n_categoricals > 0:
            self.categorical_combos = self.domain.get_categorical_combinations()
        else:
            self.categorical_combos = None

        # Input columns
        self.input_columns = []
        for v in self.domain.input_variables:
            if type(v) == ContinuousVariable:
                self.input_columns.append(v.name)
            elif (
                type(v) == CategoricalVariable
                and v.ds is not None
                and self.use_descriptors
            ):
                self.input_columns += [c[0] for c in v.ds.columns]


        # Spectral sampling settings
        self.n_spectral_points = kwargs.get("n_spectral_points", 1500)
        self.n_retries = kwargs.get("n_retries", 10)

        # NSGA-II tsemo_settings
        self.generations = kwargs.get("generations", 1000)
        self.pop_size = kwargs.get("pop_size", 100)

        self.logger = kwargs.get("logger", logging.getLogger(__name__))
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        """Suggest experiments using TSEMO

        Parameters
        ----------
        num_experiments : int
            The number of experiments (i.e., samples) to generate
        prev_res : :class:`~summit.utils.data.DataSet`, optional
            Dataset with data from previous experiments.
            If no data is passed, then latin hypercube sampling will
            be used to suggest an initial design.

        Returns
        -------
        next_experiments : :class:`~summit.utils.data.DataSet`
            A Dataset object with the suggested experiments
            The lengthscales column tells the significance of each variable (assuming automatic relevance detection is turned on, which it is in Botorch).
            Smaller values mean significant changes in output happen on a smaller change in the input, suggesting a more important input.
            The variance column scales the output of the posterior of the kernel to the correct scale for your objective
            The noise column is the constant noise in outputs (e.g., assumed uniform experiment error)

        """
        # Suggest lhs initial design or append new experiments to previous experiments
        cat_method = "descriptors" if self.use_descriptors else None
        if prev_res is None:
            lhs = LHS(self.domain, categorical_method=cat_method)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            return lhs.suggest_experiments(k, criterion="maximin")
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = pd.concat([self.all_experiments, prev_res], axis=0)
        
        if self.all_experiments.shape[0] <= 3:
            lhs = LHS(self.domain, categorical_method=cat_method)
            self.iterations += 1
            return lhs.suggest_experiments(num_experiments)

        # Get inputs (decision variables) and outputs (objectives)
        cat_method = "descriptors" if self.use_descriptors else "one-hot"
        inputs, outputs = self.transform.transform_inputs_outputs(
            self.all_experiments,
            categorical_method=cat_method,
            min_max_scale_inputs=True,
            standardize_outputs=True,
        )
        if inputs.shape[0] < self.domain.num_continuous_dimensions():
            self.logger.warning(
                (
                    f"""The number of examples ({inputs.shape[0]}) is less the number"""
                    f"""of input dimensions ({self.domain.num_continuous_dimensions()}."""
                )
            )

        # Train and sample
        n_outputs = len(self.domain.output_variables)
        train_results = [0] * n_outputs
        self.models = [0] * n_outputs
        rmse_train_spectral = np.zeros(n_outputs)
        for i, v in enumerate(self.domain.output_variables):
            # Training
            self.models[i] = ThompsonSampledModel(v.name)
            train_results[i] = self.models[i].fit(
                inputs,
                outputs[[v.name]],
                n_retries=self.n_retries,
                n_spectral_points=self.n_spectral_points,
            )

            # Evaluate spectral sampled functions
            sample_f = lambda x: np.atleast_2d(self.models[i].rff(x)).T
            rmse_train_spectral[i] = rmse(
                sample_f(inputs.to_numpy().astype("float")),
                outputs[[v.name]].to_numpy().astype("float"),
                mean=self.transform.output_means[v.name],
                std=self.transform.output_stds[v.name],
            )
            self.logger.debug(
                f"RMSE train spectral {v.name} = {rmse_train_spectral[i].round(2)}"
            )

        # NSGAII internal optimisation on spectrally sampled functions
        self.logger.info("Optimizing models using NSGAII.")

        # Categorical only domain
        if (self.domain.num_continuous_dimensions() == 0) and (
            self.domain.num_categorical_variables() == 1
        ):
            X, yhat = self._categorical_enumerate(self.models)
        # Mixed domains
        elif self.categorical_combos is not None and len(self.input_columns) > 1:
            X, yhat = self._nsga_optimize_mixed(self.models)
        # Continous domains
        elif self.categorical_combos is None and len(self.input_columns) > 0:
            X, yhat = self._nsga_optimize(self.models)

        # Return if no suggestiosn found
        if X.shape[0] == 0 and yhat.shape[0] == 0:
            self.logger.warning("No suggestions found.")
            self.iterations += 1
            return None

        # Select points that give maximum hypervolume improvement
        self.hv_imp, indices = self._select_max_hvi(y=outputs, yhat=yhat, num_evals=num_experiments)

        # Join to get single dataset with inputs and outputs and get suggestion
        result = X.join(yhat)
        result = result.iloc[indices, :]

        # Do any necessary transformations back
        result = self.transform.un_transform(
            result,
            categorical_method=cat_method,
            min_max_scale_inputs=True,
            standardize_outputs=True,
        )

        # Add model hyperparameters as metadata columns
        result[("strategy", "METADATA")] = "TSEMO"
        i = 0
        for res in train_results:
            output_name = res["name"]
            result[(f"{output_name}_variance", "METADATA")] = res["outputscale"]
            result[(f"{output_name}_noise", "METADATA")] = res["noise"]
            result[f"rmse_train_spectral", "METADATA"] = rmse_train_spectral[i]
            i += 1
            for var, l in zip(self.domain.input_variables, res["lengthscales"]):
                result[(f"{output_name}_{var.name}_lengthscale", "METADATA")] = l
        self.iterations += 1
        result[("iterations", "METADATA")] = self.iterations
        return result

    def _nsga_optimize(self, models):
        """NSGA-II optimization with categorical domains"""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.factory import get_termination

        optimizer = NSGA2(pop_size=self.pop_size)
        problem = TSEMOInternalWrapper(models, self.domain)
        termination = get_termination("n_gen", self.generations)
        self.internal_res = minimize(
            problem, optimizer, termination, seed=1, verbose=False
        )

        X = np.atleast_2d(self.internal_res.X).tolist()
        y = np.atleast_2d(self.internal_res.F).tolist()
        X = DataSet(X, columns=problem.X_columns)
        y = DataSet(y, columns=[v.name for v in self.domain.output_variables])
        for v in self.domain.output_variables:
            if v.maximize:
                y[v.name] = -y[v.name]
        return X, y

    def _nsga_optimize_mixed(self, models):
        """NSGA-II optimization with mixed continuous-categorical domains"""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.factory import get_termination

        combos = self.categorical_combos
        transformed_combos = self._transform_categorical(combos)

        X_list, y_list = [], []

        # Loop through all combinations of categoricals and run optimization
        bar = progress_bar(
            transformed_combos.iterrows(), total=transformed_combos.shape[0]
        )
        for _, combo in bar:
            # bar.comment = "NSGA Mixed Optimization"
            optimizer = NSGA2(pop_size=self.pop_size)
            problem = TSEMOInternalWrapper(
                models, self.domain, fixed_variables=combo.to_dict()
            )
            termination = get_termination("n_gen", self.generations)
            self.internal_res = minimize(
                problem, optimizer, termination, seed=1, verbose=False
            )

            X = np.atleast_2d(self.internal_res.X).tolist()
            y = np.atleast_2d(self.internal_res.F).tolist()
            X = DataSet(X, columns=problem.X_columns)
            y = DataSet(y, columns=[v.name for v in self.domain.output_variables])
            for v in self.domain.output_variables:
                if v.maximize:
                    y[v.name] = -y[v.name]
            # Add in categorical variables
            for key, value in combo.to_dict().items():
                X[key] = value
            X_list.append(X)
            y_list.append(y)

        return pd.concat(X_list, axis=0), pd.concat(y_list, axis=0)

    def _transform_categorical(self, X):
        transformed_combos = {}
        for v in self.domain.input_variables:
            if v.variable_type == "categorical":
                values = X[v.name].to_numpy()

                # Descriptor transformation
                if self.use_descriptors and v.ds is not None:
                    transformed_values = v.ds.loc[values]
                    for col in transformed_values:
                        transformed_combos[col] = transformed_values[col[0]].to_numpy()
                        var_max = v.ds[col[0]].max()
                        var_min = v.ds[col[0]].min()
                        transformed_combos[col] = (
                            transformed_combos[col] - var_min
                        ) / (var_max - var_min)
                elif self.use_descriptors and v.ds is None:
                    raise DomainError(
                        f"use_descriptors is true, but {v.name} has no descriptors."
                    )
                # One hot encoding transformation
                else:
                    enc = self.transform.encoders[v.name]
                    one_hot_values = enc.transform(values[:, np.newaxis])
                    if issparse(one_hot_values):
                        one_hot_values = one_hot_values.toarray()
                    for loc, l in enumerate(v.levels):
                        column_name = f"{v.name}_{l}"
                        transformed_combos[(column_name, "DATA")] = one_hot_values[
                            :, loc
                        ]
        return DataSet(transformed_combos)

    def _categorical_enumerate(self, models):
        """Make predictions on all combinations of categorical domain"""
        combos = self.categorical_combos
        X = self._transform_categorical(combos)
        n_obj = len(self.domain.output_variables)
        y = np.zeros([X.shape[0], n_obj])
        for i, v in enumerate(self.domain.output_variables):
            y[:, i] = models[i].predict(X)
        y = DataSet(y, columns=[v.name for v in self.domain.output_variables])
        return X, y

    def reset(self):
        """Reset TSEMO state"""
        self.all_experiments = None
        self.iterations = 0
        self.sample_fs = [0 for i in range(len(self.domain.output_variables))]
        self.uuid_val = uuid.uuid4()

    def to_dict(self):
        ae = (
            self.all_experiments.to_dict() if self.all_experiments is not None else None
        )
        strategy_params = dict(
            all_experiments=ae,
            n_spectral_points=self.n_spectral_points,
            n_retries=self.n_retries,
            pop_size=self.pop_size,
            generation=self.generations,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):

        tsemo = super().from_dict(d)
        ae = d["strategy_params"]["all_experiments"]
        if ae is not None:
            tsemo.all_experiments = DataSet.from_dict(ae)
        return tsemo

    def _select_max_hvi(self, y, yhat, num_evals=1):
        """Returns the point(s) that maximimize hypervolume improvement

        Parameters
        ----------
        samples: np.ndarray
             The samples on which hypervolume improvement is calculated
        num_evals: `int`
            The number of points to return (with top hypervolume improvement)

        Returns
        -------
        hv_imp, index
            Returns a tuple with lists of the best hypervolume improvement
            and the indices of the corresponding points in samples

        """
        yhat = yhat.copy()
        y = y.copy()

        # Set up maximization and minimization
        for v in self.domain.output_variables:
            if v.maximize:
                y[v.name] = -1 * y[v.name]
                yhat[v.name] = -1 * yhat[v.name]

        # samples, mean, std = samples.standardize(return_mean=True, return_std=True)
        yhat = yhat.data_to_numpy()
        Ynew = y.data_to_numpy()

        # Reference
        Yfront, _ = pareto_efficient(Ynew, maximize=False)
        r = np.max(Yfront, axis=0) + 0.01 * (
            np.max(Yfront, axis=0) - np.min(Yfront, axis=0)
        )

        indices = []
        n = yhat.shape[1]
        mask = np.ones(yhat.shape[0], dtype=bool)
        samples_indices = np.arange(0, yhat.shape[0])

        for i in range(num_evals):
            masked_samples = yhat[mask, :]
            Yfront, _ = pareto_efficient(Ynew, maximize=False)
            if len(Yfront) == 0:
                raise ValueError("Pareto front length too short")

            hv_improvement = []
            hvY = hypervolume(Yfront, r)
            # Determine hypervolume improvement by including
            # each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1, n)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = pareto_efficient(A, maximize=False)
                hv = hypervolume(Afront, r)
                hv_improvement.append(hv - hvY)

            hvY0 = hvY if i == 0 else hvY0
            hv_improvement = np.array(hv_improvement)
            masked_index = np.argmax(hv_improvement)

            # Housekeeping: find the max HvI point and mask out for next round
            original_index = samples_indices[mask][masked_index]
            new_point = yhat[original_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[original_index] = False
            indices.append(original_index)

        if len(hv_improvement) == 0:
            hv_imp = 0
        elif len(indices) == 0:
            indices = []
            hv_imp = 0
        else:
            # Total hypervolume improvement
            Yfront, _ = pareto_efficient(Ynew, maximize=False)
            hv_imp =  hypervolume(Yfront, r) - hvY0
        return hv_imp, indices


def rmse(Y_pred, Y_true, mean, std):
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean
    square_error = (Y_pred[:, 0] - Y_true[:, 0]) ** 2
    return np.sqrt(np.mean(square_error))


class ThompsonSampledModel:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.input_columns_ordered = None
        self.output_columns_ordered = None
        self.logger = logging.getLogger(__name__)

    def fit(self, X: DataSet, y: DataSet, **kwargs):
        """Train model and take spectral samples"""
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_model
        from gpytorch.mlls.exact_marginal_log_likelihood import (
            ExactMarginalLogLikelihood,
        )
        import pyrff
        import torch

        self.input_columns_ordered = X.columns

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
        import pyrff
        if filepath is None:
            filepath = get_summit_config_path() / "tsemo" / str(self.uuid_val)
            os.makedirs(filepath, exist_ok=True)
            filepath = filepath / "models.h5"
        pyrff.save_rffs([self.rff], filepath)

    def load(self, filepath=None):
        import pyrff
        if filepath is None:
            filepath = get_summit_config_path() / "tsemo" / str(self.uuid_val)
            os.makedirs(filepath, exist_ok=True)
            filepath = filepath / "models.h5"
        self.rff = pyrff.load_rffs(filepath)[0]


class TSEMOInternalWrapper(Problem):
    """Wrapper for NSGAII internal optimisation

    Parameters
    ----------
    models : list of :class:`ThompsonSampledModel`
        The models to optimize
    domain : :class:`~summit.domain.Domain`
        Domain used for optimisation.
    fixed_variable_names : list, optional
        A list of variables which should take on fixed values.
    Notes
    -----
    It is assumed that the inputs are scaled between 0 and 1.

    """

    def __init__(self, models, domain, fixed_variables: dict = None):
        import pyrff

        self.models = models
        self.domain = domain
        self.fixed_variables = fixed_variables

        # Number of decision variables
        # Categoricals are not optimized by NSGA, hence no descriptors
        n_var = domain.num_continuous_dimensions(include_descriptors=False)
        self.X_columns = [
            v.name
            for v in self.domain.input_variables
            if v.variable_type == "continuous"
        ]

        # Number of objectives
        n_obj = len(domain.output_variables)

        # Number of constraints
        n_constr = len(domain.constraints)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        # Convert X to a DataSet
        X = DataSet(np.atleast_2d(X), columns=self.X_columns)

        # Add in any fixed columns (i.e., values for cateogricals)
        if self.fixed_variables is not None:
            for key, value in self.fixed_variables.items():
                X[key] = value

        F = np.zeros([X.shape[0], self.n_obj])
        for i in range(self.n_obj):
            F[:, i] = self.models[i].predict(X)

        # Negate objectives that are need to be maximized
        for i, v in enumerate(self.domain.output_variables):
            if v.maximize:
                F[:, i] *= -1
        out["F"] = F

        # Add constraints if necessary
        if self.domain.constraints:
            constraint_res = [
                X.eval(c.lhs, resolvers=[X]) for c in self.domain.constraints
            ]
            out["G"] = [c.tolist()[0] for c in constraint_res]
