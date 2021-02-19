from .base import Strategy
from .random import LHS
from summit.domain import *
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit.utils.dataset import DataSet
from summit import get_summit_config_path

from pymoo.model.problem import Problem

import pathlib
import os
import numpy as np
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
    >>> result = strategy.suggest_experiments(5)

    Notes
    -----

    TSEMO trains a gaussian process (GP) to model each objective. Internally, we use
    `GPy <https://github.com/SheffieldML/GPy>`_ for GPs, and we accept any kernel in the Matérn family, including the
    exponential and squared exponential kernel. See [Rasmussen]_ for more information about GPs.

    A deterministic function is sampled from each of the trained GPs. We use spectral sampling available in `pyrff <https://github.com/michaelosthege/pyrff>`_.
    These sampled functions are optimised using NSGAII (via `pymoo <https://pymoo.org/>`_) to find a selection of potential conditions.
    Each of these conditions are evaluated using the hypervolume improvement (HVI) criterion, and the one(s) that offer the best
    HVI are suggested as the next experiments. More details about TSEMO can be found in the original paper [Bradford]_.

    References
    ----------

    .. [Rasmussen] C. E. Rasmussen et al.
       Gaussian Processes for Machine Learning, MIT Press, 2006.

    .. [Bradford] E. Bradford et al.
       "Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm."
       J. Glob. Optim., 2018, 71, 407–438.

    """

    def __init__(self, domain, transform=None, **kwargs):
        from GPy.kern import Exponential

        Strategy.__init__(self, domain, transform, **kwargs)

        # Input bounds
        lowers = []
        uppers = []
        self.columns = []
        for v in self.domain.input_variables:
            if type(v) == ContinuousVariable:
                lowers.append(v.bounds[0])
                uppers.append(v.bounds[1])
                self.columns.append(v.name)
            elif type(v) == CategoricalVariable and v.ds is not None:
                lowers += v.ds.min().to_list()
                uppers += v.ds.max().to_list()
                self.columns += [c[0] for c in v.ds.columns]
            elif type(v) == CategoricalVariable and v.ds is None:
                raise DomainError(
                    "TSEMO only supports categorical variables with descriptors."
                )
        self.inputs_min = DataSet([lowers], columns=self.columns)
        self.inputs_max = DataSet([uppers], columns=self.columns)
        self.kern_dim = len(self.columns)

        # Kernel
        self.kernel = kwargs.get("kernel", Exponential)

        # Spectral sampling settings
        self.n_spectral_points = kwargs.get("n_spectral_points", 1500)
        self.n_retries = kwargs.get("n_retries", 10)

        # NSGA-II tsemo_settings
        self.generations = kwargs.get("generations", 100)
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
        """
        from GPy.models import GPRegression as gpr
        from GPy.core.parameterization.priors import LogGaussian
        from GPy.kern import Exponential, Matern32, Matern52, RBF
        import pyrff
        from pymoo.algorithms.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.factory import get_termination

        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            return lhs.suggest_experiments(k, criterion="maximin")
        elif self.iterations == 1 and len(prev_res) == 1:
            lhs = LHS(self.domain)
            self.iterations += 1
            self.all_experiments = prev_res
            return lhs.suggest_experiments(num_experiments)
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)

        # Get inputs (decision variables) and outputs (objectives)
        inputs, outputs = self.transform.transform_inputs_outputs(
            self.all_experiments, categorical_method="descriptors"
        )
        if inputs.shape[0] < self.domain.num_continuous_dimensions():
            self.logger.warning(
                f"The number of examples ({inputs.shape[0]}) is less the number of input dimensions ({self.domain.num_continuous_dimensions()}."
            )

        # Scale decision variables [0,1]
        inputs_scaled = (inputs - self.inputs_min.to_numpy()) / (
            self.inputs_max.to_numpy() - self.inputs_min.to_numpy()
        )

        # Standardize objectives
        self.output_mean = outputs.mean()
        std = outputs.std()
        std[std < 1e-5] = 1e-5
        self.output_std = std
        outputs_scaled = (
            outputs - self.output_mean.to_numpy()
        ) / self.output_std.to_numpy()

        # Set up models
        input_dim = self.kern_dim
        self.models = {
            v.name: gpr(
                inputs_scaled.to_numpy(),
                outputs_scaled[[v.name]].to_numpy(),
                kernel=self.kernel(input_dim=input_dim, ARD=True),
            )
            for v in self.domain.output_variables
        }

        output_dim = len(self.domain.output_variables)
        rmse_train = np.zeros(output_dim)
        rmse_train_spectral = np.zeros(output_dim)
        lengthscales = [None for _ in range(output_dim)]
        variances = [None for _ in range(output_dim)]
        noises = [None for _ in range(output_dim)]
        rffs = [None for _ in range(output_dim)]
        i = 0
        num_restarts = kwargs.get(
            "num_restarts", 100
        )  # This is a kwarg solely for debugging
        self.logger.debug(
            f"Fitting models (number of optimization restarts={num_restarts})\n"
        )
        for name, model in self.models.items():
            # Constrain hyperparameters
            model.kern.lengthscale.constrain_bounded(
                np.sqrt(1e-3), np.sqrt(1e3), warning=False
            )
            model.kern.lengthscale.set_prior(LogGaussian(0, 10), warning=False)
            model.kern.variance.constrain_bounded(
                np.sqrt(1e-3), np.sqrt(1e3), warning=False
            )
            model.kern.variance.set_prior(LogGaussian(-6, 10), warning=False)
            model.Gaussian_noise.constrain_bounded(np.exp(-6), 1, warning=False)

            # Train model
            model.optimize_restarts(
                num_restarts=num_restarts, max_iters=10000, parallel=True, verbose=False
            )

            # self.logger.info model hyperparameters
            lengthscales[i] = model.kern.lengthscale.values
            variances[i] = model.kern.variance.values[0]
            noises[i] = model.Gaussian_noise.variance.values[0]
            self.logger.debug(f"Model {name} lengthscales: {lengthscales[i]}")
            self.logger.debug(f"Model {name} variance: {variances[i]}")
            self.logger.debug(f"Model {name} noise: {noises[i]}")

            # Model validation
            rmse_train[i] = rmse(
                model.predict(inputs_scaled.to_numpy())[0],
                outputs_scaled[[name]].to_numpy(),
                mean=self.output_mean[name].values[0],
                std=self.output_std[name].values[0],
            )
            self.logger.debug(f"RMSE train {name} = {rmse_train[i].round(2)}")

            # Spectral sampling
            self.logger.debug(
                f"Spectral sampling {name} with {self.n_spectral_points} spectral points."
            )
            if type(model.kern) == Exponential:
                matern_nu = 1
            elif type(model.kern) == Matern32:
                matern_nu = 3
            elif type(model.kern) == Matern52:
                matern_nu = 5
            elif type(model.kern) == RBF:
                matern_nu = np.inf
            else:
                raise TypeError(
                    "Spectral sample currently only works with Matern type kernels, including RBF."
                )

            for _ in range(self.n_retries):
                try:
                    rffs[i] = pyrff.sample_rff(
                        lengthscales=lengthscales[i],
                        scaling=np.sqrt(variances[i]),
                        noise=noises[i],
                        kernel_nu=matern_nu,
                        X=inputs_scaled.to_numpy(),
                        Y=outputs_scaled[[name]].to_numpy()[:, 0],
                        M=self.n_spectral_points,
                    )
                    break
                except np.linalg.LinAlgError as e:
                    self.logger.error(e)
                except ValueError as e:
                    self.logger.error(e)
            if rffs[i] is None:
                raise RuntimeError(
                    f"Spectral sampling failed after {self.n_retries} retries."
                )
            sample_f = lambda x: np.atleast_2d(rffs[i](x)).T

            rmse_train_spectral[i] = rmse(
                sample_f(inputs_scaled.to_numpy()),
                outputs_scaled[[name]].to_numpy(),
                mean=self.output_mean[name].values[0],
                std=self.output_std[name].values[0],
            )
            self.logger.debug(
                f"RMSE train spectral {name} = {rmse_train_spectral[i].round(2)}"
            )

            i += 1

        # Save spectral samples
        dp_results = get_summit_config_path() / "tsemo" / str(self.uuid_val)
        os.makedirs(dp_results, exist_ok=True)
        pyrff.save_rffs(rffs, pathlib.Path(dp_results, "models.h5"))

        # NSGAII internal optimisation
        self.logger.info("Optimizing models using NSGAII.")
        optimizer = NSGA2(pop_size=self.pop_size)
        problem = TSEMOInternalWrapper(
            pathlib.Path(dp_results, "models.h5"), self.domain, n_var=self.kern_dim
        )
        termination = get_termination("n_gen", self.generations)
        self.internal_res = minimize(
            problem, optimizer, termination, seed=1, verbose=False
        )
        X = DataSet(self.internal_res.X, columns=self.columns)
        y = DataSet(
            self.internal_res.F, columns=[v.name for v in self.domain.output_variables]
        )

        # Select points that give maximum hypervolume improvement
        if X.shape[0] != 0 and y.shape[0] != 0:
            self.hv_imp, indices = self._select_max_hvi(
                outputs_scaled, y, num_experiments
            )

            # Unscale data
            X = (
                X * (self.inputs_max.to_numpy() - self.inputs_min.to_numpy())
                + self.inputs_min.to_numpy()
            )
            y = y * self.output_std.to_numpy() + self.output_mean.to_numpy()

            # Join to get single dataset with inputs and outputs
            result = X.join(y)
            result = result.iloc[indices, :]

            # Do any necessary transformations back
            result[("strategy", "METADATA")] = "TSEMO"
            result = self.transform.un_transform(
                result, categorical_method="descriptors"
            )

            # Add model hyperparameters as metadata columns
            self.iterations += 1
            i = 0
            for name, model in self.models.items():
                result[(f"{name}_variance", "METADATA")] = variances[i]
                result[(f"{name}_noise", "METADATA")] = noises[i]
                for var, l in zip(self.domain.input_variables, lengthscales[i]):
                    result[(f"{name}_{var.name}_lengthscale", "METADATA")] = l
                result[("iterations", "METADATA")] = self.iterations
                i += 1
            return result
        else:
            self.logger.warning("No suggestions found.")
            self.iterations += 1
            return None

    def reset(self):
        """Reset TSEMO state"""
        self.all_experiments = None
        self.iterations = 0
        self.samples = []  # Samples drawn using NSGA-II
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

    def _select_max_hvi(self, y, samples, num_evals=1):
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
        samples_original = samples.copy()
        samples = samples.copy()
        y = y.copy()

        # Set up maximization and minimization
        for v in self.domain.variables:
            if v.is_objective and v.maximize:
                y[v.name] = -1 * y[v.name]
                samples[v.name] = -1 * samples[v.name]

        # samples, mean, std = samples.standardize(return_mean=True, return_std=True)
        samples = samples.data_to_numpy()
        Ynew = y.data_to_numpy()

        # Reference
        Yfront, _ = pareto_efficient(Ynew, maximize=False)
        r = np.max(Yfront, axis=0) + 0.01 * (
            np.max(Yfront, axis=0) - np.min(Yfront, axis=0)
        )

        indices = []
        n = samples.shape[1]
        mask = np.ones(samples.shape[0], dtype=bool)
        samples_indices = np.arange(0, samples.shape[0])

        for i in range(num_evals):
            masked_samples = samples[mask, :]
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
            new_point = samples[original_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[original_index] = False
            indices.append(original_index)

            # Append current estimate of the pareto front to sample_paretos
            samples_copy = samples_original.copy()
            samples_copy = samples_copy * self.output_std + self.output_mean
            samples_copy[("hvi", "DATA")] = hv_improvement
            self.samples.append(samples_copy)

        if len(hv_improvement) == 0:
            hv_imp = 0
        elif len(indices) == 0:
            indices = []
            hv_imp = 0
        else:
            # Total hypervolume improvement
            # Includes all points added to batch (hvY + last hv_improvement)
            # Subtracts hypervolume without any points added (hvY0)
            hv_imp = hv_improvement[masked_index] + hvY - hvY0
        return hv_imp, indices


def rmse(Y_pred, Y_true, mean, std):
    Y_pred = Y_pred * std + mean
    Y_true = Y_true * std + mean
    square_error = (Y_pred[:, 0] - Y_true[:, 0]) ** 2
    return np.sqrt(np.mean(square_error))


class TSEMOInternalWrapper(Problem):
    """Wrapper for NSGAII internal optimisation

    Parameters
    ----------
    fp : os.PathLike
        Path to a folder containing the rffs
    domain : :class:`~summit.domain.Domain`
        Domain used for optimisation.
    Notes
    -----
    It is assumed that the inputs are scaled between 0 and 1.

    """

    def __init__(self, fp: os.PathLike, domain, n_var=None):
        import pyrff

        self.rffs = pyrff.load_rffs(fp)
        self.domain = domain
        # Number of decision variables
        if n_var is None:
            n_var = domain.num_continuous_dimensions()

        # Number of objectives
        n_obj = len(domain.output_variables)
        # Number of constraints
        n_constr = len(domain.constraints)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        # input_columns = [v.name for v in self.domain.input_variables]
        # X = DataSet(np.atleast_2d(X), columns=input_columns)
        F = np.zeros([X.shape[0], self.n_obj])
        for i in range(self.n_obj):
            F[:, i] = self.rffs[i](X)

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
