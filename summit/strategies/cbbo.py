import pdb
from summit.strategies.factorial_doe import fullfact
from .base import Strategy, Transform
from .random import LHS
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.utils.thompson_sampling import ThompsonSampledModel
from scipy import optimize
import numpy as np
from typing import Callable, Dict, List, Tuple, Union, Optional


class CBBO(Strategy):
    """Bayesian Optimisation

    This strategy enables pre-training a model with past reaction data
    in order to enable faster optimisation.

    Parameters
    ----------

    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    pretraining_data : :class:`~summit.utils.data.DataSet`
        A DataSet with pretraining data. Must contain a metadata column named "task"
        that specfies the task for all data.
    task : int, optional
        The index of the task being optimized. Defaults to 1.
    categorical_method : str, optional
        The method for transforming categorical variables. Either
        "one-hot" or "descriptors". Descriptors must be included in the
        categorical variables for the later.

    Notes
    -----


    References
    ----------

    Examples
    --------

    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import NelderMead
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name="yld", description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> strategy = NelderMead(domain)
    >>> next_experiments  = strategy.suggest_experiments()
    >>> print(next_experiments)
    NAME temperature flowrate_a             strategy
    TYPE        DATA       DATA             METADATA
    0          0.500      0.500  Nelder-Mead Simplex
    1          0.625      0.500  Nelder-Mead Simplex
    2          0.500      0.625  Nelder-Mead Simplex

    """

    def __init__(
        self,
        domain: Domain,
        levels_dict: Dict[str, int],
        transform: Transform = None,
        categorical_method: str = "one-hot",
        **kwargs,
    ):
        Strategy.__init__(self, domain, transform, **kwargs)
        self.categorical_method = categorical_method
        if self.categorical_method not in ["one-hot", "descriptors"]:
            raise ValueError(
                "categorical_method must be one of 'one-hot' or 'descriptors'."
            )
        self.levels_dict = levels_dict
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        q = num_experiments
        if q < 2:
            raise ValueError("CBBO requires at least 2 experiments")

        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            conditions = lhs.suggest_experiments(k)
            return conditions
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)
        self.iterations += 1
        data = self.all_experiments

        # Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            data,
            categorical_method=self.categorical_method,
            standardize_inputs=True,
            standardize_outputs=True,
        )

        # Train and sample model
        samples = []
        models = []
        for j in range(q):
            model = ThompsonSampledModel("test_model_{j}")
            model.fit(
                inputs,
                output,
                n_retries=10,
                n_spectral_points=1500,
            )
            models.append(model)

        # Optimize Thompson sampled functions
        # q is batch size
        # m is the input space dimension
        levels_list = [self.levels_dict[v.name] for v in self.domain.input_variables]
        design_ind = fullfact(levels_list).astype(int)
        maximize = True if self.domain.output_variables[0].maximize else False
        restarts = 50
        x0s = np.zeros((restarts, sum(levels_list)))
        var_to_X = []
        bounds = np.zeros((sum(levels_list), 2))
        k = 0
        for v in self.domain.input_variables:
            n_levels = self.levels_dict[v.name]
            if isinstance(v, ContinuousVariable):
                b = np.array(v.bounds)
                bounds[k : k + n_levels, :] = b
            elif isinstance(v, CategoricalVariable):
                raise DomainError("Categorical variables not supported yet")
            var_to_X.append(np.arange(k, k + n_levels))
            x0s[:, k : k + n_levels] = b[0] + np.random.rand(restarts, n_levels) * (
                b[1] - b[0]
            )
            k += n_levels

        m = len(self.domain.input_variables)
        res_x, _ = multi_start_optimize(
            f_opt,
            x0s,
            func_args=(models, m, var_to_X, design_ind, maximize),
            bounds=bounds,
        )
        results = np.zeros_like(design_ind, dtype=float)
        for i in range(m):
            # xi are the levels of the ith decision variable
            xi = res_x[var_to_X[i]]
            # design_ind[:, i] is the indices ith column of the design
            results[:, i] = xi[design_ind[:, i]]

        # Convert result to datset
        result = DataSet(
            results,
            columns=inputs.data_columns,
        )

        # Untransform
        result = self.transform.un_transform(
            result, categorical_method=self.categorical_method, standardize_inputs=True
        )

        # Add metadata
        result[("strategy", "METADATA")] = "STBO"
        return result

    def _get_bounds(self):
        bounds = []
        for v in self.domain.input_variables:
            if isinstance(v, ContinuousVariable):
                mean = self.transform.input_means[v.name]
                std = self.transform.input_stds[v.name]
                v_bounds = np.array(v.bounds)
                v_bounds = (v_bounds - mean) / std
                bounds.append(v_bounds)
            elif (
                isinstance(v, CategoricalVariable)
                and self.categorical_method == "one-hot"
            ):
                bounds += [[0, 1] for _ in v.levels]
        return bounds

    def _get_levels_dict(self, init_levels_dict: Dict[str, int] = None):
        """Get the levels dictionary"""
        if init_levels_dict is None:
            levels_dict = {}
        else:
            levels_dict = init_levels_dict
        for v in self.domain.input_variables:
            if v.name not in levels_dict:
                levels_dict[v.name] = None
        return levels_dict

    def reset(self):
        """Reset MTBO state"""
        self.all_experiments = None
        self.iterations = 0


def multi_start_optimize(
    fun: Callable[[np.ndarray], float], x0s: np.ndarray, func_args, **kwargs
) -> Tuple[np.ndarray, float]:
    """
    Helper function to run fmin-optimization from many start points.
    Parameters
    ----------
    fun : callable
        the function to minimize
    x0s : numpy.ndarray
        (N_starts, D) array of initial guesses
    Returns
    -------
    x_best : numpy.ndarray
        (D,) coordinate of the found minimum
    y_best : float
        function value at the minimum

    Notes
    ------
    Copied from pyrff

    """
    x_peaks = [optimize.minimize(fun, x0=x0, args=func_args, **kwargs).x for x0 in x0s]
    y_peaks = [fun(x, *func_args) for x in x_peaks]
    ibest = np.argmin(y_peaks)
    return x_peaks[ibest], y_peaks[ibest]


def f_opt(
    X,
    models: List[ThompsonSampledModel],
    m: int,
    var_to_X: np.ndarray,
    design_ind: np.ndarray,
    maximize: bool,
):
    """Objective function for Thompson sampling

    Parameters
    ----------
    X : np.ndarray
        Decision variables
    models : list of ThompsonSampledModel
        List of models to use for Thompson sampling
    m : int
        Number of decision variables
    var_to_X : np.ndarray
        Mapping from variable index to indices in X
    design_ind : np.ndarray
        Enumeration of full factorial design
    maximize : bool
        Whether to maximize or minimize the objective.

    """
    # Generate design
    design = np.zeros_like(design_ind, dtype=float)
    for i in range(m):
        # xi are the levels of the ith decision variable
        xi = X[var_to_X[i]]
        # design_ind[:, i] is the indices ith column of the design
        design[:, i] = xi[design_ind[:, i]]

    # Calculate objecitve
    f = np.sum([model.rff(xs) for xs, model in zip(design, models)])
    if maximize:
        f *= -1.0
    return f