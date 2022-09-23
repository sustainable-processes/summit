import pdb
from summit.strategies.factorial_doe import fullfact
from .base import Strategy, Transform
from .random import LHS
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.utils.thompson_sampling import ThompsonSampledModel
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from scipy import optimize
from fastprogress.fastprogress import progress_bar
import numpy as np
from typing import Callable, Dict, List, Tuple


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
        ## Suggest lhs initial design or append new experiments to previous experiments
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

        n_experiments_expected = np.prod(self.levels_dict.values())
        if prev_res is None and num_experiments != n_experiments_expected:
            raise ValueError(
                f"Number of experiments expected to be {n_experiments_expected}."
            )

        ## Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            data,
            categorical_method=self.categorical_method,
            standardize_inputs=True,
            standardize_outputs=True,
        )

        ## Train and sample model
        models = [None] * num_experiments
        for j in range(num_experiments):
            model = ThompsonSampledModel(f"test_model_{j}")
            model.fit(
                inputs,
                output,
                n_retries=10,
                # CHANGE BACK TO 1500
                n_spectral_points=50,
            )
            models[j] = model

        ## Optimize Thompson sampled functions using a GA
        X, y = self._optimize(models)
        result = X.join(y)

        ## Finish up
        #  Convert result to datset
        result = DataSet(
            result,
            columns=inputs.data_columns,
        )

        # Untransform
        result = self.transform.un_transform(
            result,
            categorical_method=self.categorical_method,
            standardize_inputs=True,
            standardize_outputs=True,
        )

        # Add metadata
        result[("strategy", "METADATA")] = "CBBO"
        return result

    def _optimize(self, models):
        """Pymoo optimization"""
        ##  Set up problem
        # Only continuous
        if (
            self.domain.num_continuous_dimensions() > 0
            and self.domain.get_categorical_combinations().shape[0] == 0
        ):
            optimizer = PatternSearch()
        # Mixed domain
        elif (
            self.domain.num_continuous_dimensions() > 0
            and self.domain.get_categorical_combinations().shape[0] > 0
        ):
            optimizer = MixedVariableGA(pop_size=10)
        problem = CBBOInternalWrapper(
            models=models,
            domain=self.domain,
            levels_dict=self.levels_dict,
            transform=self.transform,
            categorical_method=self.categorical_method,
        )
        termination = get_termination("n_gen", 100)

        # Optimize
        self.internal_res = minimize(
            problem, optimizer, termination, seed=1, verbose=False
        )

        # Extract batch design
        X = np.atleast_2d(self.internal_res.X)
        levels_list = [self.levels_dict[v.name] for v in self.domain.input_variables]
        self.design_ind = fullfact(levels_list).astype(int)
        design = np.zeros_like(self.design_ind, dtype=float)
        for i, v in enumerate(self.domain.input_variables):
            n_levels = self.levels_dict[v.name]
            xi = [X[f"{v.name}_{j}"] for j in range(n_levels)]
            design[:, i] = xi[self.design_ind[:, i]]

        # Transform into model space
        design = DataSet(design, columns=self.domain.input_variables)
        X_transformed, _ = self.transform.transform_inputs_outputs(
            design,
            categorical_method=self.categorical_method,
            standardize_inputs=True,
            standardize_outputs=True,
            use_existing=True,
        )

        # Evaluate models
        y = [model.predict(design_i) for model, design_i in zip(models, X_transformed)]

        return X_transformed, y

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


class CBBOInternalWrapper(ElementwiseProblem):
    """Wrapper for Pymoo internal optimization

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

    def __init__(
        self,
        models: List[ThompsonSampledModel],
        domain: Domain,
        levels_dict: Dict[str, int],
        transform: Transform,
        categorical_method: str,
        **kwargs,
    ):
        self.models = models
        self.domain = domain
        self.summit_transform = transform
        self.levels_dict = levels_dict
        self.categorical_method = categorical_method

        # Set up design
        levels_list = [levels_dict[v.name] for v in self.domain.input_variables]
        self.design_ind = fullfact(levels_list).astype(int)

        # Create variables
        pymoo_vars = {}
        for v in self.domain.input_variables:
            n_levels = self.levels_dict[v.name]
            if isinstance(v, ContinuousVariable):
                for j in range(n_levels):
                    pymoo_vars[f"{v.name}_{j}"] = Real(bounds=tuple(v.bounds))
            elif isinstance(v, CategoricalVariable):
                for j in range(n_levels):
                    pymoo_vars[f"{v.name}_{j}"] = Choice(options=v.levels)

        # Objective direction
        self.maximize = True if self.domain.output_variables[0].maximize else False
        # Number of constraints
        # n_constr = len(domain.constraints)
        super().__init__(
            vars=pymoo_vars, n_obj=1, elementwise_evaluation=True, **kwargs
        )

    def _evaluate(self, X, out, *args, **kwargs):

        # Extract batch design
        design = [[0] * self.design_ind.shape[1]] * self.design_ind.shape[0]
        design = np.zeros_like(self.design_ind, dtype=object)
        for i, v in enumerate(self.domain.input_variables):
            n_levels = self.levels_dict[v.name]
            xi = np.array([X[f"{v.name}_{j}"] for j in range(n_levels)])
            designs_col_i = xi[self.design_ind[:, i]]
            for j, design_col_ij in enumerate(designs_col_i):
                design[j][i] = design_col_ij

        # Transform into model space
        design = DataSet(design, columns=[v.name for v in self.domain.input_variables])

        design_transformed, _ = self.summit_transform.transform_inputs_outputs(
            design,
            categorical_method=self.categorical_method,
            standardize_inputs=True,
            standardize_outputs=True,
            use_existing=True,
        )

        # Evaluate the models
        F = 0
        for i, model in enumerate(self.models):
            F += model.predict(design_transformed.loc[0, :])

        # Negate if needs to be maximized
        if self.maximize:
            F *= -1
        out["F"] = F

        # Add constraints if necessary
        # if self.domain.constraints:
        #     constraint_res = [
        #         X.eval(c.lhs, resolvers=[X]) for c in self.domain.constraints
        #     ]
        #     out["G"] = [c.tolist()[0] for c in constraint_res]
