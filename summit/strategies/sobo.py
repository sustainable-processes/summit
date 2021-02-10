from .base import Strategy
from .random import LHS
from summit.domain import *
from summit.utils.dataset import DataSet


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class SOBO(Strategy):
    """Single-objective Bayesian Optimization (SOBO)

    This is a general BO method since it is a wrapper around GPyOpt.

    Parameters
    ----------
    domain: :class:`~summit.domain.Domain`
        The Summit domain describing the optimization problem.
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    gp_model_type: string, optional
        The GPy Gaussian Process model type. See notes for options.
        By default, gaussian processes with the Matern 5.2 kernel will be used.
    use_descriptors : bool, optional
        Whether to use descriptors of categorical variables. Defaults to False.
    acquisition_type: string, optional
        The acquisition function type from GPyOpt. See notes for options.
        By default, Excpected Improvement (EI).
    optimizer_type: string, optional
        The internal optimizer used in GPyOpt for maximization of the acquisition function.
        By default, lfbgs will be used.
    evaluator_type: string, optional
        The evaluator type used for batch mode (how multiple points are chosen in one iteration).
        By default, thompson sampling will be used.
    kernel: :class:`~GPy.kern.kern`, optional
        The kernel used in the GP.
        By default a Matern 5.2 kernel (GPy object) will be used.
    exact_feval: boolean, optional
        Whether the function evaluations are exact (True) or noisy (False).
        By default: False.
    ARD: boolean, optional
        Whether automatic relevance determination should be applied (True).
        By default: True.
    standardize_outputs: boolean, optional
        Whether the outputs should be standardized (True).
        By default: True.

    Examples
    --------
    >>> from summit.domain import *
    >>> from summit.strategies import SOBO
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += CategoricalVariable(name='flowrate_a', description='flow of reactant a in mL/min', levels=[1,2,3,4,5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='yield', description='yield of reaction', bounds=[0,100], is_objective=True)
    >>> strategy = SOBO(domain)
    >>> next_experiments = strategy.suggest_experiments(5)

    Notes
    ----------

    Gaussian Process (GP) model
        GP: standard Gaussian Process

        GP_MCMC: Gaussian Process with prior in hyperparameters

        sparseGP: sparse Gaussian Process

        warpedGP: warped Gaussian Process

        InputWarpedGP: input warped Gaussian Process

        RF: random forest (scikit-learn)

    Acquisition function type
        EI: expected improvement

        EI_MCMC: integrated expected improvement (requires GP_MCMC model) (https://dash.harvard.edu/bitstream/handle/1/11708816/snoek-bayesopt-nips-2012.pdf?sequence%3D1)

        LCB: lower confidence bound

        LCB_MCMC:  integrated GP-Lower confidence bound (requires GP_MCMC model)

        MPI: maximum probability of improvement

        MPI_MCMC: maximum probability of improvement (requires GP_MCMC model)

        LP: local penalization

        ES: entropy search

    This implementation uses the python package GPyOpt provided by
    the Machine Learning Group of the University of Sheffield.

    Github repository: https://github.com/SheffieldML/GPyOpt


    """

    def __init__(
        self,
        domain,
        transform=None,
        gp_model_type=None,
        acquisition_type=None,
        optimizer_type=None,
        evaluator_type=None,
        **kwargs
    ):
        from GPy.kern import Matern52

        Strategy.__init__(self, domain, transform=transform, **kwargs)

        self.use_descriptors = kwargs.get("use_descriptors", False)
        # TODO: notation - discrete in our model (e.g., catalyst type) = categorical?
        self.input_domain = []
        for v in self.domain.variables:
            if not v.is_objective:
                if isinstance(v, ContinuousVariable):
                    self.input_domain.append(
                        {
                            "name": v.name,
                            "type": v.variable_type,
                            "domain": (v.bounds[0], v.bounds[1]),
                        }
                    )
                elif isinstance(v, CategoricalVariable):
                    if not self.use_descriptors:
                        self.input_domain.append(
                            {
                                "name": v.name,
                                "type": "categorical",
                                "domain": tuple(self.categorical_wrapper(v.levels)),
                            }
                        )
                    elif v.ds is not None and self.use_descriptors:
                        if v.ds is None:
                            raise ValueError(
                                "No descriptors provided for variable: {}".format(
                                    v.name
                                )
                            )
                        descriptor_names = v.ds.data_columns
                        descriptors = np.asarray(
                            [
                                v.ds.loc[:, [l]].values.tolist()
                                for l in v.ds.data_columns
                            ]
                        )
                        for j, d in enumerate(descriptors):
                            self.input_domain.append(
                                {
                                    "name": descriptor_names[j],
                                    "type": "continuous",
                                    "domain": (
                                        np.min(np.asarray(d)),
                                        np.max(np.asarray(d)),
                                    ),
                                }
                            )
                    elif v.ds is None and self.use_descriptors:
                        raise ValueError(
                            "Cannot use descriptors because none are provided."
                        )
                    # TODO: GPyOpt currently does not support mixed-domains w/ bandit inputs, there is a PR for this though
                else:
                    raise TypeError("Unknown variable type.")

        # TODO: how to handle equality constraints? Could we remove '==' from constraint types as each equality
        #  constraint reduces the degrees of freedom?
        if self.domain.constraints is not None:
            constraints = self.constr_wrapper(self.domain)
            self.constraints = [
                {
                    "name": "constr_" + str(i),
                    "constraint": c[0]
                    if c[1] in ["<=", "<"]
                    else "(" + c[0] + ")*(-1)",
                }
                for i, c in enumerate(constraints)
                if not (c[1] == "==")
            ]
        else:
            self.constraints = None

        self.input_dim = len(self.domain.input_variables)

        if gp_model_type in [
            "GP",
            "GP_MCMC",
            "sparseGP",
            "warpedGP",
            "InputWarpedGP",
            "RF",
        ]:
            self.gp_model_type = gp_model_type
        else:
            self.gp_model_type = "GP"  # default model type is a standard Gaussian Process (from GPy package)

        if acquisition_type in [
            "EI",
            "EI_MCMC",
            "LCB",
            "LCB_MCMC",
            "MPI",
            "MPI_MCMC",
            "LP",
            "ES",
        ]:
            self.acquisition_type = acquisition_type
        else:
            self.acquisition_type = (
                "EI"  # default acquisition function is expected utility improvement
            )

        """ 
        Method for optimization of acquisition function
           lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shanno,
           DIRECT: Dividing Rectangles,
           CMA: covariance matrix adaption
        """
        if optimizer_type in ["lbfgs", "DIRECT", "CMA"]:
            self.optimizer_type = optimizer_type
        else:
            self.optimizer_type = "lbfgs"  # default optimizer: lbfgs

        if evaluator_type in [
            "sequential",
            "random",
            "local_penalization",
            "thompson_sampling",
        ]:
            self.evaluator_type = evaluator_type
        else:
            self.evaluator_type = "random"

        # specify GPy kernel: # https://gpy.readthedocs.io/en/deploy/GPy.kern.html#subpackages
        self.kernel = kwargs.get("kernel", Matern52(self.input_dim))
        # Are function values exact (w/o noise)?
        self.exact_feval = kwargs.get("exact_feval", False)
        # automatic relevance determination
        self.ARD = kwargs.get("ARD", True)
        # Standardization of outputs?
        self.standardize_outputs = kwargs.get("standardize_outputs", True)
        self.prev_param = None

    def suggest_experiments(
        self, num_experiments=1, prev_res: DataSet = None, **kwargs
    ):
        """Suggest experiments using GPyOpt single-objective Bayesian Optimization

        Parameters
        ----------
        num_experiments: int, optional
            The number of experiments (i.e., samples) to generate. Default is 1.
        prev_res: :class:`~summit.utils.data.DataSet`, optional
            Dataset with data from previous experiments of previous iteration.
            If no data is passed, then random sampling will
            be used to suggest an initial design.

        Returns
        -------
        next_experiments : :class:`~summit.utils.data.DataSet`
            A Dataset object with the suggested experiments

        """
        import GPyOpt

        param = None
        xbest = np.zeros(self.domain.num_continuous_dimensions())
        obj = self.domain.output_variables[0]
        objective_dir = -1.0 if obj.maximize else 1.0
        fbest = float("inf")

        # Suggest random initial design
        if prev_res is None:
            """lhs design does not consider constraints
            lhs = LHS(self.domain)
            next_experiments = lhs.suggest_experiments((num_experiments))
            return next_experiments, None, float("inf"), None
            """
            feasible_region = GPyOpt.Design_space(
                space=self.input_domain, constraints=self.constraints
            )
            request = GPyOpt.experiment_design.initial_design(
                "random", feasible_region, num_experiments
            )
        else:
            # Get inputs and outputs
            inputs, outputs = self.transform.transform_inputs_outputs(
                prev_res,
                cateogrical_method="descriptors" if self.use_descriptors else None,
            )

            # Set up maximization and minimization by converting maximization to minimization problem
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]
                if isinstance(v, CategoricalVariable):
                    if not self.use_descriptors:
                        inputs[v.name] = self.categorical_wrapper(
                            inputs[v.name], v.levels
                        )

            inputs = inputs.to_numpy()
            outputs = outputs.to_numpy()

            if self.prev_param is not None:
                X_step = self.prev_param[0]
                Y_step = self.prev_param[1]

                X_step = np.vstack((X_step, inputs))
                Y_step = np.vstack((Y_step, outputs))

            else:
                X_step = inputs
                Y_step = outputs

            sobo_model = GPyOpt.methods.BayesianOptimization(
                f=None,
                domain=self.input_domain,
                constraints=self.constraints,
                model_type=self.gp_model_type,
                kernel=self.kernel,
                acquisition_type=self.acquisition_type,
                acquisition_optimizer_type=self.optimizer_type,
                normalize_Y=self.standardize_outputs,
                batch_size=num_experiments,
                evaluator_type=self.evaluator_type,
                maximize=False,
                ARD=self.ARD,
                exact_feval=self.exact_feval,
                X=X_step,
                Y=Y_step,
            )
            request = sobo_model.suggest_next_locations()

            # Store parameters (history of suggested points and function evaluations)
            param = [X_step, Y_step]

            fbest = np.min(Y_step)
            xbest = X_step[np.argmin(Y_step)]

        # Generate DataSet object with variable values of next
        next_experiments = None
        transform_descriptors = False
        if request is not None and len(request) != 0:
            next_experiments = {}
            i_inp = 0
            for v in self.domain.variables:
                if not v.is_objective:
                    if isinstance(v, CategoricalVariable):
                        if v.ds is None or not self.use_descriptors:
                            cat_list = []
                            for j, entry in enumerate(request[:, i_inp]):
                                cat_list.append(
                                    self.categorical_unwrap(entry, v.levels)
                                )
                            next_experiments[v.name] = np.asarray(cat_list)
                            i_inp += 1
                        else:
                            descriptor_names = v.ds.data_columns
                            for d in descriptor_names:
                                next_experiments[d] = request[:, i_inp]
                                i_inp += 1
                            transform_descriptors = True
                    else:
                        next_experiments[v.name] = request[:, i_inp]
                        i_inp += 1
            next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
            next_experiments[("strategy", "METADATA")] = "Single-objective BayOpt"

        self.fbest = objective_dir * fbest
        self.xbest = xbest
        self.prev_param = param

        # Do any necessary transformation back
        next_experiments = self.transform.un_transform(
            next_experiments,
            cateogrical_method="descriptors" if self.use_descriptors else None,
        )

        return next_experiments

    def reset(self):
        """Reset the internal parameters"""
        self.prev_param = None

    def to_dict(self):
        if self.prev_param is not None:
            param = [self.prev_param[0].tolist(), self.prev_param[1].tolist()]
        else:
            param = None

        strategy_params = dict(
            prev_param=param,
            use_descriptors=self.use_descriptors,
            gp_model_type=self.gp_model_type,
            acquisition_type=self.acquisition_type,
            optimizer_type=self.optimizer_type,
            evaluator_type=self.evaluator_type,
            kernel=self.kernel.to_dict(),
            exact_feval=self.exact_feval,
            ARD=self.ARD,
            standardize_outputs=self.standardize_outputs,
        )

        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        from GPy.kern import Kern

        # Get kernel
        kernel = d["strategy_params"].get("kernel")
        if kernel is not None:
            kernel = Kern.from_dict(kernel)
            d["strategy_params"]["kernel"] = kernel

        # Setup SOBO
        sobo = super().from_dict(d)
        param = d["strategy_params"]["prev_param"]
        if param is not None:
            param = [np.array(param[0]), np.array(param[1])]
            sobo.prev_param = param
        return sobo

    def constr_wrapper(self, summit_domain):
        v_input_names = [v.name for v in summit_domain.variables if not v.is_objective]
        gpyopt_constraints = []
        for c in summit_domain.constraints:
            tmp_c = c.lhs
            for v_input_index, v_input_name in enumerate(v_input_names):
                v_gpyopt_name = "x[:," + str(v_input_index) + "]"
                tmp_c = tmp_c.replace(v_input_name, v_gpyopt_name)
            gpyopt_constraints.append([tmp_c, c.constraint_type])
        return gpyopt_constraints

    def categorical_wrapper(self, categories, reference_categories=None):
        if not reference_categories:
            return [i for i, _ in enumerate(categories)]
        else:
            return [reference_categories.index(c) for c in categories]

    def categorical_unwrap(self, gpyopt_level, categories):
        return categories[int(gpyopt_level)]
