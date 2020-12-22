from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet
from summit import get_summit_config_path

import numpy as np
import pandas as pd

import json

import os
import copy
import pickle
import uuid
import pathlib


class GRYFFIN(Strategy):
    """Gryffin is a single objective Bayesian optimisation strategy.

    It is designed to work well with mixed domains (i.e., categorical and continuous variables).

    Parameters
    ----------
    domain: :class:`~summit.domain.Domain`
        The Summit domain describing the optimization problem.
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    use_descriptors: bool, optional
        Whether descriptors of categorical variables are used.
        If not,auto_desc_gen must be True when categorical variables are used.
        Default is True.
    auto_desc_gen: bool, optional
        Whether Dynamic Gryffin is used if descriptors are provided.
        Gryffin applies automatic descriptor generation, hence transforms the given descriptors with a non-linear transformation to new descriptors (more "meaningful" or higher-correlated ones).
        Defaults to False (i.e., Static Gryffin with originally given descriptors is used).
    sampling_strategies: int, optional
        Number of sampling strategies (similar to sampling of GPs).
        One factor (next to batches) for the number of suggested new points in one optimization step.
        Total number of suggested points: sampling_strategies x batches.
        Defaults to 4.
    batches: int, optional
        Number of suggested points within one sampling strategy.
        One factor (next to sampling_strategies) for the number of suggested new points in one optimization step.
        Total number of suggested points: sampling_strategies x batches. Defaults to 1.
    logging: -1, optional
        Corresponds to the verbosity level of logging of Gryffin. See the Notes for potential logging levels.
        Defaults to -1
    parallel: Boolean, optional
        Run optimisation in parallel. Default True.
    boosted: Boolean, optional
        Whether "pseudo-boosting" is applied See the original paper in references below for more details.
    sampler: string, optional
        A priori distribution of categorical variables. By default: 'uniform'
    softness: float, optional
        Softness of Chimera. By default: 0.001
    continuous_optimizer: string, optional
        Optimizer type for continuous variables (available: "adam").
        By default: 'adam'
    categorical_optimizer: string, optional
        Optimizer type for categorical variables (available: "naive").
        By default: naive
    discrete_optimizer: string, optional
        Optimizer type for discrete variables ((available: "naive").
        By default: naive

    Attributes
    ----------

    xbest : internal state
        Best point from all iterations.
    fbest : internal state
        Objective value at best point from all iterations.
    param : internal state
        A list containing all evaluated X and corresponding Y values.

    Examples
    --------

    >>> from summit.domain import *
    >>> from summit.strategies import GRYFFIN
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name="temperature", description="reaction temperature in celsius", bounds=[50, 100])
    >>> domain += CategoricalVariable(name="flowrate_a", description="flow of reactant a in mL/min", levels=[1,2,3,4,5])
    >>> base_df = DataSet([[1,2,3],[2,3,4],[8,8,8]], index = ["solv1","solv2","solv3"], columns=["MP","mol_weight","area"])
    >>> domain += CategoricalVariable(name="solvent", description="solvent type - categorical", descriptors=base_df)
    >>> domain += ContinuousVariable(name="yield", description="yield of reaction", bounds=[0,100], is_objective=True)
    >>> strategy = GRYFFIN(domain, auto_desc_gen=True)
    >>> next_experiments = strategy.suggest_experiments()

    Notes
    -----

    verbosity_levels:
    * -1= ''
    * 0= ['INFO', 'FATAL']
    * 1= ['INFO', 'ERROR', 'FATAL']
    * 2= ['INFO', 'WARNING', 'ERROR', 'FATAL']
    * 3= ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']

    Gryffin was created by the Aspuru-Guzik group. See the paper by [Hase]_ or the
    `Github repository <https://github.com/aspuru-guzik-group/gryffin>`_.

    References
    ----------

    .. [Hase] Häse, F., Roch, L.M. and Aspuru-Guzik, A., 2020. Gryffin: An algorithm for Bayesian
           optimization for categorical variables informed by physical intuition with applications to chemistry.
           arXiv preprint `arXiv:2003.12127 <https://arxiv.org/pdf/2003.12127.pdf>`_.

    """

    def __init__(
        self,
        domain,
        transform=None,
        use_descriptors=True,
        auto_desc_gen=False,
        sampling_strategies=4,
        batches=1,
        logging=-1,
        parallel=True,
        boosted=True,
        sampler="uniform",
        softness=0.001,
        continuous_optimizer="adam",
        categorical_optimizer="naive",
        discrete_optimizer="naive",
        **kwargs,
    ):
        kwargs.update({"categorical_method": None})
        Strategy.__init__(self, domain, transform=transform, **kwargs)

        self.domain_inputs = []
        self.domain_objectives = []
        self.prev_param = None

        tmp_dir = self._get_tmp_dir()

        # create a temporary config.json file to initialize GRYFFIN
        self.use_descriptors = use_descriptors
        config_dict = {
            "general": {
                "auto_desc_gen": auto_desc_gen,
                "parallel": parallel,
                "boosted": boosted,
                "sampling_strategies": sampling_strategies,
                "batches": batches,
                "sampler": sampler,
                "softness": softness,
                "continuous_optimizer": continuous_optimizer,
                "categorical_optimizer": categorical_optimizer,
                "discrete_optimizer": discrete_optimizer,
                "verbosity": {
                    "default": logging,
                    "bayesian_network": logging,
                    "random_sampler": logging,
                },
            }
        }

        delay_setup = kwargs.get("delay_setup", False)
        if not delay_setup:
            self._setup_gryffin(config_dict, tmp_dir)

    def suggest_experiments(self, prev_res: DataSet = None, **kwargs):
        """Suggest experiments using Gryffin optimization strategy

        Parameters
        ----------
        prev_res: :class:`~summit.utils.data.DataSet`, optional
            Dataset with data from previous experiments of previous iteration.
            If no data is passed, then random sampling will
            be used to suggest an initial design.

        Returns
        -------
        next_experiments : :class:`~summit.utils.data.DataSet`
            A Dataset object with the suggested experiments

        """

        param = None
        xbest = np.zeros(self.domain.num_continuous_dimensions())
        obj = self.domain.output_variables[0]
        fbest = float("inf")

        # Suggest random initial design
        if prev_res is None:
            request = self.gryffin.recommend(observations=[])
        else:
            # Get inputs and outputs
            inputs, outputs = self.transform.transform_inputs_outputs(
                prev_res, categorical_method=None
            )

            # Set up maximization and minimization by converting maximization to minimization problem
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]

            inputs_dict = inputs.to_dict(orient="records")
            outputs_dict = outputs.to_dict(orient="records")
            prev_samples = [
                {
                    **{k1[0]: [v1] for k1, v1 in inputs_dict[i].items()},
                    **{k2[0]: v2 for k2, v2 in outputs_dict[i].items()},
                }
                for i in range(len(inputs_dict))
            ]

            observations = []
            if self.prev_param is not None:
                observations = self.prev_param
            observations.extend(prev_samples)
            param = observations

            request = self.gryffin.recommend(observations=observations)

            for obs in observations:
                if obs[obj.name] < fbest:
                    fbest = obs[obj.name]
                    xbest = np.asarray([v[0] for k, v in obs.items() if k != obj.name])

        # Generate DataSet object with variable values of next
        next_experiments = None
        if request is not None and len(request) != 0:
            next_experiments = {}
            for k in request[0].keys():
                next_experiments[k] = [r[k][0] for r in request]
            next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
            next_experiments[("strategy", "METADATA")] = "GRYFFIN"

        obj = self.domain.output_variables[0]
        objective_dir = -1.0 if obj.maximize else 1.0
        fbest = objective_dir * fbest
        self.fbest = fbest
        self.xbest = xbest
        self.prev_param = param

        # Do any necessary transformation back
        next_experiments = self.transform.un_transform(
            next_experiments, categorical_method=None
        )

        return next_experiments

    def _create_gryffin_domain(self, tmp_dir):
        for v in self.domain.variables:
            if not v.is_objective:
                if v.variable_type == "continuous":
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": v.variable_type,
                            "low": float(v.bounds[0]),
                            "high": float(v.bounds[1]),
                            "size": 1,
                        }
                    )
                elif v.variable_type == "categorical":
                    if v.ds is not None and self.use_descriptors:
                        descriptors = [
                            v.ds.loc[[l], :].values[0].tolist() for l in v.ds.index
                        ]
                    else:
                        descriptors = None
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": "categorical",
                            "size": 1,
                            "levels": v.levels,
                            "descriptors": descriptors,
                            "category_details": str(
                                tmp_dir / "CatDetails" / f"cat_details_{v.name}.pkl"
                            ),
                        }
                    )
                else:
                    raise TypeError(
                        "Unknown variable type: {}.".format(v.variable_type)
                    )
            else:
                self.domain_objectives.append(
                    {
                        "name": v.name,
                        "goal": "minimize",
                    }
                )

        if len(self.domain_objectives) > 1:
            raise ValueError(
                "Gryffin only works with single objective problems. Use a transform for multiobjective problems"
            )

        # TODO: how does GRYFFIN handle constraints?
        if self.domain.constraints != []:
            raise NotImplementedError("Gryffin can not handle constraints yet.")
            # keep SOBO constraint wrapping for later application when gryffin adds constraint handling
            # constraints = self.constr_wrapper(self.domain)
            # self.constraints = [{"name": "constr_" + str(i),
            #                     "constraint": c[0] if c[1] in ["<=", "<"] else "(" + c[0] + ")*(-1)"}
            #                    for i,c in enumerate(constraints) if not (c[1] == "==")]
        else:
            self.constraints = None

    def _setup_gryffin(self, config_dict: dict, tmp_dir: pathlib.Path):
        # Create class attribute
        self.config_dict = copy.deepcopy(config_dict)
        self._create_gryffin_domain(tmp_dir)

        # Update paramters
        config_dict["parameters"] = self.domain_inputs
        config_dict["objectives"] = self.domain_objectives
        config_dict["general"]["scratch_dir"] = str(tmp_dir / "scratch")
        config_dict["database"] = {
            "format": "pickle",
            "path": str(tmp_dir / "SearchProgress"),
        }

        # Save config file
        config_file = "config.json"
        config_file_path = tmp_dir / config_file
        with open(config_file_path, "w") as configfile:
            json.dump(config_dict, configfile, indent=2)

        # write categories
        category_writer = CategoryWriter(inputs=self.domain_inputs)
        category_writer.write_categories(save_dir=tmp_dir)

        # initialize gryffin

        from gryffin import Gryffin

        self.gryffin = Gryffin(config_file_path)

    def _get_tmp_dir(self):
        # Create directories to store temporary files
        summit_config_path = get_summit_config_path()
        self.uuid_val = uuid.uuid4()  # Unique identifier for this run
        tmp_dir = summit_config_path / "gryffin" / str(self.uuid_val)
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    def reset(self):
        """Reset the internal parameters"""
        self.prev_param = None

    def to_dict(self):
        if self.prev_param is not None:
            param = self.prev_param
        else:
            param = None
        strategy_params = dict(
            config_dict=self.config_dict,
            use_descriptors=self.use_descriptors,
            prev_param=param,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        # Gather parameters
        strategy_params = d["strategy_params"]
        d["strategy_params"]["delay_setup"] = True
        param = strategy_params["prev_param"]

        # Setup gryffin
        gryffin = super().from_dict(d)
        tmp_dir = gryffin._get_tmp_dir()
        if strategy_params.get("config_dict") is not None:
            gryffin._setup_gryffin(strategy_params["config_dict"], tmp_dir)
        gryffin.prev_param = param
        return gryffin

    # TODO: update constraint wrapper when Gryffin can handle constraints
    """ 
    def constr_wrapper(self, summit_domain):
        v_input_names = [v.name for v in summit_domain.variables if not v.is_objective]
        gpyopt_constraints = []
        for c in summit_domain.constraints:
            tmp_c = c.lhs
            for v_input_index, v_input_name in enumerate(v_input_names):
                v_gpyopt_name = "x[:,"+str(v_input_index)+"]"
                tmp_c = tmp_c.replace(v_input_name, v_gpyopt_name)
            gpyopt_constraints.append([tmp_c, c.constraint_type])
        return gpyopt_constraints
    """


class CategoryWriter(object):
    """Category Writer for Gryffin (adapted from https://github.com/aspuru-guzik-group/gryffin)

    Parameters
    ----------
    inputs: array-like
        List containing the input variables. Each entry is a dictionary describing the features of the input variable.

    Notes
    ----------
    This implementation uses the software package Gryffin provided by
    the Aspuru-Guzik Group and published by Haese et al. (2020), arXiv:2003.12127.

    Copyright (C) 2020, Harvard University.
    All rights reserved.

    """

    def __init__(self, inputs):
        self.cat_inputs = [
            [ent["name"], ent["levels"], ent["descriptors"]]
            for ent in inputs
            if ent["type"] == "categorical"
        ]

    def write_categories(self, save_dir):
        """Writes categories to pkl file
        :param save_dir: string, path where category details will be saved.
        """

        for cat_inp in self.cat_inputs:
            param_name = cat_inp[0]
            param_opt = cat_inp[1]
            param_descr = cat_inp[2]

            opt_list = []
            for opt in range(len(param_opt)):
                # TODO: descriptors all the same?
                if param_descr is not None:
                    descriptors = np.array(param_descr[opt])
                    opt_dict = {"name": param_opt[opt], "descriptors": descriptors}
                else:
                    opt_dict = {"name": param_opt[opt]}
                opt_list.append(copy.deepcopy(opt_dict))

            # create cat_details dir if necessary
            if not os.path.isdir("%s/CatDetails" % save_dir):
                os.mkdir("%s/CatDetails" % save_dir)

            cat_details_file = "%s/CatDetails/cat_details_%s.pkl" % (
                save_dir,
                param_name,
            )
            pickle.dump(opt_list, open(cat_details_file, "wb"))
