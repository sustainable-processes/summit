from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet
from summit import get_summit_config_path

import numpy as np
import pandas as pd

from gryffin import Gryffin
import json

import os
import copy
import pickle
import uuid
import pathlib
        
class GRYFFIN(Strategy):
    """ Gryffin strategy
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The Summit domain describing the optimization problem.
    auto_desc_gen: Boolean, optional
        Whether Dynamic Gryffin is used if descriptors are provided,
        i.e., Gryffin applies automatic descriptor generation,
        hence transforms the given descriptors with a non-linear transformation
        to new descriptors (more "meaningful" or higher-correlated ones).
        By default: False (i.e., Static Gryffin with originally given descriptors is used).
    sampling_strategies: int, optional
        Number of sampling strategies (similar to sampling of GPs).
        One factor (next to batches) for the number of suggested new points in one optimization step.
        Total number of suggested points: sampling_strategies x batches
        By default: 4
    batches: int, optional
        Number of suggested points within one sampling strategy.
        One factor (next to sampling_strategies) for the number of suggested new points in one optimization step.
        Total number of suggested points: sampling_strategies x batches
        By default: 1
    logging: -1, optional
        Corresponds to the verbosity level of logging of Gryffin
	    verbosity_levels = -1: '', 0: ['INFO', 'FATAL'], 1: ['INFO', 'ERROR', 'FATAL'], 2: ['INFO', 'WARNING', 'ERROR', 'FATAL'], 3: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']
        By default: -1
    parallel: Boolean, optional
        Whether ... (no information found)
        By default: True (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    boosted: Boolean, optional
        Whether "pseudo-boosting" is applied (see original Paper).
        By default: True (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    sampler: string, optional
        A priori distribution of categorical variables.
        By default: 'uniform' (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    softness: float, optional
        ... (no information found)
        By default: 0.001 (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    continuous_optimizer: string, optional
        Optimizer type for continuous variables (available: "adam").
        By default: 'adam' (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    categorical_optimizer: string, optional
        Optimizer type for categorical variables (available: "naive").
        By default: naive (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)
    discrete_optimizer: string, optional
        Optimizer type for discrete variables ((available: "naive").
        By default: naive (cf. default settings in git repo of Gryffin at gryffin/src/gryffin/utilities/defaults.py)

    Notes
    ----------
    This implementation uses the software package Gryffin provided by
    the Aspuru-Guzik Group and published by Haese et al. (2020), arXiv:2003.12127.

    Copyright (C) 2020, Harvard University.
    All rights reserved.

    Github repository: https://github.com/aspuru-guzik-group/gryffin

    Please cite their work, when using this strategy.

    Examples
    --------
    >>> from summit.domain import *
    >>> from summit.strategies import GRYFFIN
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name="temperature", description="reaction temperature in celsius", bounds=[50, 100])
    >>> domain += CategoricalVariable(name="flowrate_a", description="flow of reactant a in mL/min", levels=[1,2,3,4,5])
    >>> base_df = DataSet([[1,2,3],[2,3,4],[8,8,8]], index = ["solv1","solv2","solv3"], columns=["MP","mol_weight","area"])
    >>> domain += DescriptorsVariable(name="solvent", description="solvent type - categorical", ds=base_df)
    >>> domain += ContinuousVariable(name="yield", description="yield of reaction", bounds=[0,100], is_objective=True)
    >>> strategy = GRYFFIN(domain, auto_desc_gen=True)
    >>> next_experiments = strategy.suggest_experiments()

    """

    def __init__(self, domain, transform=None, save_dir=None, auto_desc_gen=False, sampling_strategies=4,
                 batches=1, logging=-1, parallel=True, boosted=True, sampler="uniform", softness=0.001,
                 continuous_optimizer="adam", categorical_optimizer="naive", discrete_optimizer="naive", **kwargs):
        Strategy.__init__(self, domain, transform=transform)

        self.domain_inputs = []
        self.domain_objectives = []
        self.prev_param = None

        # Create directories to store temporary files
        summit_config_path = get_summit_config_path()
        self.uuid_val = uuid.uuid4()  # Unique identifier for this run
        tmp_dir = summit_config_path / "gryffin" / str(self.uuid_val)
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

        # Parse Summit domain to Gryffin domain
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
                    descriptors = [v.ds.loc[[l], :].values[0].tolist() for l in v.ds.index] if v.ds is not None else None
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": "categorical",
                            "size": 1,
                            "levels": v.levels,
                            "descriptors": descriptors,
                            "category_details": str(tmp_dir / "CatDetails" / f"cat_details_{v.name}.pkl"),
                        }
                    )
                elif v.variable_type == "descriptors":
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": "categorical",
                            "size": 1,
                            "levels": [l for l in v.ds.index],
                            "descriptors": [v.ds.loc[[l],:].values[0].tolist() for l in v.ds.index],
                            "category_details": str(tmp_dir / "CatDetails" / f"cat_details_{v.name}.pkl"),
                        }
                    )
                else:
                    raise TypeError("Unknown variable type: {}.".format(v.variable_type))
            else:
                self.domain_objectives.append(
                    {
                        "name": v.name,
                        "goal": "minimize",
                    }
                )

        # TODO: how does GRYFFIN handle constraints?
        if self.domain.constraints != []:
            raise NotImplementedError("Gryffin can not handle constraints yet.")
            # keep SOBO constraint wrapping for later application when gryffin adds constraint handling
            #constraints = self.constr_wrapper(self.domain)
            #self.constraints = [{"name": "constr_" + str(i),
            #                     "constraint": c[0] if c[1] in ["<=", "<"] else "(" + c[0] + ")*(-1)"}
            #                    for i,c in enumerate(constraints) if not (c[1] == "==")]
        else:
            self.constraints = None

        # create a temporary config.json file to initialize GRYFFIN
        config_dict = {
            "general": {
                "auto_desc_gen": auto_desc_gen,
                "parallel": parallel,
                "boosted": boosted,
                "sampling_strategies": sampling_strategies,
                "batches": batches,
                "scratch_dir": str(tmp_dir / "scratch"),
                "sampler": sampler,
                "softness": softness,
                "continuous_optimizer": continuous_optimizer,
                "categorical_optimizer": categorical_optimizer,
                "discrete_optimizer": discrete_optimizer,
                'verbosity': {
                    'default': logging,
                    'bayesian_network': logging,
                    'random_sampler': logging,
                }
            },
            "database": {
                'format': 'pickle',
                "path": str(tmp_dir / "SearchProgress"),
            },
            "parameters": self.domain_inputs,
            "objectives": self.domain_objectives,
        }

        config_file = "config.json"
        config_file_path = tmp_dir / config_file
        with open(config_file_path, 'w') as configfile:
            json.dump(config_dict, configfile, indent=2)

        # write categories
        category_writer = CategoryWriter(inputs=self.domain_inputs)
        category_writer.write_categories(save_dir=tmp_dir)

        # initialize gryffin
        self.gryffin = Gryffin(config_file_path)


    def suggest_experiments(self,
                            prev_res: DataSet=None, **kwargs):
        """ Suggest experiments using Gryffin optimization strategy
        
        Parameters
        ----------  
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments of previous iteration.
            If no data is passed, then random sampling will
            be used to suggest an initial design.
        
        Returns
        -------
        next_experiments
            A `Dataset` object with points to be evaluated next

        Notes
        -------
        xbest, internal state
            Best point from all iterations.
        fbest, internal state
            Objective value at best point from all iterations.
        param, internal state
            A list containing all evaluated X and corresponding Y values.

        """

        param = None
        xbest = np.zeros(self.domain.num_continuous_dimensions())
        obj = self.domain.output_variables[0]
        fbest = float("inf")


        # Suggest random initial design
        if prev_res is None:
            request = self.gryffin.recommend(observations = [])
        else:
            # Get inputs and outputs
            inputs, outputs = self.transform.transform_inputs_outputs(prev_res)

            # Set up maximization and minimization by converting maximization to minimization problem
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]

            inputs_dict = inputs.to_dict(orient='records')
            outputs_dict = outputs.to_dict(orient='records')
            prev_samples = [{**{k1[0]: [v1] for k1, v1 in inputs_dict[i].items()}, **{k2[0]: v2 for k2, v2 in outputs_dict[i].items()}} for i in range(len(inputs_dict))]

            observations = []
            if self.prev_param is not None:
                observations = self.prev_param
            observations.extend(prev_samples)
            param = observations

            request = self.gryffin.recommend(observations=observations)

            for obs in observations:
                if obs[obj.name] < fbest:
                    fbest = obs[obj.name]
                    xbest = np.asarray([v[0] for k, v in obs.items() if k!=obj.name])


        # Generate DataSet object with variable values of next
        next_experiments = None
        if request is not None and len(request)!=0:
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
        return next_experiments

    def reset(self):
        """Reset the internal parameters"""
        self.prev_param = None

    def to_dict(self):
        if self.prev_param is not None:
            param = self.prev_param
            strategy_params = dict(prev_param=param)
        else:
            strategy_params = dict(prev_param=None)
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        gryffin = super().from_dict(d)
        param = d["strategy_params"]["prev_param"]
        if param is not None:
            gryffin.prev_param = param
        return gryffin

    # TODO: update constraint wrapper when Gryffin can handle constraints
    ''' 
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
    '''

class CategoryWriter(object):
    """ Category Writer for Gryffin (adapted from https://github.com/aspuru-guzik-group/gryffin)

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
        self.cat_inputs = [[ent["name"], ent["levels"], ent["descriptors"]] for ent in inputs if ent["type"] == "categorical"]

    def write_categories(self, save_dir):
        """ Writes categories to pkl file
        :param save_dir: string, path where category details will be saved.
        """

        for cat_inp in self.cat_inputs:
            param_name = cat_inp[0]
            param_opt = cat_inp[1]
            param_descr = cat_inp[2]

            opt_list = []
            for opt in range(len(param_opt)):
                #TODO: descriptors all the same?
                if param_descr is not None:
                    descriptors = np.array(param_descr[opt])
                    opt_dict = {'name': param_opt[opt], 'descriptors': descriptors}
                else:
                    opt_dict = {'name': param_opt[opt]}
                opt_list.append(copy.deepcopy(opt_dict))

            # create cat_details dir if necessary
            if not os.path.isdir('%s/CatDetails' % save_dir):
                os.mkdir('%s/CatDetails' % save_dir)

            cat_details_file = '%s/CatDetails/cat_details_%s.pkl' % (save_dir, param_name)
            pickle.dump(opt_list, open(cat_details_file, 'wb'))

