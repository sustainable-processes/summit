from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from gryffin import Gryffin
import json

import os
import copy
import pickle
        
class GRYFFIN(Strategy):
    """ Gryffin strategy
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The Summit domain describing the optimization problem.
    XYZ

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
    >>> from summit.domain import Domain, ContinuousVariable, DiscreteVariable
    >>> from summit.strategies import GRYFFIN
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name="temperature", description="reaction temperature in celsius", bounds=[50, 100])
    >>> domain += DiscreteVariable(name="flowrate_a", description="flow of reactant a in mL/min", levels=[1,2,3,4,5])
    >>> domain += ContinuousVariable(name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name="yield", description="yield of reaction", bounds=[0,100], is_objective=True)
    >>> strategy = GRYFFIN(domain)
    >>> next_experiments = strategy.suggest_experiments()

    """

    def __init__(self, domain, **kwargs):
        Strategy.__init__(self, domain)

        # TODO: notation - discrete in our model (e.g., catalyst type) = categorical?
        self.domain_inputs = []
        self.domain_objectives = []
        with_descriptors = False
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
                elif v.variable_type == "discrete":
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": "categorical",
                            "size": 1,
                            "levels": v.levels,
                            "descriptors": None,
                            "category_details": "CatDetails/cat_details_" + str(v.name) + ".pkl",
                        }
                    )
                elif v.variable_type == "descriptors":
                    with_descriptors = True
                    self.domain_inputs.append(
                        {
                            "name": v.name,
                            "type": "categorical",
                            "size": 1,
                            "levels": [l for l in v.ds.index],
                            "descriptors": [v.ds.loc[[l],:].values[0].tolist() for l in v.ds.index],
                            "category_details": "CatDetails/cat_details_" + str(v.name) + ".pkl",
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
            raise NotImplementedError("Gryffin can not handle constraints.")
            # keep SOBO constraint wrapping for later application when gryffin adds constraint handling
            #constraints = self.constr_wrapper(self.domain)
            #self.constraints = [{"name": "constr_" + str(i),
            #                     "constraint": c[0] if c[1] in ["<=", "<"] else "(" + c[0] + ")*(-1)"}
            #                    for i,c in enumerate(constraints) if not (c[1] == "==")]
        else:
            self.constraints = None

        self.input_dim = self.domain.num_continuous_dimensions() + self.domain.num_discrete_variables()

        self.auto_desc_gen = kwargs.get("auto_desc_gen", "False")
        self.parallel = kwargs.get("parallel", "True")
        self.boosted = kwargs.get("boosted", "True")
        self.sampling_strategies = kwargs.get("sampling_strategies", 4)
        self.batches = kwargs.get("batches", 1)
        self.logging = kwargs.get("logging", -1)
        self.prev_param = None

        # Create tmp_files
        tmp_files = os.path.join(os.path.dirname(os.path.realpath(__file__)),"tmp_files")
        tmp_dir = os.path.join(tmp_files, "gryffin")
        if not os.path.isdir(tmp_files):
            os.mkdir(tmp_files)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        # create a temporary config.json file to initialize GRYFFIN
        config_dict = {
            "general": {
                "auto_desc_gen": self.auto_desc_gen,
                "parallel": self.parallel,
                "boosted": self.boosted,
                "sampling_strategies": self.sampling_strategies,
                "batches": self.batches,
                "scratch_dir": os.path.join(tmp_dir, "scratch"),
                'verbosity': {
                    'default': self.logging,
                    'bayesian_network': self.logging,
                    'random_sampler': self.logging,
                }
            },
            "database": {
                'format': 'pickle',
                "path": os.path.join(tmp_dir, "SearchProgress"),
            },
            "parameters": self.domain_inputs,
            "objectives": self.domain_objectives,
        }

        config_file = "config.json"
        with open(config_file, 'w') as configfile:
            json.dump(config_dict, configfile, indent=2)

        # write categories
        category_writer = CategoryWriter(inputs=self.domain_inputs)
        category_writer.write_categories(home_dir='./')

        # initialize gryffin
        self.gryffin = Gryffin(config_file)


    def suggest_experiments(self,
                            prev_res: DataSet=None, **kwargs):
        """ Suggest experiments using GPyOpt single-objective Bayesian Optimization
        
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
            next_experiments[("strategy", "METADATA")] = "Gryffin"

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

    # TODO: update constraint wrapper
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


class CategoryWriter(object):

    def __init__(self, inputs):
        self.cat_inputs = [[ent["name"], ent["levels"], ent["descriptors"]] for ent in inputs if ent["type"] == "categorical"]

    def write_categories(self, home_dir, with_descriptors=True):

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
            if not os.path.isdir('%s/CatDetails' % home_dir):
                os.mkdir('%s/CatDetails' % home_dir)

            cat_details_file = '%s/CatDetails/cat_details_%s.pkl' % (home_dir, param_name)
            pickle.dump(opt_list, open(cat_details_file, 'wb'))

# =========================================================================
