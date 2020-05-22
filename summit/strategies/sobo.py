from .base import Strategy
from .random import LHS
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet

import GPy
import GPyOpt

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
        
class SOBO(Strategy):
    ''' Single-objective Bayesian Optimization (SOBO)
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the optimization
    gp_model_type: string, optional
        A dictionary of surrogate models or a ModelGroup to be used in the optimization.
        By default, gaussian processes with the Matern kernel will be used.
    acquisition_type: string, optional
        Whether optimization should be treated as a maximization or minimization problem.
        Defaults to maximization. 
    optimizer_type: string, optional
        The internal optimizer used in GPyOpt for maximization of the acquisition function. By default,
        lfbgs will be used if there is a combination of continuous, discrete and/or descriptors variables.
        If there is a single descriptors variable, then all of the potential values of the descriptors
        will be evaluated.
    evaluator_type: string, optional

    Notes
    ----------
    This implementation uses the python package GPyOpt provided by
    the Machine Learning Group of the University of Sheffield.

    Copyright (c) 2016, the GPyOpt Authors.
    All rights reserved.

    Github repository: https://github.com/SheffieldML/GPyOpt
    Homepage: http://sheffieldml.github.io/GPyOpt/

    Please cite their work, when using this strategy.

    Examples
    --------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import SOBO
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = SOBO(domain, random_state=np.random.RandomState(3))
    >>> result = strategy.suggest_experiments(5

    '''

    def __init__(self, domain, gp_model_type=None, acquisition_type=None, optimizer_type=None, evaluator_type=None, **kwargs):
        Strategy.__init__(self, domain)

        # TODO: notation - discrete in our model (e.g., catalyst type) = categorical?
        # TODO: decrypt levels of discrete/categorical inputs
        self.input_domain = []
        for v in self.domain.variables:
            if not v.is_objective:
                print(v.variable_type)
                if v.variable_type == 'continuous':
                    self.input_domain.append(
                        {'name': v.name,
                        'type': v.variable_type,
                        'domain': (v.bounds[0], v.bounds[1])})
                elif v.variable_type == 'discrete':
                    self.input_domain.append(
                        {'name': v.name,
                        'type': 'discrete',
                        'domain': tuple(v.levels)})
                # TODO: GPyOpt currently does not support mixed-domains w/ bandit inputs, there is a PR for this though
                # Do we need descriptors/bandit variables here? We can use categorical...
                elif v.variable_type == 'descriptors':
                    '''
                    self.input_domain.append({'name': v.name,
                     'type': 'bandit',
                     'domain': [tuple(t) for t in v.ds.data_to_numpy().tolist()]})
                    '''
                    self.input_domain.append({'name': v.name,
                                            'type': 'categorical',
                                            'domain': tuple(np.arange(v.ds.data_to_numpy().shape[0]).tolist())})
                else:
                    raise TypeError('Unknown variable type.')

        print(self.input_domain)

        #self.input_domain = [{'name': v.name,
        #                      'type': v.variable_type if v.variable_type == 'continuous' else 'categorial',
        #                      'domain': (v.bounds[0], v.bounds[1]) if v.variable_type == 'continuous' else (ind_l for ind_l, l in enumerate(v.levels))}
        #                    for v in self.domain.variables if not v.is_objective and not (v.variable_type == 'descriptor')]

        # TODO: how to handle equality constraints? Could we remove '==' from constraint types as each equality
        #  constraint reduces the degrees of freedom?
        if self.domain.constraints is not None:
            constraints = self.constr_wrapper(self.domain)
            self.constraints = [{'name': 'constr_' + str(i),
                                 'constraint': c[0] if c[1] in ['<=', '<'] else '(' + c[0] + ')*(-1)'}
                                for i,c in enumerate(constraints) if not (c[1] == '==')]
        else:
            self.constraints = None

        self.maximize = None
        for v in self.domain.variables:
            if v.is_objective:
                if self.maximize is not None:
                    raise ValueError("Single-objective Bayesian Operation strategy only operates "
                                     "on domains with one objective.")
                self.maximize = v.maximize
        if self.maximize is None:
            raise ValueError("No objective is defined. Please add an objective to the domain.")

        self.input_dim = self.domain.num_continuous_dimensions() + self.domain.num_discrete_variables()

        """
        Gaussian Process (GP) model 
            GP: standard Gaussian Process
            GP_MCMC: Gaussian Process with prior in hyperparameters
            sparseGP: sparse Gaussian Process
            warpedGP: warped Gaussian Process
            InputWarpedGP: input warped Gaussian Process
            RF: random forest (scikit-learn)
        """

        if gp_model_type in ['GP', 'GP_MCMC', 'sparseGP', 'warpedGP', 'InputWarpedGP', 'RF']:
            self.gp_model_type = gp_model_type
        else:
            self.gp_model_type = 'GP'

        """
        Acquisition function type
            EI: expected improvement
            EI_MCMC: integrated expected improvement (requires GP_MCMC model) (https://dash.harvard.edu/bitstream/handle/1/11708816/snoek-bayesopt-nips-2012.pdf?sequence%3D1)
            LCB: lower confidence bound
            LCB_MCMC:  integrated GP-Lower confidence bound (requires GP_MCMC model)
            MPI: maximum probability of improvement
            MPI_MCMC: maximum probability of improvement (requires GP_MCMC model)
            LP: local penalization
            ES: entropy search 
        """
        if acquisition_type in ['EI', 'EI_MCMC', 'LCB', 'LCB_MCMC', 'MPI', 'MPI_MCMC', 'LP', 'ES']:
            self.acquisition = acquisition_type
        else:
            self.acquisition_type = 'EI'  # default acquisition function is expected utility improvement

        """ 
        Method for optimization of acquisition function
           lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shanno,
           DIRECT: Dividing Rectangles,
           CMA: covariance matrix adaption
        """
        if optimizer_type in ['lbfgs', 'DIRECT', 'CMA']:
            self.optimizer_type = optimizer_type   # Optimizer in GPyOpt
        else:
            self.optimizer_type = 'lbfgs'   # default optimizer: lbfgs

        if evaluator_type in ['sequential', 'random', 'local_penalization', 'thompson_sampling']:
            self.evaluator_type = evaluator_type
        else:
            self.evaluator_type = 'random'


        self.kernel = kwargs.get('kernel', GPy.kern.Matern52(self.input_dim))
        self.exact_feval = kwargs.get('exact_feval', False)
        self.ARD = kwargs.get('ARD', True)




    def suggest_experiments(self, num_experiments, 
                            prev_res: DataSet=None, prev_param=None):
        """ Suggest experiments using GPyOpt single-objective Bayesian Optimization
        
        Parameters
        ----------  
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        previous_results: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, then latin hypercube sampling will
            be used to suggest an initial design.
        
        Returns
        -------
        ds
            A `Dataset` object with points to be evaluated next
        """

        param = None
        xbest = None
        fbest = float("inf")

        # Suggest random initial design
        if prev_res is None:
            '''lhs design does not consider constraints
            lhs = LHS(self.domain)
            next_experiments = lhs.suggest_experiments((num_experiments))
            return next_experiments, None, float("inf"), None
            '''
            feasible_region = GPyOpt.Design_space(space=self.input_domain, constraints=self.constraints)
            request = GPyOpt.experiment_design.initial_design('random', feasible_region, num_experiments)


        else:
            # Get inputs and outputs
            inputs, outputs = self.get_inputs_outputs(prev_res)

            inputs = inputs.to_numpy()
            outputs = outputs.to_numpy()

            if prev_param is not None:
                X_step = prev_param[0]
                Y_step = prev_param[1]

                X_step = np.vstack((X_step, inputs))
                Y_step = np.vstack((Y_step, outputs))

            else:
                X_step = inputs
                Y_step = outputs

            sobo_model = GPyOpt.methods.BayesianOptimization(f=None,
                                                             domain=self.input_domain,
                                                             constraints=self.constraints,
                                                             model_type=self.gp_model_type,
                                                             kernel=self.kernel,
                                                             acquisition_type=self.acquisition_type,
                                                             acquisition_optimizer_type=self.optimizer_type,
                                                             batch_size=num_experiments,
                                                             evaluator_type=self.evaluator_type,
                                                             maximize=self.maximize,
                                                             ARD=self.ARD,
                                                             exact_feval=self.exact_feval,
                                                             X=X_step,
                                                             Y=Y_step)
            request = sobo_model.suggest_next_locations()

            # Store parameters (history of suggested points and function evaluations)
            param = [X_step, Y_step]

            fbest = np.min(Y_step)
            xbest = X_step[np.argmin(Y_step)]


        # Generate DataSet object with variable values of next
        next_experiments = None
        if request is not None and len(request)!=0:
            next_experiments = {}
            i_inp = 0
            for v in self.domain.variables:
                if not v.is_objective:
                    next_experiments[v.name] = request[:, i_inp]
                    i_inp += 1
            next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))


        return next_experiments, xbest, fbest, param


    def constr_wrapper(self, summit_domain):
        v_input_names = [v.name for v in summit_domain.variables if not v.is_objective]
        gpyopt_constraints = []
        for c in summit_domain.constraints:
            tmp_c = c.lhs
            for v_input_index, v_input_name in enumerate(v_input_names):
                v_gpyopt_name = 'x[:,'+str(v_input_index)+']'
                tmp_c = tmp_c.replace(v_input_name, v_gpyopt_name)
            gpyopt_constraints.append([tmp_c, c.constraint_type])
        return gpyopt_constraints
