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
        The Summit domain describing the optimization problem.
    gp_model_type: string, optional
        The GPy Gaussian Process model type.
        By default, gaussian processes with the Matern 5.2 kernel will be used.
    acquisition_type: string, optional
        The acquisition function type from GPyOpt.
        By default, Excpected Improvement (EI).
    optimizer_type: string, optional
        The internal optimizer used in GPyOpt for maximization of the acquisition function.
        By default, lfbgs will be used.
    evaluator_type: string, optional
        The evaluator type used for batch mode (how multiple points are chosen in one iteration).
        By default, thompson sampling will be used.
    kernel: GPy kernel object, optional
        The kernel used in the GP.
        By default a Matern 5.2 kernel (GPy object) will be used.
    exact_feval: boolean, optional
        Whether the function evaluations are exact (True) or noisy (False).
        By default: False.
    ard: boolean, optional
        Whether automatic relevance determination should be applied (True).
        By default: True.
    standardize_outputs: boolean, optional
        Whether the outputs should be standardized (True).
        By default: True.

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
    >>> domain += DiscreteVariable(name='flowrate_a', description='flow of reactant a in mL/min', levels=[1,2,3,4,5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = SOBO(domain)
    >>> result, xbest, fbest, param = strategy.suggest_experiments(5)

    '''

    def __init__(self, domain, gp_model_type=None, acquisition_type=None, optimizer_type=None, evaluator_type=None, **kwargs):
        Strategy.__init__(self, domain)

        # TODO: notation - discrete in our model (e.g., catalyst type) = categorical?
        self.input_domain = []
        for v in self.domain.variables:
            if not v.is_objective:
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
                elif v.variable_type == 'descriptors':
                    self.input_domain.append({'name': v.name,
                     'type': 'bandit',
                     'domain': [tuple(t) for t in v.ds.data_to_numpy().tolist()]})

                    ''' possible workaround for mixed-type variable problems: treat descriptor as categorical variables
                    self.input_domain.append({'name': v.name,
                                            'type': 'categorical',
                                            'domain': tuple(np.arange(v.ds.data_to_numpy().shape[0]).tolist())})
                    '''
                else:
                    raise TypeError('Unknown variable type.')

        # TODO: how to handle equality constraints? Could we remove '==' from constraint types as each equality
        #  constraint reduces the degrees of freedom?
        if self.domain.constraints is not None:
            constraints = self.constr_wrapper(self.domain)
            self.constraints = [{'name': 'constr_' + str(i),
                                 'constraint': c[0] if c[1] in ['<=', '<'] else '(' + c[0] + ')*(-1)'}
                                for i,c in enumerate(constraints) if not (c[1] == '==')]
        else:
            self.constraints = None

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
            self.gp_model_type = 'GP'   # default model type is a standard Gaussian Process (from GPy package)

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
            self.acquisition_type = acquisition_type
        else:
            self.acquisition_type = 'EI'  # default acquisition function is expected utility improvement

        """ 
        Method for optimization of acquisition function
           lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shanno,
           DIRECT: Dividing Rectangles,
           CMA: covariance matrix adaption
        """
        if optimizer_type in ['lbfgs', 'DIRECT', 'CMA']:
            self.optimizer_type = optimizer_type
        else:
            self.optimizer_type = 'lbfgs'   # default optimizer: lbfgs

        if evaluator_type in ['sequential', 'random', 'local_penalization', 'thompson_sampling']:
            self.evaluator_type = evaluator_type
        else:
            self.evaluator_type = 'random'

        # specify GPy kernel: # https://gpy.readthedocs.io/en/deploy/GPy.kern.html#subpackages
        self.kernel = kwargs.get('kernel', GPy.kern.Matern52(self.input_dim))   
        # Are function values exact (w/o noise)?
        self.exact_feval = kwargs.get('exact_feval', False)
        # automatic relevance determination
        self.ARD = kwargs.get('ARD', True)
        # Standardization of outputs?
        self.standardize_outputs = kwargs.get('standardize_outputs', True)


    def suggest_experiments(self, num_experiments, 
                            prev_res: DataSet=None, prev_param=None):
        """ Suggest experiments using GPyOpt single-objective Bayesian Optimization
        
        Parameters
        ----------  
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments of previous iteration.
            If no data is passed, then random sampling will
            be used to suggest an initial design.
        prev_param: array-like, optional TODO: how to handle this?
            File with results from previous iterations of SOBO algorithm.
            If no data is passed, only results from prev_res will be used.
        
        Returns
        -------
        next_experiments
            A `Dataset` object with points to be evaluated next
        xbest
            Best point from all iterations.
        fbest
            Objective value at best point from all iterations.
        param
            A list containing all evaluated X and corresponding Y values.
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
            inputs, outputs = self.transform.transform_inputs_outputs(prev_res)

            # Set up maximization and minimization by converting maximization to minimization problem
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]

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
                                                             normalize_Y=self.standardize_outputs,
                                                             batch_size=num_experiments,
                                                             evaluator_type=self.evaluator_type,
                                                             maximize=False,
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
            next_experiments[('strategy', 'METADATA')] = 'Single-objective BayOpt'

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
