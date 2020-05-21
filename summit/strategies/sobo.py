from .base import Strategy
from .random import LHS
from summit.domain import Domain, DomainError
from summit.utils.multiobjective import pareto_efficient, HvI
from summit.utils.optimizers import NSGAII
from summit.utils.models import ModelGroup, GPyModel
from summit.utils.dataset import DataSet

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
    models: dictionary of summit.utils.model.Model or summit.utils.model.ModelGroup, optional
        A dictionary of surrogate models or a ModelGroup to be used in the optimization.
        By default, gaussian processes with the Matern kernel will be used.
    maximize: bool, optional
        Whether optimization should be treated as a maximization or minimization problem.
        Defaults to maximization. 
    optimizer: summit.utils.Optimizer, optional
        The internal optimizer for maximization of the acquisition function. By default,
        XXX will be used if there is a combination of continuous, discrete and/or descriptors variables.
        If there is a single descriptors variable, then all of the potential values of the descriptors
        will be evaluated.


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
    >>> result = strategy.suggest_experiments(5)
 
    ''' 
    def __init__(self, domain, acquisition=None, optimizer=None, **kwargs):
        Strategy.__init__(self, domain)

        # TODO: notation - discrete in our model (e.g., catalyst type) = categorial?
        self.input_domain = [{'name': v.name,
                              'type': v.variable_type if v.variable_type == 'continuous' else 'categorial',
                              'domain': (v.bounds[0], v.bounds[1])}
                            for v in self.domain.variables if not v.is_objective and not (v.variable_type == 'descriptor')]

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
        Acquisition function type
            EI: expected improvement
            EI_MCMC: expected improvement - markov chain monte carlo (https://dash.harvard.edu/bitstream/handle/1/11708816/snoek-bayesopt-nips-2012.pdf?sequence%3D1)
            LCB: lower confidence bound
            LCB_MCMC: confidence bound - markov chain monte carlo
            MPI: maximum probability of improvement
            MPI_MCMC: maximum probability of improvement - markov chain monte carlo
            LP: local penalization
            ES: entropy search 
        """
        if acquisition in ['EI', 'EI_MCMC', 'LCB', 'LCB_MCMC', 'MPI', 'MPI_MCMC', 'LP', 'ES']:
            self.acquisition = acquisition
        else:
            self.acquisition = 'EI'  # default acquisition function is expected utility improvement

        """ 
        Method for optimization of acquisition function
           lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shanno,
           DIRECT: Dividing Rectangles,
           CMA: covariance matrix adaption
        """
        if optimizer in ['lbfgs', 'DIRECT', 'CMA']:
            self.optimizer = optimizer   # Optimizer in GPyOpt
        elif optimizer in ['NSGAII']:
            self.optimizer = NSGAII(self.domain)   # Optimizers outside GPyOpt but included in Summit
        else:
            self.optimizer = 'lbfgs'   # default optimizer:

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

            print(X_step)

            sobo_model = GPyOpt.methods.BayesianOptimization(f=None,
                                                             domain=self.input_domain,
                                                             constraints=self.constraints,
                                                             batch_size=num_experiments,
                                                             evaluator_type='local_penalization',
                                                             acquisition_type=self.acquisition,
                                                             exact_feval=False,
                                                             acquisition_optimizer_type=self.optimizer,
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
            for i, v in enumerate(self.domain.variables):
                if not v.is_objective:
                    next_experiments[v.name] = request[:, i]
            next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))


        return next_experiments, xbest, fbest, param

    def constr_wrapper(self, summit_domain):
        v_input_names = [v.name for v in summit_domain.variables if not v.is_objective]
        gpyopt_constraints = []
        for c in summit_domain.constraints:
            tmp_c = c.lhs
            for v_input_index, v_input_name in enumerate(v_input_names):
                print(v_input_name)
                print(tmp_c)
                v_gpyopt_name = 'x[:,'+str(v_input_index)+']'
                tmp_c = tmp_c.replace(v_input_name, v_gpyopt_name)
            gpyopt_constraints.append([tmp_c, c.constraint_type])
        return gpyopt_constraints
