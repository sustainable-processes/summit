from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.multiobjective import pareto_efficient, HvI
from summit.utils.optimizers import NSGAII
from summit.utils.models import ModelGroup
from summit.utils.dataset import DataSet

import numpy as np
from abc import ABC, abstractmethod
        
class TSEMO2(Strategy):
    ''' A modified version of Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO)
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the optimization
    models: summit.models.Model
        Any list of surrogate models to be used in the optimization
    maximize: bool, optional
        Whether optimization should be treated as a maximization or minimization problem.
        Defaults to maximization. 
    optimizer: summit.utils.Optimizer, optional
        The internal optimizer for estimating the pareto front prior to maximization
        of the acquisition function. By default, NSGAII will be used if there is a combination
        of continuous, discrete and/or descriptors variables. If there is a single descriptors 
        variable, then all of the potential values of the descriptors will be evaluated.
    
    
    Examples
    --------
    # >>> from summit.domain import Domain, ContinuousVariable, DescriptorsVariable
    # >>> import GPy
    # >>> domain += DescriptorsVariable('solvent',
    #                                  'solvents in the lab',
    #                                   solvent_ds)
    # >>> domain+= ContinuousVariable(name='yield',
    #                             description='relative conversion to triphenylphosphine oxide determined by LCMS',
    #                             bounds=[0, 100],
    #                             is_objective=True)
    # >>> domain += ContinuousVariable(name='de',
    #                             description='diastereomeric excess determined by ratio of LCMS peaks',
    #                             bounds=[0, 100],
    #                             is_objective=True)
    # >>> input_dim = domain.num_continuous_dimensions()+domain.num_discrete_variables()
    # >>> kernels = [GPy.kern.Matern52(input_dim = input_dim, ARD=True)
    #                for _ in range(2)]
    #  >>> models = [GPyModel(kernel=kernels[i]) for i in range(2)]
    # >>> tsemo = TSEMO(domain, models, acquisition=acquisition)
    # >>> prevous_result = 
    # design = tsemo.generate_experiments(previous_results, batch_size, 
    #                                     normalize_inputs=True)
 
    ''' 
    def __init__(self, domain, models, optimizer=None, **kwargs):
        Strategy.__init__(self, domain)

        if isinstance(models, ModelGroup):
            self.models = models
        elif isinstance(models, dict):
            self.models = ModelGroup(models)
        else: 
            raise TypeError('models must be a ModelGroup or a dictionary of models.')

        if not optimizer:
            self.optimizer = NSGAII(self.domain)
        else:
            self.optimizer = optimizer

        self._reference = kwargs.get('reference', [0,0])
        self._random_rate = kwargs.get('random_rate', 0.0)

    def suggest_experiments(self, previous_results: DataSet, num_experiments):
        #Get inputs and outputs
        inputs, outputs = self.get_inputs_outputs(previous_results)
        #Fit models to new data
        self.models.fit(inputs, outputs)

        internal_res = self.optimizer.optimize(self.models)
        
        if internal_res is not None and len(internal_res.fun)!=0:
            hv_imp, indices = self.select_max_hvi(outputs, internal_res.fun, num_experiments)
            result = internal_res.x.join(internal_res.fun) 
            return result.iloc[indices, :]
        else:
            return None

    def select_max_hvi(self, y, samples, num_evals=1):
        '''  Returns the point(s) that maximimize hypervolume improvement 
        
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
        
        ''' 
        #Get the reference point, r
        # r = self._reference + 0.01*(np.max(samples, axis=0)-np.min(samples, axis=0)) 
        r = self._reference

        samples = samples.copy()
        y = y.copy()
        
        #Set up maximization and minimization
        for v in self.domain.variables:
            if v.is_objective and v.maximize:
                y[v.name] = -1 * y[v.name]
                samples[v.name] = -1 * samples[v.name]
        
        # samples, mean, std = samples.standardize(return_mean=True, return_std=True)
        samples = samples.data_to_numpy()
        Ynew = y.data_to_numpy()
        # Ynew = (Ynew - mean)/std
        
        index = []
        n = samples.shape[1]
        mask = np.ones(samples.shape[0], dtype=bool)

        #Set up random selection
        if not (self._random_rate <=1.) | (self._random_rate >=0.):
            raise ValueError('Random Rate must be between 0 and 1.')

        if self._random_rate>0:
            num_random = round(self._random_rate*num_evals)
            random_selects = np.random.randint(0, num_evals, size=num_random)
        else:
            random_selects = np.array([])
        
        for i in range(num_evals):
            masked_samples = samples[mask, :]
            Yfront, _ = pareto_efficient(Ynew, maximize=False)
            if len(Yfront) == 0:
                raise ValueError('Pareto front length too short')

            hv_improvement = []
            hvY = HvI.hypervolume(Yfront, [0, 0])
            #Determine hypervolume improvement by including
            #each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1,n)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = pareto_efficient(A, maximize=False)
                hv = HvI.hypervolume(Afront, [0,0])
                hv_improvement.append(hv-hvY)
            
            hvY0 = hvY if i==0 else hvY0

            if i in random_selects:
                masked_index = np.random.randint(0, masked_samples.shape[0])
            else:
                #Choose the point that maximizes hypervolume improvement
                masked_index = hv_improvement.index(max(hv_improvement))
            
            samples_index = np.where((samples == masked_samples[masked_index, :]).all(axis=1))[0][0]
            new_point = samples[samples_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[samples_index] = False
            index.append(samples_index)

        if len(hv_improvement)==0:
            hv_imp = 0
        elif len(index) == 0:
            index = []
            hv_imp = 0
        else:
            #Total hypervolume improvement
            #Includes all points added to batch (hvY + last hv_improvement)
            #Subtracts hypervolume without any points added (hvY0)
            hv_imp = hv_improvement[masked_index] + hvY-hvY0
        return hv_imp, index

