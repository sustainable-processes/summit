from .base import Strategy, Transform
from .random import LHS
from summit.domain import Domain, DomainError
from summit.utils.multiobjective import pareto_efficient, HvI
from summit.utils.optimizers import NSGAII
from summit.utils.models import ModelGroup, GPyModel
from summit.utils.dataset import DataSet

import platypus as pp
import numpy as np
from abc import ABC, abstractmethod
import logging


        
class TSEMO(Strategy):
    """ Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO)
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the optimization
    transform: `summit.strategies.base.Transform`, optional
        A transform class (i.e, not the object itself). By default
        no transformation will be done the input variables or
        objectives.
    models: a dictionary of summit.utils.model.Model or a summit.utils.model.ModelGroup, optional
        A dictionary of surrogate models or a ModelGroup to be used in the optimization.
        By default, gaussian processes with the Matern kernel will be used.
    maximize: bool, optional
        Whether optimization should be treated as a maximization or minimization problem.
        Defaults to maximization. 
    optimizer: summit.utils.Optimizer, optional
        The internal optimizer for estimating the pareto front prior to maximization
        of the acquisition function. By default, NSGAII will be used if there is a combination
        of continuous, discrete and/or descriptors variables. If there is a single descriptors 
        variable, then all of the potential values of the descriptors will be evaluated.
    random_rate: float, optional
        The rate of random exploration. This must be a float between 0 and 1.
        Default is 0.25
    reference: array-like, optional
        The reference used for hypervolume calculations. Should be an array of length of the number 
        of objectives.  Defaults to 0 for all objectives.


    Examples
    --------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import TSEMO
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = TSEMO(domain, random_state=np.random.RandomState(3))
    >>> result = strategy.suggest_experiments(5)
 
    """

    def __init__(self, domain, transform=None, models=None, **kwargs):
        Strategy.__init__(self, domain, transform)

        # Internal models
        if models is None:
            input_dim = (
                self.domain.num_continuous_dimensions()
                + self.domain.num_discrete_variables()
            )
            models = {
                v.name: GPyModel(input_dim=input_dim)
                for v in self.domain.variables
                if v.is_objective
            }
            self.models = ModelGroup(models)
        elif isinstance(models, ModelGroup):
            self.models = models
        elif isinstance(models, dict):
            self.models = ModelGroup(models)
        else:
            raise TypeError("models must be a ModelGroup or a dictionary of models.")

        # NSGAII internal optimizer
        self.optimizer = NSGAII(self.domain)

        self._reference = kwargs.get(
            "reference", [0 for v in self.domain.variables if v.is_objective]
        )
        self._random_rate = kwargs.get("random_rate", 0.25)
        if self._random_rate < 0.0 or self._random_rate > 1.0:
            raise ValueError("Random rate must be between 0 and 1.")

        self.logger = kwargs.get('logger', logging.getLogger(__name__))

        self.reset()

    def suggest_experiments(self, num_experiments, 
                            prev_res: DataSet = None, **kwargs):
        """ Suggest experiments using TSEMO
        
        Parameters
        ----------  
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, then latin hypercube sampling will
            be used to suggest an initial design. 
        n_spectral_points: int
            Number of spectral points used in spectral sampling.
            Default is 1500.
            
        Returns
        -------
        ds
            A `Dataset` object with the random design
        """
        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            return lhs.suggest_experiments(num_experiments)
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)

        #Get inputs (decision variables) and outputs (objectives)
        inputs, outputs = self.transform.transform_inputs_outputs(self.all_experiments)
        if inputs.shape[0] < self.domain.num_continuous_dimensions():
            raise ValueError(f'The number of examples ({inputs.shape[0]}) is less the number of input dimensions ({self.domain.num_continuous_dimensions()}. Add more examples, for example, using a LHS.')
        
        # Scale decision variables [0,1]
        inputs_scaled, input_min, input_max = inputs.zero_to_one(return_min_max=True)

        # Standardize objectives
        outputs_scaled, output_mean, output_std = outputs.standardize(return_mean=True, return_std=True)
        output_names = [v.name for v in self.domain.output_variables]
        outputs_scaled = DataSet(outputs_scaled, columns=output_names)

        # Fit models to data
        self.logger.info(f"Fitting {len(self.models._models)} models.")
        self.models.fit(inputs_scaled, outputs_scaled, spectral_sample=False, **kwargs)

        # Spectral sampling
        n_spectral_points = kwargs.get('n_spectral_points', 1500)
        for name, model in self.models.models.items():
            self.logger.info(f"Spectral sampling for model {name}.")
            model.spectral_sample(inputs_scaled, outputs_scaled,
                                  n_spectral_points=n_spectral_points)

        # Make optimizer scale to internal transformation instead of domain
        for i in range(self.domain.num_continuous_dimensions()):
            self.optimizer.problem.types[i] = pp.Real(0,1)
        self.logger.info("Optimizing models using NSGAII.")
        internal_res = self.optimizer.optimize(self.models, use_spectral_sample=True, **kwargs)
        
        if internal_res is not None and len(internal_res.fun) != 0:
            # Select points that give maximum hypervolume improvement
            hv_imp, indices = self.select_max_hvi(
                outputs_scaled, internal_res.fun, num_experiments
            )
            
            # Unscale data
            X = internal_res.x*(input_max-input_min) + input_min
            y = internal_res.fun*output_std+output_mean

            # Join to get single dataset with inputs and outputs
            result = X.join(y)
            result = result.iloc[indices, :]

            # Do any necessary transformations back
            result = self.transform.un_transform(result)

            # State the strategy used
            result[("strategy", "METADATA")] = "TSEMO"

            # Add model hyperparameters as metadata columns
            self.iterations += 1
            for name,model in self.models.models.items():
                lengthscales, var, noise = model.hyperparameters
                result[(f'{name}_variance' ,'METADATA')] = var
                result[(f'{name}_noise' ,'METADATA')] = noise
                for var, l in zip(self.domain.input_variables, lengthscales):
                    result[(f'{name}_{var.name}_lengthscale' ,'METADATA')] = l
                result[('iterations', 'METADATA')] = self.iterations
            return result
        else:
            self.iterations += 1
            return None

    def reset(self):
        self.all_experiments = None
        self.iterations = 0

    def to_dict(self):
        ae = (
            self.all_experiments.to_dict() if self.all_experiments is not None else None
        )
        strategy_params = dict(
            models=self.models.to_dict(),
            random_rate=self._random_rate,
            all_experiments=ae,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        d["strategy_params"]["models"] = ModelGroup.from_dict(
            d["strategy_params"]["models"]
        )
        tsemo = super().from_dict(d)
        ae = d["strategy_params"]["all_experiments"]
        if ae is not None:
            tsemo.all_experiments = DataSet.from_dict(ae)
        return tsemo

    def select_max_hvi(self, y, samples, num_evals=1):
        """  Returns the point(s) that maximimize hypervolume improvement 
        
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
        
        """
        samples = samples.copy()
        y = y.copy()

        # Set up maximization and minimization
        for v in self.domain.variables:
            if v.is_objective and v.maximize:
                y[v.name] = -1 * y[v.name]
                samples[v.name] = -1 * samples[v.name]

        # samples, mean, std = samples.standardize(return_mean=True, return_std=True)
        samples = samples.data_to_numpy()
        Ynew = y.data_to_numpy()
        # Ynew = (Ynew - mean)/std

        #Reference
        Yfront, _ = pareto_efficient(Ynew, maximize=False)
        r = np.max(Yfront, axis=0)+0.01*(np.max(Yfront, axis=0)-np.min(Yfront,axis=0))
        
        index = []
        n = samples.shape[1]
        mask = np.ones(samples.shape[0], dtype=bool)

        # Set up random selection
        if not (self._random_rate <= 1.0) | (self._random_rate >= 0.0):
            raise ValueError("Random Rate must be between 0 and 1.")

        if self._random_rate > 0:
            num_random = round(self._random_rate * num_evals)
            random_selects = np.random.randint(0, num_evals, size=num_random)
        else:
            random_selects = np.array([])
        
        for i in range(num_evals):
            masked_samples = samples[mask, :]
            Yfront, _ = pareto_efficient(Ynew, maximize=False)
            if len(Yfront) == 0:
                raise ValueError("Pareto front length too short")

            hv_improvement = []
            hvY = HvI.hypervolume(Yfront, r)
            #Determine hypervolume improvement by including
            #each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1, n)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = pareto_efficient(A, maximize=False)
                hv = HvI.hypervolume(Afront, r)
                hv_improvement.append(hv-hvY)
            
            hvY0 = hvY if i==0 else hvY0

            if i in random_selects:
                masked_index = np.random.randint(0, masked_samples.shape[0])
            else:
                # Choose the point that maximizes hypervolume improvement
                masked_index = hv_improvement.index(max(hv_improvement))

            samples_index = np.where(
                (samples == masked_samples[masked_index, :]).all(axis=1)
            )[0][0]
            new_point = samples[samples_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[samples_index] = False
            index.append(samples_index)

        if len(hv_improvement) == 0:
            hv_imp = 0
        elif len(index) == 0:
            index = []
            hv_imp = 0
        else:
            # Total hypervolume improvement
            # Includes all points added to batch (hvY + last hv_improvement)
            # Subtracts hypervolume without any points added (hvY0)
            hv_imp = hv_improvement[masked_index] + hvY - hvY0
        return hv_imp, index
