from .base import Strategy, Transform
from .random import LHS
from summit.domain import Domain, DomainError
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit.utils.models import ModelGroup, GPyModel
from summit.utils.dataset import DataSet
from summit import get_summit_config_path

from GPy.models import GPRegression as gpr
import GPy
import pyrff

from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.factory import get_termination

import pathlib
import os
import numpy as np
import uuid
from abc import ABC, abstractmethod
import logging
import warnings


class TSEMO(Strategy):
    """ Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO)
    
    Parameters
    ---------- 
    domain : :class:~summit.domain.Domain
        The domain of the optimization
    transform : :class:~summit.strategies.base.Transform, optional
        A transform object. By default
        no transformation will be done the input variables or
        objectives.
    kernel : :class:~GPy.kern.Kern, optional
        A GPy kernel class (not instantiated). Must be Exponential,
        Matern32, Matern52 or RBF. Default Exponential.

    Examples
    --------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import TSEMO
    >>> from summit.utils.dataset import DataSet
    >>> import numpy as np
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    >>> strategy = TSEMO(domain)
    >>> result = strategy.suggest_experiments(5)
 
    """

    def __init__(self, domain, transform=None, **kwargs):
        Strategy.__init__(self, domain, transform)
        
        # Bounds
        self.inputs_min = DataSet([[v.bounds[0] for v in self.domain.input_variables]],
                                  columns=[v.name for v in self.domain.input_variables])
        self.inputs_max = DataSet([[v.bounds[1] for v in self.domain.input_variables]],
                                  columns=[v.name for v in self.domain.input_variables])

        # Kernel
        self.kernel = kwargs.get("kernel", GPy.kern.Exponential)
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        """ Suggest experiments using TSEMO
        
        Parameters
        ----------  
        num_experiments : int
            The number of experiments (i.e., samples) to generate
        prev_res : summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, then latin hypercube sampling will
            be used to suggest an initial design. 
        n_spectral_points : int, optional
            Number of spectral points used in spectral sampling.
            Default is 1500.
        generations : int, optional
            Number of generations used in the internal optimisation with NSGAII.
            Default is 100.
        pop_size : int, optional
            Population size used in the internal optimisation with NSGAII.
            Default is 100.
        Returns
        -------
        ds
            A `Dataset` object with the random design
        """
        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None :
            lhs = LHS(self.domain)
            self.iterations +=1
            return lhs.suggest_experiments(num_experiments)
        elif (self.iterations == 1 and len(prev_res)==1):
            lhs = LHS(self.domain)
            self.iterations +=1
            self.all_experiments = prev_res
            return lhs.suggest_experiments(num_experiments)
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)

        
        # Get inputs (decision variables) and outputs (objectives)
        inputs, outputs = self.transform.transform_inputs_outputs(self.all_experiments)
        if inputs.shape[0] < self.domain.num_continuous_dimensions():
            self.logger.warning(
                f"The number of examples ({inputs.shape[0]}) is less the number of input dimensions ({self.domain.num_continuous_dimensions()}."
            )

        # Scale decision variables [0,1]
        inputs_scaled = (inputs-self.inputs_min.to_numpy())/(self.inputs_max.to_numpy()-self.inputs_min.to_numpy())

        # Standardize objectives
        self.output_mean = outputs.mean()
        self.output_std = outputs.std()
        outputs_scaled = (outputs-self.output_mean.to_numpy())/self.output_std.to_numpy()

        # Set up models
        input_dim = self.domain.num_continuous_dimensions()
        self.models = {v.name: gpr(inputs_scaled.to_numpy(), 
                                   outputs_scaled[[v.name]].to_numpy(),
                                   kernel=self.kernel(input_dim=input_dim, ARD=True)
                              )
                       for v in self.domain.output_variables}
        
        rmse_train = np.zeros(2)
        rmse_train_spectral = np.zeros(2)
        rffs = [None for  _ in range(len(self.domain.output_variables))]
        i = 0
        num_restarts=kwargs.get("num_restarts", 100)
        self.logger.debug(f"Fitting models (number of optimization restarts={num_restarts})\n")
        for name, model in self.models.items():
            # Constrain hyperparameters
            model.kern.lengthscale.constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3),warning=False)
            model.kern.lengthscale.set_prior(GPy.priors.LogGaussian(0, 10), warning=False)
            model.kern.variance.constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3), warning=False)
            model.kern.variance.set_prior(GPy.priors.LogGaussian(-6, 10), warning=False)
            model.Gaussian_noise.constrain_bounded(np.exp(-6), 1, warning=False)
            
            # Train model
            model.optimize_restarts( 
                num_restarts=num_restarts,
                max_iters=kwargs.get("max_iters", 10000),
                parallel=kwargs.get("parallel", True),
                verbose=False)
            
            # self.logger.info model hyperparameters
            lengthscale=model.kern.lengthscale.values
            variance=model.kern.variance.values[0]
            noise=model.Gaussian_noise.variance.values[0]
            self.logger.debug(f"Model {name} lengthscales: {lengthscale}")
            self.logger.debug(f"Model {name} variance: {variance}")
            self.logger.debug(f"Model {name} noise: {noise}")

            # Model validation
            rmse_train[i] = rmse(model.predict(inputs_scaled.to_numpy())[0], 
                                 outputs_scaled[[name]].to_numpy(),
                                 mean=self.output_mean[name].values[0], std=self.output_std[name].values[0])
            self.logger.debug(f"RMSE train {name} = {rmse_train[i].round(2)}")
        
            # Spectral sampling
            if type(model.kern) == GPy.kern.Exponential:
                matern_nu = 1
            elif type(model.kern) == GPy.kern.Matern32:
                matern_nu = 3
            elif type(model.kern) == GPy.kern.Matern52:
                matern_nu = 5
            elif type(model.kern) == GPy.kern.RBF:
                matern_nu = np.inf
            else:
                raise TypeError("Spectral sample currently only works with Matern type kernels, including RBF.")
            
            n_spectral_points = kwargs.get('n_spectral_points', 1500)
            n_retries = kwargs.get('n_retries',10)
            self.logger.debug(f"Spectral sampling {name} with {n_spectral_points} spectral points.")
            for _ in range(n_retries):
                try:
                    rffs[i] = pyrff.sample_rff(
                        lengthscales=lengthscale,
                        scaling=np.sqrt(variance),
                        noise=noise,
                        kernel_nu=matern_nu,
                        X=inputs_scaled.to_numpy(),
                        Y=outputs_scaled[[name]].to_numpy()[:,0],
                        M=n_spectral_points
                )
                    break
                except np.linalg.LinAlgError as e:
                    self.logger.error(e)
                except ValueError as e:
                    self.logger.error(e)
            if rffs[i] is None:
                raise RuntimeError(f"Spectral sampling failed after {n_retries} retries.")

            sample_f = lambda x: np.atleast_2d(rffs[i](x)).T

            rmse_train_spectral[i] = rmse(sample_f(inputs_scaled.to_numpy()), 
                                          outputs_scaled[[name]].to_numpy(),
                                          mean=self.output_mean[name].values[0], 
                                          std=self.output_std[name].values[0])
            self.logger.debug(f"RMSE train spectral {name} = {rmse_train_spectral[i].round(2)}")
            
            i+=1
            
        # Save spectral samples
        dp_results = get_summit_config_path() / 'tsemo' / str(self.uuid_val)
        os.makedirs(dp_results, exist_ok=True)
        pyrff.save_rffs(rffs, pathlib.Path(dp_results, 'models.h5'))

        # NSGAII internal optimisation
        generations = kwargs.get("generations", 100)
        pop_size = kwargs.get("pop_size", 100)
        self.logger.info("Optimizing models using NSGAII.")
        optimizer = NSGA2(pop_size=pop_size)
        problem = TSEMOInternalWrapper(pathlib.Path(dp_results, 'models.h5'),
                                       self.domain)
        termination = get_termination("n_gen", generations)
        self.internal_res = minimize(
            problem, optimizer, termination, seed=1, verbose=False
        )
        X = DataSet(self.internal_res.X, columns=[v.name for v in self.domain.input_variables])
        y = DataSet(self.internal_res.F, columns=[v.name for v in self.domain.output_variables])

        if X.shape[0] != 0 and y.shape[0] != 0:
            # Select points that give maximum hypervolume improvement
            self.hv_imp, indices = self.select_max_hvi(
                outputs_scaled, y, num_experiments
            )

            # Unscale data
            X = X * (self.inputs_max.to_numpy() - self.inputs_min.to_numpy()) + self.inputs_min.to_numpy()
            y = y * self.output_std.to_numpy() + self.output_mean.to_numpy()


            # Join to get single dataset with inputs and outputs
            result = X.join(y)
            result = result.iloc[indices, :]

            # Do any necessary transformations back
            result = self.transform.un_transform(result)

            # State the strategy used
            result[("strategy", "METADATA")] = "TSEMO"

            # Add model hyperparameters as metadata columns
            self.iterations += 1
            # for name, model in self.models.models.items():
            #     lengthscales, var, noise = model.hyperparameters
            #     result[(f"{name}_variance", "METADATA")] = var
            #     result[(f"{name}_noise", "METADATA")] = noise
            #     for var, l in zip(self.domain.input_variables, lengthscales):
            #         result[(f"{name}_{var.name}_lengthscale", "METADATA")] = l
            #     result[("iterations", "METADATA")] = self.iterations
            return result
        else:
            self.iterations += 1
            return None

    def reset(self):
        """Reset TSEMO state"""
        self.all_experiments = None
        self.iterations = 0
        self.samples = [] # Samples drawn using NSGA-II
        self.sample_fs = [0 for i in range(len(self.domain.output_variables))]
        self.uuid_val = uuid.uuid4()

    def to_dict(self):
        ae = (
            self.all_experiments.to_dict() if self.all_experiments is not None else None
        )
        strategy_params = dict(
            all_experiments=ae,
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
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
        samples_original = samples.copy()
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

        # Reference
        Yfront, _ = pareto_efficient(Ynew, maximize=False)
        r = np.max(Yfront, axis=0) + 0.01 * (
            np.max(Yfront, axis=0) - np.min(Yfront, axis=0)
        )

        indices = []
        n = samples.shape[1]
        mask = np.ones(samples.shape[0], dtype=bool)
        samples_indices = np.arange(0, samples.shape[0])

        for i in range(num_evals):
            masked_samples = samples[mask, :]
            Yfront, _ = pareto_efficient(Ynew, maximize=False)
            if len(Yfront) == 0:
                raise ValueError("Pareto front length too short")

            hv_improvement = []
            hvY = hypervolume(Yfront, r)
            # Determine hypervolume improvement by including
            # each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1, n)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = pareto_efficient(A, maximize=False)
                hv = hypervolume(Afront, r)
                hv_improvement.append(hv - hvY)

            hvY0 = hvY if i == 0 else hvY0
            hv_improvement = np.array(hv_improvement)
            masked_index = np.argmax(hv_improvement)

            # Housekeeping: find the max HvI point and mask out for next round
            original_index = samples_indices[mask][masked_index]
            new_point = samples[original_index, :].reshape(1, n)
            Ynew = np.append(Ynew, new_point, axis=0)
            mask[original_index] = False
            indices.append(original_index)

            # Append current estimate of the pareto front to sample_paretos
            samples_copy = samples_original.copy()
            samples_copy = samples_copy*self.output_std+self.output_mean
            samples_copy[('hvi', 'DATA')] = hv_improvement
            self.samples.append(samples_copy)

        if len(hv_improvement) == 0:
            hv_imp = 0
        elif len(indices) == 0:
            indices = []
            hv_imp = 0
        else:
            # Total hypervolume improvement
            # Includes all points added to batch (hvY + last hv_improvement)
            # Subtracts hypervolume without any points added (hvY0)
            hv_imp = hv_improvement[masked_index] + hvY - hvY0
        return hv_imp, indices

def rmse(Y_pred, Y_true, mean, std):
    Y_pred = Y_pred*std+mean
    Y_true = Y_true*std+mean
    square_error = (Y_pred[:,0]-Y_true[:,0])**2
    return np.sqrt(np.mean(square_error))

class TSEMOInternalWrapper(Problem):
    """ Wrapper for NSGAII internal optimisation 
    
    Parameters
    ---------- 
    fp : os.PathLike
        Path to a folder containing the rffs
    domain : :class:`~summit.domain.Domain`
        Domain used for optimisation.
    Notes
    -----
    It is assumed that the inputs are scaled between 0 and 1.
    
    """
    def __init__(self, fp:os.PathLike, domain):
        self.rffs =  pyrff.load_rffs(fp)
        self.domain = domain
        # Number of decision variables
        n_var = domain.num_continuous_dimensions()
        # Number of objectives
        n_obj = len(domain.output_variables)
        # Number of constraints
        n_constr = len(domain.constraints)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        # input_columns = [v.name for v in self.domain.input_variables]
        # X = DataSet(np.atleast_2d(X), columns=input_columns)
        F = np.zeros([X.shape[0], self.n_obj])
        for i in range(self.n_obj):
            F[:,i] = self.rffs[i](X)
        
        # Negate objectives that are need to be maximized
        for i, v in enumerate(self.domain.output_variables):
            if v.maximize:
                F[:,i] *= -1
        out["F"] = F

        # Add constraints if necessary
        if self.domain.constraints:
            constraint_res = [
                X.eval(c.lhs, resolvers=[X]) for c in self.domain.constraints
            ]
            out["G"] = [c.tolist()[0] for c in constraint_res]
