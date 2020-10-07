from .base import Strategy, Transform
from .random import LHS
from summit.domain import Domain
from summit.utils.dataset import DataSet

import botorch
import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import numpy as np


class MTBO(Strategy):
    """Multitask Bayesian Optimisation

    This strategy enables pre-training a model with past reaction data
    in order to enable faster optimisation.

    Parameters
    ----------

    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    pretraining_data : :class:`~summit.utils.data.DataSet`
        A DataSet with pretraining data. Must contain a metadata column named "task"
        that specfies the task for all data.
    task : int
        The index of the task being optimized. Defaults to 1.

    Notes
    -----


    References
    ----------

    .. [Swersky] K. Swersky et al., in `NIPS Proceedings <http://papers.neurips.cc/paper/5086-multi-task-bayesian-optimization>`_, 2013, pp. 2004–2012.

    Examples
    --------

    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import NelderMead
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> strategy = NelderMead(domain)
    >>> next_experiments  = strategy.suggest_experiments()
    >>> print(next_experiments)
    NAME temperature flowrate_a             strategy
    TYPE        DATA       DATA             METADATA
    0          0.500      0.500  Nelder-Mead Simplex
    1          0.625      0.500  Nelder-Mead Simplex
    2          0.500      0.625  Nelder-Mead Simplex

    """

    def __init__(
        self,
        domain: Domain,
        pretraining_data,
        transform: Transform = None,
        task=1,
        **kwargs
    ):
        Strategy.__init__(self, domain, transform, **kwargs)
        self.pretraining_data = pretraining_data
        self.task = task
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            conditions = lhs.suggest_experiments(k, criterion="maximin")
            conditions[("task", "METADATA")] = self.task
            return conditions
        elif self.iterations == 1 and len(prev_res) == 1:
            lhs = LHS(self.domain)
            self.iterations += 1
            self.all_experiments = prev_res
            conditions = lhs.suggest_experiments(num_experiments, criterion="maximin")
            conditions[("task", "METADATA")] = self.task
            return conditions
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)

        # Combine pre-training and experiment data
        data = self.all_experiments.append(self.pretraining_data)

        # Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            data, transform_descriptors=True
        )
        inputs_ct, output_ct = self.transform.transform_inputs_outputs(
            self.all_experiments, transform_descriptors=True
        )  # only current task data

        # Standardize decision variables and objectives
        inputs_scaled, self.input_mean, self.input_std = self.standardize(inputs)
        output_scaled, self.output_mean, self.output_std = self.standardize(output)
        inputs_ct_scaled = (inputs_ct - self.input_mean) / self.input_std
        inputs_ct_scaled = inputs_ct_scaled.to_numpy()
        output_ct_scaled = (output_ct - self.output_mean) / self.output_std
        output_ct_scaled = output_ct_scaled.to_numpy()

        # Add column to inputs indicating task
        task_data = data["task"].to_numpy()
        task_data = np.atleast_2d(task_data).T
        inputs_scaled_task = np.append(inputs_scaled, task_data, axis=1)

        # Train model
        model = botorch.models.MultiTaskGP(
            torch.tensor(inputs_scaled_task).float(),
            torch.tensor(output_scaled).float(),
            task_feature=-1,
            output_tasks=[self.task],
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_model(mll)

        # Create acquisition function
        # Create weights for tasks, but only weight desired task
        # TODO: properly handle maximization and minimzation
        if self.domain.output_variables[0].maximize:
            fbest_scaled = output_ct_scaled.max()
            maximize = True
        else:
            fbest_scaled = output_ct_scaled.min()
            maximize = False
        ei = botorch.acquisition.ExpectedImprovement(
            model, best_f=fbest_scaled, maximize=maximize
        )

        # Optimize acquisition function
        bounds = torch.tensor([v.bounds for v in self.domain.input_variables]).float()
        bounds = (bounds.T - self.input_mean.to_numpy()) / self.input_std.to_numpy()
        results, acq_values = botorch.optim.optimize_acqf(
            acq_function=ei,
            bounds=bounds.float(),
            num_restarts=20,
            q=num_experiments,
            raw_samples=100,
        )

        #  Return result
        self.iterations += 1
        result = DataSet(
            results.detach().numpy(),
            columns=[v.name for v in self.domain.input_variables],
        )
        result = result * self.input_std.to_numpy() + self.input_mean.to_numpy()
        result = self.transform.un_transform(result, transform_descriptors=True)
        result[("strategy", "METADATA")] = "MTBO"
        result[("task", "METADATA")] = self.task
        return result

    def reset(self):
        """Reset MTBO state"""
        self.all_experiments = None
        self.iterations = 0
        self.fbest = (
            float("inf") if self.domain.output_variables[0].maximize else -float("inf")
        )

    @staticmethod
    def standardize(X):
        mean, std = X.mean(), X.std()
        std[std < 1e-5] = 1e-5
        scaled = (X - mean.to_numpy()) / std.to_numpy()
        return scaled.to_numpy(), mean, std