from .base import Strategy, Transform
from .random import LHS
from summit.domain import Domain
from summit.utils.dataset import DataSet

import botorch
import torch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


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
    pretraining_data : :class:`~summit.utils.data.DataSet`, optional
        A DataSet with pretraining data. Must contain a metadata column

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
        transform: Transform = None,
        pretraining_data=None,
        **kwargs
    ):
        Strategy.__init__(self, domain, transform, **kwargs)
        self.reset()

    def suggest_experiments(self, num_experiments, prev_res: DataSet = None, **kwargs):
        # Suggest lhs initial design or append new experiments to previous experiments
        if prev_res is None:
            lhs = LHS(self.domain)
            self.iterations += 1
            k = num_experiments if num_experiments > 1 else 2
            return lhs.suggest_experiments(k, criterion="maximin")
        elif self.iterations == 1 and len(prev_res) == 1:
            lhs = LHS(self.domain)
            self.iterations += 1
            self.all_experiments = prev_res
            return lhs.suggest_experiments(num_experiments)
        elif prev_res is not None and self.all_experiments is None:
            self.all_experiments = prev_res
        elif prev_res is not None and self.all_experiments is not None:
            self.all_experiments = self.all_experiments.append(prev_res)

        # Get inputs (decision variables) and outputs (objectives)
        inputs, output = self.transform.transform_inputs_outputs(
            self.all_experiments, transform_descriptors=True
        )

        # Standardize decision variables and objectives
        inputs_scaled, self.input_mean, self.input_std = self.standardize(inputs)
        output_scaled, self.output_mean, self.output_std = self.standardize(output)

        # Train model

        model = botorch.models.SingleTaskGP(
            torch.tensor(inputs_scaled).float(), torch.tensor(output_scaled).float()
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        botorch.fit.fit_gpytorch_model(mll)

        # Optimize acquisition function
        fbest_scaled = (
            output_scaled.max()
            if self.domain.output_variables[0].maximize
            else output_scaled.min()
        )
        ei = botorch.acquisition.ExpectedImprovement(model, best_f=fbest_scaled)
        bounds = torch.tensor([v.bounds for v in self.domain.input_variables]).float().T
        results, acq_values = botorch.optim.optimize_acqf(
            acq_function=ei,
            bounds=bounds,
            num_restarts=20,
            q=1,
            raw_samples=100,
        )

        # Return result
        return DataSet(
            results.detach().numpy(),
            columns=[v.name for v in self.domain.input_variables],
        )

    def reset(self):
        """Reset MTBO state"""
        self.all_experiments = None
        self.iterations = 0

    @staticmethod
    def standardize(X):
        mean, std = X.mean(), X.std()
        std[std < 1e-5] = 1e-5
        scaled = (X - mean.to_numpy()) / std.to_numpy()
        return scaled.to_numpy(), mean, std