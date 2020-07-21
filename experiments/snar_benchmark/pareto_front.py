from summit.benchmarks import SnarBenchmark
from summit.strategies import Random
import platypus as pp
from pymoo.model.problem import Problem
from summit.utils.dataset import DataSet
from scipy.optimize import OptimizeResult
import numpy as np
import json

def determine_pareto_front(n_points=5000, random_seed=100):
    exp = SnarBenchmark()
    rand = Random(exp.domain, 
                  random_state=np.random.RandomState(random_seed))
    experiments = rand.suggest_experiments(n_points)
    exp.run_experiments(experiments)
    return exp

class NSGAII():
    def __init__(self, experiment):
        self.experiment = experiment
        self.domain = self.experiment.domain

        # Set up platypus problem
        self.problem = pp.Problem(
            nvars=self.domain.num_variables(),
            nobjs=len(self.domain.output_variables),
            nconstrs=len(self.domain.constraints),
        )
        # Set maximization or minimization for each objective
        j=0
        for i, v in enumerate(self.domain.variables):
            if v.is_objective:
                direction = self.problem.MAXIMIZE if v.maximize else self.problem.MINIMIZE
                self.problem.directions[j] = direction
                j+=1
            elif v.variable_type == "continuous":
                self.problem.types[i] = pp.Real(v.lower_bound, v.upper_bound)
            elif v.variable_type == "discrete":
                # Select a subset of one of the available options
                raise NotImplementedError(
                    "The NSGAII optimizer does not work with discrete variables"
                )
                # self.problem.types[i] = pp.Subset(elements=v.levels, size=1)
            elif v.variable_type == "descriptors":
                raise NotImplementedError(
                    "The NSGAII optimizer does not work with descriptors variables"
                )
            else:
                raise TypeError(f"{v.variable_type} is not a valid variable type.")

        # Set up constraints
        self.problem.constraints[:] = [
            c.constraint_type + "0" for c in self.domain.constraints
        ]

    def optimize(self, **kwargs):
        input_columns = [v.name for v in self.domain.variables if not v.is_objective]
        output_columns = [v.name for v in self.domain.variables if v.is_objective]

        def problem_wrapper(X):
            X = DataSet(np.atleast_2d(X), columns=input_columns)
            X[("strategy", "METADATA")] = "NSGAII"
            result = self.experiment.run_experiments(X)
            if self.domain.constraints:
                constraint_res = [
                    X.eval(c.lhs, resolvers=[X]) for c in self.domain.constraints
                ]
                constraint_res = [c.tolist()[0] for c in constraint_res]

                return result[output_columns].to_numpy()[0,:], constraint_res
            else:
                return result[output_columns].to_numpy()[0,:]

        # Run optimization
        self.problem.function = problem_wrapper
        algorithm = pp.NSGAII(self.problem, population_size=1000)
        iterations = kwargs.get("iterations", 1000)
        algorithm.run(iterations)

        x = [
            [s.variables[i] for i in range(self.domain.num_variables())]
            for s in algorithm.result
            if s.feasible
        ]
        x = DataSet(x, columns=input_columns)
        y = [
            [s.objectives[i] for i in range(len(self.domain.output_variables))]
            for s in algorithm.result
            if s.feasible
        ]
        y = DataSet(y, columns=output_columns)
        return OptimizeResult(x=x, fun=y, success=True)

class DomainWrapper(Problem):
    """ Wrapper for NSGAII internal optimisation 
    
    Parameters
    ---------- 
    models : :class:`~summit.utils.models.ModelGroup`
        Model group used for optimisation
    domain : :class:`~summit.domain.Domain`
        Domain used for optimisation.
    Notes
    -----
    It is assumed that the inputs are scaled between 0 and 1.
    
    """
    def __init__(self, experiment):
        self.experiment = experiment
        self.domain = self.experiment.domain
        # Number of decision variables
        n_var = self.domain.num_continuous_dimensions()
        # Number of objectives
        n_obj = len(self.domain.output_variables)
        # Number of constraints
        n_constr = len(self.domain.constraints)
        #Lower bounds
        xl = [v.bounds[0] for v in self.domain.input_variables]
        #Upper bounds
        xu = [v.bounds[1] for v in self.domain.input_variables]
        self.input_columns = [v.name for v in self.domain.input_variables]
        self.output_columns = [v.name for v in self.domain.output_variables]

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = DataSet(np.atleast_2d(X), columns=self.input_columns)
        X[("strategy", "METADATA")] = "NSGAII"
        F = self.experiment.run_experiments(X)
        F = F[self.output_columns].data_to_numpy()
        
        # Negate objectives that  need to be maximized
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

if __name__ == "__main__":
    exp = determine_pareto_front(n_points=5000)
    d = exp.to_dict()
    with open('pareto_front_experiment.json', 'w') as f:
        json.dump(d, f)