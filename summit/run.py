from summit.strategies import Strategy, strategy_from_dict
from summit.experiment import Experiment
from summit.benchmarks import *
from summit.utils.multiobjective import pareto_efficient, HvI

from fastprogress.fastprogress import progress_bar

import os
import json
import pkg_resources


class Runner:
    """  Run a closed-loop strategy and experiment cycle
    
    Parameters
    ---------- 
    strategy: `summit.strategies.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment: `summit.experiment.Experiment`, optional
        The experiment class to use for running experiments. If None,
        the ExternalExperiment class will be used, which assumes that
        data from each experimental run will be added as a keyword
        argument to the `run` method.
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is None.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments.

    Examples
    --------    
    
    """

    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        max_iterations=100,
        batch_size=1,
    ):
        self.strategy = strategy
        self.experiment = experiment
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def run(self, **kwargs):
        """  Run the closed loop experiment cycle
        """
        prev_res = None
        i = 0
        for i in progress_bar(range(self.max_iterations)):
            next_experiments = self.strategy.suggest_experiments(
                num_experiments=self.batch_size, prev_res=prev_res
            )
            prev_res = self.experiment.run_experiments(next_experiments)

    def to_dict(self,):
        runner_params = dict(
            max_iterations=self.max_iterations, batch_size=self.batch_size
        )

        return dict(
            runner=runner_params,
            strategy=self.strategy.to_dict(),
            experiment=self.experiment.to_dict(),
        )

    @classmethod
    def from_dict(cls, d):
        strategy = strategy_from_dict(d["strategy"])
        experiment = experiment_from_dict(d["experiment"])
        return cls(
            strategy=strategy,
            experiment=experiment,
            max_iterations=d["runner"]["max_iterations"],
            batch_size=d["runner"]["batch_size"],
        )

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


class NeptuneRunner(Runner):
    """  Run a closed-loop strategy and experiment cycle
    
    Parameters
    ---------- 
    strategy : `summit.strategies.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : `summit.experiment.Experiment`, optional
        The experiment class to use for running experiments. If None,
        the ExternalExperiment class will be used, which assumes that
        data from each experimental run will be added as a keyword
        argument to the `run` method.
    neptune_project : str
        The name of the Neptune project to log data to
    neptune_experiment_name : str
        A name for the neptune experiment
    netpune_description : str, optional
        A description of the neptune experiment
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is 100.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments. Default is 1.

    Examples
    --------    
    
    """
    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        neptune_project: str,
        neptune_experiment_name: str,
        neptune_description: str = None,
        max_iterations=100,
        batch_size=1,
        hypervolume_ref=None,
    ):

        super().__init__(strategy, experiment, 
                         max_iterations=max_iterations, 
                         batch_size=batch_size)

        # Hypervolume reference for multiobjective experiments
        n_objs = len(self.experiment.domain.output_variables)
        self.ref =  hypervolume_ref if hypervolume_ref is not None else n_objs*[0]

        # Check that Neptune-client is installed
        installed = {pkg.key for pkg in pkg_resources.working_set}
        if "neptune-client" in installed:
            from neptune.sessions import Session, HostedNeptuneBackend
        else:
            raise RuntimeError(
                "Neptune-client not installed. Use pip install summit[extras] to add extra dependencies."
            )

        # Set up Neptune session
        self.session = Session(backend=HostedNeptuneBackend())
        self.proj = self.session.get_project(neptune_project)
        self.neptune_exp = self.proj.create_experiment(
            name=neptune_experiment_name,
            description=neptune_description,
            params=self.to_dict(),
        )

    def run(self, **kwargs):
        """  Run the closed loop experiment cycle

        Parameters
        ----------
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to not saving locally.
        """
        prev_res = None
        save_freq = kwargs.get('save_freq')
        save_dir = kwargs.get('save_dir')
        for i in progress_bar(range(self.max_iterations)):
            next_experiments = self.strategy.suggest_experiments(
                num_experiments=self.batch_size, prev_res=prev_res
            )
            prev_res = self.experiment.run_experiments(next_experiments)
            
            #Send best objective values
            for v in self.experiment.domain.output_variables:
                if v.maximize:
                    fbest = self.experiment.data[v.name].max()
                elif not v.maximize:
                    fbest = self.experiment.data[v.name].min()
                
                self.neptune_exp.send_metric(v.name+"_best", fbest)
            
            # Send hypervoluem for multiobjective experiments
            n_objs = len(self.experiment.domain.output_variables)
            if n_objs >1:
                output_names = [v.name for v in self.experiment.domain.output_variables]
                data = self.experiment.data[output_names].to_numpy()
                for j, v in enumerate(self.experiment.domain.output_variables):
                    if v.maximize:
                        data[:, j] = -1.0*data[:, j]
                y_pareto, _ = pareto_efficient(data, maximize=False) 
                hv = HvI.hypervolume(y_pareto, self.ref)
                self.neptune_exp.send_metric('hypervolume', hv)
            
            # Save state
            if save_freq is not None:
                file = f'iteration_{i}.json'
                if save_dir is not None:
                    file = save_dir + "/" + file
                if i % save_freq == 0:
                    self.save(file)
                    self.neptune_exp.send_artifact(file)
                if not save_dir:
                    os.remove(file)

    def to_dict(self,):
        runner_params = dict(
            max_iterations=self.max_iterations, batch_size=self.batch_size, hypervolume_ref=self.ref,
        
        )
        
        return dict(
            runner=runner_params,
            strategy=self.strategy.to_dict(),
            experiment=self.experiment.to_dict(),
        )

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError("From dict does not work on NeptuneRunner becuase Neptune cannot be serialized.")
        
def experiment_from_dict(d):
    if d["name"] == "SnarBenchmark":
        return SnarBenchmark.from_dict(d)
    elif d["name"] == "Hartmann3D":
        return Hartmann3D.from_dict(d)
    elif d["name"] == "Himmelblau":
        return Himmelblau.from_dict(d)
