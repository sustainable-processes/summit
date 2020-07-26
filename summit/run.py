from summit.strategies import Strategy, strategy_from_dict
from summit.experiment import Experiment
from summit.benchmarks import *
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit import get_summit_config_path

from neptune.sessions import Session, HostedNeptuneBackend

from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import pathlib
import uuid
import json
import pkg_resources
import logging


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
    num_initial_experiments : int, optional
        Number of initial experiments to run, if different than batch size.
        Default is to start with batch size.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments.
    f_tol : float, optional
        How much difference between successive best objective values will be tolerated before stopping.
        This is generally useful for nonglobal algorithms like Nelder-Mead. Default is None.
    Examples
    --------    
    
    """

    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        num_initial_experiments=None,
        max_iterations=100,
        batch_size=1,
        f_tol = None,
        **kwargs
    ):
        self.strategy = strategy
        self.experiment = experiment
        self.n_init = num_initial_experiments
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.f_tol = f_tol

        #Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """  Run the closed loop experiment cycle

        Parameters
        ----------
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to not saving locally.
        """
        save_freq = kwargs.get('save_freq')
        save_dir = kwargs.get('save_dir', str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get('save_at_end', True)

        prev_res = None
        for i in progress_bar(range(self.max_iterations)):
            # Get experiment suggestions
            if i==0:
                k = self.n_init if self.n_init is not None else self.batch_size
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=k)
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res
                )
            prev_res = self.experiment.run_experiments(next_experiments)

            # Save state
            if save_freq is not None:
                file = save_dir / f'iteration_{i}.json'
                if i % save_freq == 0:
                    self.save(file)

            # Stop if no improvement
            if self.f_tol is not None and i >1:
                compare = np.abs(fbest-fbest_old) < self.f_tol
                if all(compare):
                    self.logger.info(f"{self.strategy.__class__.__name__} stopped after {i+1} iterations due to no improvement in the objective(s) (less than f_tol={self.f_tol}).")
                    break
            
        # Save at end
        if save_at_end:
            file = save_dir / f'iteration_{i}.json'
            self.save(file)

    def to_dict(self,):
        runner_params = dict(
            num_initial_experiments=self.n_init,
            max_iterations=self.max_iterations, 
            batch_size=self.batch_size,
            f_tol=self.f_tol
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
            **d["runner"]
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
    files : list, optional
        A list of filenames to save to Neptune
    max_iterations: int, optional
        The maximum number of iterations to run. By default this is 100.
    batch_size: int, optional
        The number experiments to request at each call of strategy.suggest_experiments. Default is 1.
    f_tol : float, optional
        How much difference between successive best objective values will be tolerated before stopping.
        This is generally useful for nonglobal algorithms like Nelder-Mead. Default is None.
    hypervolume_ref : array-like, optional
        The reference for the hypervolume calculation if it is a multiobjective problem.
        Should be an array of length the number of objectives. Default is at the origin.
    Examples
    --------    
    
    """
    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        neptune_project: str,
        neptune_experiment_name: str,
        neptune_tags: list = None,
        neptune_description: str = None,
        neptune_files: list = None,
        max_iterations=100,
        num_initial_experiments=1,
        batch_size=1,
        f_tol = None,
        hypervolume_ref=None,
        logger = None
    ):

        super().__init__(strategy, experiment,
                         num_initial_experiments=num_initial_experiments,
                         max_iterations=max_iterations, 
                         batch_size=batch_size)

        # Hypervolume reference for multiobjective experiments
        n_objs = len(self.experiment.domain.output_variables)
        self.ref =  hypervolume_ref if hypervolume_ref is not None else n_objs*[0]
        self.f_tol = f_tol

        # Check that Neptune-client is installed
        installed = {pkg.key for pkg in pkg_resources.working_set}
        if "neptune-client" in installed:
            from neptune.sessions import Session, HostedNeptuneBackend
        else:
            raise RuntimeError(
                "Neptune-client not installed. Use pip install summit[extras] to add extra dependencies."
            )

        # Set up Neptune session
        self.neptune_project = neptune_project
        self.neptune_experiment_name = neptune_experiment_name
        self.neptune_description = neptune_description
        self.neptune_files = neptune_files
        self.neptune_tags = neptune_tags

        #Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """  Run the closed loop experiment cycle

        Parameters
        ----------
        num_initial_experiments : int, optional
            Number of initial experiments to request before iterative experimentation.
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to `~/.summit/runner`.
        delete_local_files : bool, optional
            Delete the local files once they are uploaded to Neptune.
            Defaults to True.
        """

        # Set parameters
        prev_res = None
        n_objs = len(self.experiment.domain.output_variables)
        fbest_old = np.zeros(n_objs)
        fbest = np.zeros(n_objs)
        save_freq = kwargs.get('save_freq')
        save_dir = kwargs.get('save_dir', str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get('save_at_end', True)
        delete_local_files = kwargs.get('delete_local_files', True)

        # Create neptune experiment
        session = Session(backend=HostedNeptuneBackend())
        proj = session.get_project(self.neptune_project)
        neptune_exp = proj.create_experiment(
            name=self.neptune_experiment_name,
            description=self.neptune_description,
            params=self.to_dict(),
            upload_source_files=self.neptune_files,
            logger=self.logger,
            tags=self.neptune_tags
        )

        # Run optimization loop
        for i in progress_bar(range(self.max_iterations)):
            # Get experiment suggestions
            if i==0:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.n_init)
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res
                )

            #Run experiment suggestions
            prev_res = self.experiment.run_experiments(next_experiments)
            
            #Send best objective values to Neptune
            for j, v in enumerate(self.experiment.domain.output_variables):
                if i > 0:
                    fbest_old[j] = fbest[j]
                if v.maximize:
                    fbest[j] = self.experiment.data[v.name].max()
                elif not v.maximize:
                    fbest[j] = self.experiment.data[v.name].min()
                
                neptune_exp.send_metric(v.name+"_best", fbest[j])
            
            # Send hypervolume for multiobjective experiments
            if n_objs>1:
                output_names = [v.name for v in self.experiment.domain.output_variables]
                data = self.experiment.data[output_names]
                for v in self.experiment.domain.output_variables:
                    if v.maximize:
                        data[(v.name, 'DATA')] = -1.0*data[v.name]
                y_pareto, _ = pareto_efficient(data.to_numpy(), maximize=False) 
                hv = hypervolume(y_pareto, self.ref)
                neptune_exp.send_metric('hypervolume', hv)
            
            # Save state
            if save_freq is not None:
                file = save_dir / f'iteration_{i}.json'
                if i % save_freq == 0:
                    self.save(file)
                    neptune_exp.send_artifact(file)
                if not save_dir:
                    os.remove(file)
            
            # Stop if no improvement
            if self.f_tol is not None and i >1:
                compare = np.abs(fbest-fbest_old) < self.f_tol
                if all(compare):
                    self.logger.info(f"{self.strategy.__class__.__name__} stopped after {i+1} iterations due to no improvement in the objective(s) (less than f_tol={self.f_tol}).")
                    break
        
        # Save at end
        if save_at_end:
            file = save_dir / f'iteration_{i}.json'
            self.save(file)
            neptune_exp.send_artifact(file)
            if not save_dir:
                os.remove(file)
        
        # Stop the neptune experiment
        neptune_exp.stop()

    def to_dict(self,):
        d = super().to_dict()
        d["runner"].update(dict(
            hypervolume_ref = self.ref,
            neptune_project = self.neptune_project,
            neptune_experiment_name = self.neptune_experiment_name,
            neptune_description = self.neptune_description,
            neptune_files = self.neptune_files,
            neptune_tags = self.neptune_tags
        ))
        return d
        


def experiment_from_dict(d):
    if d["name"] == "SnarBenchmark":
        return SnarBenchmark.from_dict(d)
    elif d["name"] == "Hartmann3D":
        return Hartmann3D.from_dict(d)
    elif d["name"] == "Himmelblau":
        return Himmelblau.from_dict(d)
    elif d["name"] == "DTLZ2":
        return DTLZ2.from_dict(d)
    elif d["name"] == "VLMOP2":
        return VLMOP2.from_dict(d)
    elif d["name"] == "ThreeHumpCamel":
        return Himmelblau.from_dict(d)
