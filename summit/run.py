from summit.strategies import Strategy, strategy_from_dict
from summit.experiment import Experiment
from summit.benchmarks import *
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit import get_summit_config_path

from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import pathlib
import uuid
import json
import logging
import pkg_resources

__all__ = ["experiment_from_dict", "Runner", "NeptuneRunner"]


def experiment_from_dict(d):
    if d["name"] == "SnarBenchmark":
        return SnarBenchmark.from_dict(d)
    elif d["name"] == "MIT_case1":
        return MIT_case1.from_dict(d)
    elif d["name"] == "MIT_case2":
        return MIT_case2.from_dict(d)
    elif d["name"] == "MIT_case3":
        return MIT_case3.from_dict(d)
    elif d["name"] == "MIT_case4":
        return MIT_case4.from_dict(d)
    elif d["name"] == "MIT_case5":
        return MIT_case5.from_dict(d)
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
    elif d["name"] == "ExperimentalEmulator":
        return ExperimentalEmulator.from_dict(d)
    elif d["name"] == "ReizmanSuzukiEmulator":
        return ReizmanSuzukiEmulator.from_dict(d)
    elif d["name"] == "BaumgartnerCrossCouplingEmulator":
        return BaumgartnerCrossCouplingEmulator.from_dict(d)
    elif d["name"] == "BaumgartnerCrossCouplingDescriptorEmulator":
        raise NotImplementedError(
            "BaumgartnerCrossCouplingDescriptorEmulator has been deprecated."
        )
    elif d["name"] == "BaumgartnerCrossCouplingEmulator_Yield_Cost":
        raise NotImplementedError(
            "BaumgartnerCrossCouplingEmulator_Yield_Cost has been deprecated."
        )
    elif d["name"] == "BaumgartnerCrossCouplingBenchmark":
        raise NotImplementedError(
            "BaumgartnerCrossCouplingBenchmark has been deprecated."
        )
    else:
        raise ValueError(f"""Experiment {d["name"]} not found.""")


class Runner:
    """Run a closed-loop strategy and experiment cycle

    Parameters
    ----------
    strategy : :class:`~summit.strategies.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : :class:`~summit.experiment.Experiment`
        The experiment or benchmark class to use for running experiments
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
    max_same : int, optional
        The number of allowed iterations where the objectives don't improve by more than f_tol. Default is None.
    max_restarts : int, optional
        Number of restarts if max_same where is violated. Default is 0.

    Examples
    --------
    >>> from summit import *
    >>> benchmark = SnarBenchmark()
    >>> strategy = Random(benchmark.domain)
    >>> r = Runner(strategy=strategy, experiment=benchmark, max_iterations=10)
    >>> # Turn progress bar on by setting to True below
    >>> r.run(progress_bar=False)

    """

    def __init__(
        self,
        strategy: Strategy,
        experiment: Experiment,
        num_initial_experiments=None,
        max_iterations=100,
        batch_size=1,
        f_tol=1e-5,
        max_same=None,
        max_restarts=0,
        **kwargs,
    ):
        self.strategy = strategy
        self.experiment = experiment
        self.n_init = num_initial_experiments
        self.max_iterations = max_iterations
        self.f_tol = f_tol
        self.batch_size = batch_size
        self.max_same = max_same
        self.max_restarts = max_restarts

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """Run the closed loop experiment cycle

        Parameters
        ----------
        prev_res: DataSet, optional
            Previous results to initialize the optimization
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to not saving locally.
        """
        save_freq = kwargs.get("save_freq")
        save_dir = kwargs.get("save_dir", str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get("save_at_end", True)

        n_objs = len(self.experiment.domain.output_variables)
        fbest_old = np.zeros(n_objs)
        fbest = np.zeros(n_objs)
        prev_res = kwargs.get("prev_res")
        self.restarts = 0

        if kwargs.get("progress_bar", True):
            bar = progress_bar(range(self.max_iterations))
        else:
            bar = range(self.max_iterations)
        for i in bar:
            # Get experiment suggestions
            if i == 0 and prev_res is None:
                k = self.n_init if self.n_init is not None else self.batch_size
                next_experiments = self.strategy.suggest_experiments(num_experiments=k)
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res
                )
            prev_res = self.experiment.run_experiments(next_experiments)

            for j, v in enumerate(self.experiment.domain.output_variables):
                if i > 0:
                    fbest_old[j] = fbest[j]
                if v.maximize:
                    fbest[j] = self.experiment.data[v.name].max()
                elif not v.maximize:
                    fbest[j] = self.experiment.data[v.name].min()

            # Save state
            if save_freq is not None:
                file = save_dir / f"iteration_{i}.json"
                if i % save_freq == 0:
                    self.save(file)

            compare = np.abs(fbest - fbest_old) > self.f_tol
            if all(compare) or i <= 1:
                nstop = 0
            else:
                nstop += 1

            if self.max_same is not None:
                if nstop >= self.max_same and self.restarts >= self.max_restarts:
                    self.logger.info(
                        f"{self.strategy.__class__.__name__} stopped after {i+1} iterations and {self.restarts} restarts."
                    )
                    break
                elif nstop >= self.max_same:
                    nstop = 0
                    prev_res = None
                    self.strategy.reset()
                    self.restarts += 1

        # Save at end
        if save_at_end:
            file = save_dir / f"iteration_{i}.json"
            self.save(file)

    def reset(self):
        self.strategy.reset()
        self.experiment.reset()

    def to_dict(
        self,
    ):
        runner_params = dict(
            num_initial_experiments=self.n_init,
            max_iterations=self.max_iterations,
            batch_size=self.batch_size,
            f_tol=self.f_tol,
            max_restarts=self.max_restarts,
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
        return cls(strategy=strategy, experiment=experiment, **d["runner"])

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


class NeptuneRunner(Runner):
    """Run a closed-loop strategy and experiment cycle with logging to Neptune



    Parameters
    ----------
    strategy : :class:`~summit.strategies.base.Strategy`
        The summit strategy to be used. Note this should be an object
        (i.e., you need to call the strategy and then pass it). This allows
        you to add any transforms, options in advance.
    experiment : :class:`~summit.experiment.Experiment`
        The experiment or benchmark class to use for running experiments
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
    max_same : int, optional
        The number of iterations where the objectives don't improve by more than f_tol. Default is max_iterations.
    max_restarts : int, optional
        Number of restarts if f_tol is violated. Default is 0.
    hypervolume_ref : array-like, optional
        The reference for the hypervolume calculation if it is a multiobjective problem.
        Should be an array of length the number of objectives. Default is at the origin.
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
        hypervolume_ref=None,
        **kwargs,
    ):

        super().__init__(strategy, experiment, **kwargs)

        # Hypervolume reference for multiobjective experiments
        n_objs = len(self.experiment.domain.output_variables)
        self.ref = hypervolume_ref if hypervolume_ref is not None else n_objs * [0]

        # Check that Neptune-client is installed
        installed = {pkg.key for pkg in pkg_resources.working_set}
        if "neptune-client" not in installed:
            raise RuntimeError(
                "Neptune-client not installed. Use pip install summit[experiments] to add extra dependencies."
            )

        # Set up Neptune variables
        self.neptune_project = neptune_project
        self.neptune_experiment_name = neptune_experiment_name
        self.neptune_description = neptune_description
        self.neptune_files = neptune_files
        self.neptune_tags = neptune_tags
        self.neptune_exp = kwargs.get("neptune_exp")

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def run(self, **kwargs):
        """Run the closed loop experiment cycle

        Parameters
        ----------
        save_freq : int, optional
            The frequency with which to checkpoint the state of the optimization. Defaults to None.
        save_at_end : bool, optional
            Save the state of the optimization at the end of a run, even if it is stopped early.
            Default is True.
        save_dir : str, optional
            The directory to save checkpoints locally. Defaults to `~/.summit/runner`.
        """
        # Set parameters
        prev_res = None
        self.restarts = 0
        n_objs = len(self.experiment.domain.output_variables)
        fbest_old = np.zeros(n_objs)
        fbest = np.zeros(n_objs)

        # Serialization
        save_freq = kwargs.get("save_freq")
        save_dir = kwargs.get("save_dir", str(get_summit_config_path()))
        self.uuid_val = uuid.uuid4()
        save_dir = pathlib.Path(save_dir) / "runner" / str(self.uuid_val)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_at_end = kwargs.get("save_at_end", True)

        # Create neptune experiment
        from neptune.sessions import Session, HostedNeptuneBackend

        if self.neptune_exp is None:
            session = Session(backend=HostedNeptuneBackend())
            proj = session.get_project(self.neptune_project)
            neptune_exp = proj.create_experiment(
                name=self.neptune_experiment_name,
                description=self.neptune_description,
                upload_source_files=self.neptune_files,
                logger=self.logger,
                tags=self.neptune_tags,
            )
        else:
            neptune_exp = self.neptune_exp

        # Run optimization loop
        if kwargs.get("progress_bar", True):
            bar = progress_bar(range(self.max_iterations))
        else:
            bar = range(self.max_iterations)
        for i in bar:
            # Get experiment suggestions
            if i == 0:
                k = self.n_init if self.n_init is not None else self.batch_size
                next_experiments = self.strategy.suggest_experiments(num_experiments=k)
            else:
                next_experiments = self.strategy.suggest_experiments(
                    num_experiments=self.batch_size, prev_res=prev_res
                )
            prev_res = self.experiment.run_experiments(next_experiments)

            # Send best objective values to Neptune
            for j, v in enumerate(self.experiment.domain.output_variables):
                if i > 0:
                    fbest_old[j] = fbest[j]
                if v.maximize:
                    fbest[j] = self.experiment.data[v.name].max()
                elif not v.maximize:
                    fbest[j] = self.experiment.data[v.name].min()

                neptune_exp.send_metric(v.name + "_best", fbest[j])

            # Send hypervolume for multiobjective experiments
            if n_objs > 1:
                output_names = [v.name for v in self.experiment.domain.output_variables]
                data = self.experiment.data[output_names].copy()
                for v in self.experiment.domain.output_variables:
                    if v.maximize:
                        data[(v.name, "DATA")] = -1.0 * data[v.name]
                y_pareto, _ = pareto_efficient(data.to_numpy(), maximize=False)
                hv = hypervolume(y_pareto, self.ref)
                neptune_exp.send_metric("hypervolume", hv)

            # Save state
            if save_freq is not None:
                file = save_dir / f"iteration_{i}.json"
                if i % save_freq == 0:
                    self.save(file)
                    neptune_exp.send_artifact(str(file))
                if not save_dir:
                    os.remove(file)

            # Stop if no improvement
            compare = np.abs(fbest - fbest_old) > self.f_tol
            if all(compare) or i <= 1:
                nstop = 0
            else:
                nstop += 1

            if self.max_same is not None:
                if nstop >= self.max_same and self.restarts >= self.max_restarts:
                    self.logger.info(
                        f"{self.strategy.__class__.__name__} stopped after {i+1} iterations and {self.restarts} restarts."
                    )
                    break
                elif nstop >= self.max_same:
                    nstop = 0
                    prev_res = None
                    self.strategy.reset()
                    self.restarts += 1

        # Save at end
        if save_at_end:
            file = save_dir / f"iteration_{i}.json"
            self.save(file)
            neptune_exp.send_artifact(str(file))
            if not save_dir:
                os.remove(file)

        # Stop the neptune experiment
        neptune_exp.stop()

    def to_dict(
        self,
    ):
        d = super().to_dict()
        d["runner"].update(
            dict(
                hypervolume_ref=self.ref,
                neptune_project=self.neptune_project,
                neptune_experiment_name=self.neptune_experiment_name,
                neptune_description=self.neptune_description,
                neptune_files=self.neptune_files,
                neptune_tags=self.neptune_tags,
            )
        )
        return d
