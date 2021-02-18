__all__ = ["Experiment"]

from abc import ABC, abstractmethod
from summit.domain import Domain
from summit.utils.dataset import DataSet
from summit.utils.multiobjective import pareto_efficient
from summit.utils import jsonify_dict, unjsonify_dict

import pandas as pd
import numpy as np
import time
import logging

COLORS = [
    (165, 0, 38),
    (215, 48, 39),
    (244, 109, 67),
    (253, 174, 97),
    (254, 224, 144),
    (255, 255, 191),
    (224, 243, 248),
    (171, 217, 233),
    (116, 173, 209),
    (69, 117, 180),
    (49, 54, 149),
]
COLORS = np.array(COLORS) / 256


class Experiment(ABC):
    """Base class for experiments

    Parameters
    ----------

    domain: summit.domain.Domain
        The domain of the experiment

    Notes
    -----

    Developers that subclass `Experiment` need to implement
    `_run`, which runs the experiments.

    """

    def __init__(self, domain, **kwargs):
        self.logger = kwargs.get("logger", logging.getLogger(__name__))
        self._domain = domain
        self.reset()

    @property
    def domain(self):
        """The  domain for the experiment"""
        return self._domain

    @property
    def data(self):
        """Datast of all experiments run"""
        self._data = self._data.reset_index(drop=True)
        return self._data

    def run_experiments(self, conditions, computation_time=None, **kwargs):
        """Run the experiment(s) at the given conditions

        Parameters
        ----------

        conditions: summit.utils.dataset.Dataset
            A dataset with columns matching the variables in the domain
            of a experiment(s) to run.
        computation_time: float, optional
            The time used by the strategy in calculating the next experiments.
            By default, the time since the last call to run_experiment is used.

        """
        # Bookeeping for time used by strategy when suggesting next experiment
        if computation_time is not None:
            diff = computation_time
        if computation_time is None and self.prev_itr_time is not None:
            diff = time.time() - self.prev_itr_time
        elif self.prev_itr_time is None:
            diff = 0

        # Run experiments
        # TODO: Add an option to run these in parallel
        for i, condition in conditions.iterrows():
            start = time.time()
            res, extras = self._run(condition, **kwargs)
            # res = add_metadata_columns(res, conditions[conditions.metadata_columns])
            experiment_time = time.time() - start
            self._data = self._data.append(res)
            self._data["experiment_t"].iat[-1] = float(experiment_time)
            self._data["computation_t"].iat[-1] = float(diff)
            if condition.get("strategy") is not None:
                self._data["strategy"].iat[-1] = condition.get("strategy").values[0]
            self.extras.append(extras)
        self.prev_itr_time = time.time()
        return self._data.iloc[-len(conditions) :]

    @abstractmethod
    def _run(self, conditions, **kwargs):
        """Run experiments at the specified conditions.

        Arguments
        ---------
        conditions: summit.utils.dataset.Dataset
            A dataset with columns matching the variables in the domain
            of a experiment(s) to run.

        Returns
        -------
        res, extras
            Should return a tuple where the first element is the
            DataSet with the conditions and results.  The second element
            is a dictionary with extra parameters to store about the run.
            The later can be an empty dictionary.
        """

        raise NotImplementedError("_run be implemented by subclasses of Experiment")

    def reset(self):
        """Reset the experiment

        This will clear all data.

        """
        self.prev_itr_time = None
        columns = [var.name for var in self.domain.variables]
        md_columns = ["computation_t", "experiment_t", "strategy"]
        columns += md_columns
        self._data = DataSet(columns=columns, metadata_columns=md_columns)
        self.extras = []

    def to_dict(self, **experiment_params):
        """Serialize the class to a dictionary

        Subclasses can add a experiment_params dictionary
        key with custom parameters for the experiment
        """
        extras = []

        for e in self.extras:
            if type(e) == dict:
                extras.append(jsonify_dict(e))
            if type(e) == np.ndarray:
                extras.append(e.tolist())
            else:
                extras.append(e)

        return dict(
            domain=self.domain.to_dict(),
            name=str(self.__class__.__name__),
            data=self.data.to_dict(),
            experiment_params=experiment_params,
            extras=extras,
        )

    @classmethod
    def from_dict(cls, d):
        domain = Domain.from_dict(d["domain"])
        experiment_params = d.get("experiment_params", {})
        exp = cls(domain=domain, **experiment_params)
        exp._data = DataSet.from_dict(d["data"])
        for e in d["extras"]:
            if type(e) == dict:
                exp.extras.append(unjsonify_dict(e))
            elif type(e) == list:
                exp.extras.append(np.array(e))
            else:
                exp.extras.append(e)
        return exp

    def pareto_plot(self, objectives=None, colorbar=False, ax=None):
        """Make a 2D pareto plot of the experiments thus far

        Parameters
        ----------
        objectives: array-like, optional
            List of names of objectives to plot.
            By default picks the first two objectives
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to

        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis

        if ax is a matplotlib axis, returns only the axis

        Raises
        ------
        ValueError
            If the number of objectives is not equal to two


        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        if objectives is None:
            objectives = [v.name for v in self.domain.variables if v.is_objective]
            objectives = objectives[0:2]

        if len(objectives) != 2:
            raise ValueError("Can only plot 2 objectives")

        data = self._data[objectives].copy()

        # Handle minimize objectives
        for objective in objectives:
            if not self.domain[objective].maximize:
                data[objective] = -1.0 * data[objective]

        values, indices = pareto_efficient(data.to_numpy(), maximize=True)

        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False

        # Plot all data
        if len(self.data) > 0:
            strategies = pd.unique(self.data["strategy"])
            markers = ["o", "x"]
            for strategy, marker in zip(strategies, markers):
                strat_data = self.data[self.data["strategy"] == strategy]
                c = strat_data.index.values if colorbar else "k"
                cmap = ListedColormap(COLORS[: len(c)])
                im = ax.scatter(
                    strat_data[objectives[0]],
                    strat_data[objectives[1]],
                    cmap=cmap,
                    c=c,
                    alpha=1 if colorbar else 0.5,
                    marker=marker,
                    s=100,
                    label=strategy,
                )

            # Sort data so get nice pareto plot
            self.pareto_data = self.data.iloc[indices].copy()
            self.pareto_data = self.pareto_data.sort_values(by=objectives[0])
            if len(self.pareto_data) > 2:
                ax.plot(
                    self.pareto_data[objectives[0]],
                    self.pareto_data[objectives[1]],
                    c=(165 / 256, 0, 38 / 256),
                    label="Pareto Front",
                    linewidth=3,
                )
            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            if return_fig and colorbar:
                fig.colorbar(im)
            ax.tick_params(direction="in")
            ax.legend()

        if return_fig:
            return fig, ax
        elif return_fig and colorbar:
            return fig, ax, im
        elif not return_fig and colorbar:
            return ax, im
        else:
            return ax


def add_metadata_columns(df, metadata_df):
    for column in metadata_df.metadata_columns:
        df[(column, "METADATA")] = metadata_df[column]
    return df
