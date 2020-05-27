from abc import ABC, abstractmethod
from summit.domain import Domain
from summit.utils.dataset import DataSet
from summit.utils.multiobjective import pareto_efficient
import matplotlib.pyplot as plt
import pandas as pd
import time

class Experiment(ABC):
    """Base class for benchmarks"""
    def __init__(self, domain):
        self._domain = domain
        self.reset()

    @property
    def domain(self):
        """The  domain for the experiment"""
        return self._domain
    
    @property
    def data(self):
        """Datast of all experiments run"""
        return self._data

    def run_experiments(self, conditions,
                        computation_time=None,
                        **kwargs):
        """Run the experiment(s) at the given conditions
        
        Arguments
        ---------
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
            diff = time.time()-self.prev_itr_time
        elif self.prev_itr_time is None:
            diff = 0

        # Run experiments
        # TODO: Add an option to run these in parallel
        for i, condition in conditions.iterrows():
            start = time.time()
            res, extras = self._run(condition, **kwargs)
            experiment_time = time.time() - start
            self._data = self._data.append(res)
            self._data['experiment_time'].iat[-1] = experiment_time
            self._data['computation_time'].iat[-1] = diff
            if condition.get('strategy') is not None:
                self._data['strategy'].iat[-1] = condition.get('strategy').values[0]
            self.extras.append(extras)
        self.prev_itr_time = time.time()
        return self._data.iloc[-len(conditions):]
        
    @abstractmethod
    def _run(self, conditions, **kwargs):
        raise NotImplementedError('_run be implemented by subclasses of Benchmark')

    def reset(self):
        """Reset the experiment
        
        This will clear all data.

        """
        self.prev_itr_time = None
        columns = [var.name for var in self.domain.variables]
        md_columns = ['computation_time', 'experiment_time', 'strategy']
        columns += md_columns
        self._data = DataSet(columns=columns, metadata_columns=md_columns)
        self.extras = []
    
    def save(self, filename: str):
        """ Save data to a CSV file.
        """
        self.data.to_csv(filename)


    def pareto_plot(self, objectives=None, ax=None):
        '''  Make a 2D pareto plot of the experiments thus far
        
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
        
        Examples
        --------
        
        
        Notes
        -----
        
        
        ''' 
        if objectives is None:
            objectives = [v.name for v in self.domain.variables
                          if v.is_objective]
            objectives = objectives[0:2]
        
        if len(objectives) != 2:
            raise ValueError("Can only plot 2 objectives")

        data = self._data[objectives].copy()

        #Handle minimize objectives
        for objective in objectives:
            if not self.domain[objective].maximize:
                data[objective] = -1.0*data[objective]

        values, indices = pareto_efficient(data.to_numpy(),
                                           maximize=True)
        
        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False
        
        # Plot all data
        if len(self.data) > 0:
            strategies = pd.unique(self.data['strategy'])
            markers = ['o', 'x']
            for strategy, marker in zip(strategies, markers):
                strat_data = self.data[self.data['strategy']==strategy]
                ax.scatter(strat_data[objectives[0]],
                        strat_data[objectives[1]],
                        c='k', marker=marker, label=strategy)

            #Sort data so get nice pareto plot
            pareto_data = self.data.iloc[indices].copy()
            pareto_data = pareto_data.sort_values(by=objectives[0])
            ax.plot(pareto_data[objectives[0]], 
                    pareto_data[objectives[1]],
                    c='k', label='Pareto Front')
            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.tick_params(direction='in')
            ax.legend()

        if return_fig:
            return fig, ax
        else:
            return ax
    