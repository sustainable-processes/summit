from abc import ABC, abstractmethod
from summit.domain import Domain
from summit.utils.dataset import Dataset
import time

class Experiment(ABC):
    """Base class for benchmarks"""
    def __init__(self, domain):
        self._domain = domain
    
    @property
    def domain(self):
        """The  domain for the experiment"""
        return self._domain

    def run_experiment(self, conditions,
                       computation_time=None,
                       **kwargs):
        """Run the experiment at the given conditions
        
        Arguments
        ---------
        conditions: summit.utils.dataset.Dataset
            A dataset with columns matching the variables in the domain
            of a experiment(s) to run.
        computation_time: float, optional
            The time used by the strategy in calculating the next experiments.
            By default, the time since the last call to run_experiment is used. 
               
        """
        # Bookeeping for time used by algorithm
        # when suggesting next experiment
        if computation_time is not None:
            diff = computation_time
        if computation_time is None and self.prev_itr_time is not None:
            diff = time.time()-self.prev_itr_time
        elif self.prev_itr_time is None:
            diff = 0

        # Run experiments
        for condition in conditions.iterrows():
            start = time.time()
            r = self._run(conditions, **kwargs)
            experiment_time = time.time() - start
            self._data = self._data.append(r)
            self._data['experiment_time'].iloc[-1] = experiment_time
            self._data['computation_time'].iloc[-1] = diff
        self.prev_itr_time = time.time()
        #Limit to only the experiments run this iteration
        return self._data 
    @abstractmethod
    def _run(self, conditions, **kwargs):
        raise NotImplementedError('_run be implemented by subclasses of Benchmark')

    def reset(self):
        var_names = [v.name for v in self.domain.variables]
        metadata_columns = ['computation_time', 'experiment_time']
        self._data = Dataset(columns=var_names, metadata_columns=metadata_columns)
        self.num_experiments = 0
        self.prev_itr_time = None