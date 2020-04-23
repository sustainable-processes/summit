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
                       optimization_time=None,
                       **kwargs):
        if optimization_time is not None:
            diff = optimization_time
        if optimization_time is None and self.prev_itr_time is not None:
            diff = time.time()-self.prev_itr_time
        elif self.prev_itr_time is None:
            diff = 0
        r = self._run(conditions, **kwargs)
        self._data = self._data.append(r)
        self._data['optimization_time'].iloc[-1] = diff
        self.prev_itr_time = time.time()

    @abstractmethod
    def _run(self, conditions, **kwargs):
        raise NotImplementedError('_run be implemented by subclasses of Benchmark')

    def reset(self):
        var_names = [v.name for v in self.domain.variables]
        self._data = Dataset(columns=var_names, metadata_columns=['optimization_time'])
        self.num_iterations = 0
        self.prev_itr_time = None
    