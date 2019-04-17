from abc import ABC, abstractmethod

from GPy.models import GPRegression
from GPy.kern import Matern52

class Model(ABC):
    
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class GPyModel(Model):
    def __init__(self, kernel=None, noise_var=0.01, optimizer=None):
        if kernel:
            self._kernel = kernel
        else:
            input_dim = self.domain.num_continuous_dimensions() + self.domain.num_discrete_variables(), 
            self._kernel =  Matern52(input_dim = input_dim, ARD=True)
        self._noise_var = noise_var
        self._optimizer = optimizer
    
    def fit(self, X, Y):
        self._model = GPRegression(X,Y, self._kernel)
        if self._optimizer:
            self._model.optimize(self._optimizer)
        else:
            self._model.optimize()

    def predict(self, X):
        m, v = self._model.predict(X)
        return m,v 