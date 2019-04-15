from abc import ABC, abstractmethod

from GPy.models import GPRegression
from GPy.kern import *

class Model(ABC):
    
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class GPyModel(Model):
    def __init__(self, kernel, noise_var, optimizer):
        self._kernel = kernel
        self._noise_var = noise_var
        self._optimizer = optimizer
    
    def fit(self, X, Y):
        self._model = GPRegression(X,Y, self._kernel)

    def predict(self, X):
        m, v = self._model.predict(X)
        return m,v 