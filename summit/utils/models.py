
from .dataset import DataSet
from abc import ABC, abstractmethod
import GPy
import numpy as np
from GPy.models import GPRegression
from GPy.kern import Matern52
from sklearn.base import BaseEstimator, RegressorMixin

__all__ = ["Model", "ModelGroup", "GPyModel", "AnalyticalModel"]

class Model(ABC):
    ''' Base class for model
    
    The model format is meant to reflect the sklearn API 
    ''' 
    
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class ModelGroup:
    def __init__(self, models: dict):
        self._models = models

    @property
    def models(self):
        return self._models

    def fit(self, X, y, **kwargs):
        for column_name, model in self.models.items():
            model.fit(X, y[[column_name]]) 

    def predict(self, X, **kwargs):
        """
        Note
        -----
        This the make the assumption that each model returns a n_samples x 1 array
        from the predict method.
        """
        result = [model.predict(X)[:, 0] for model in self.models.values()]
        return np.array(result).T
    
    def __getitem__(self, key):
        return self.models[key]

    def to_dict(self):
        models = {}
        for k,v in self._models.items():
            models[k] = v.to_dict()
        return models
    
    @classmethod
    def from_dict(cls, d):
        models = {}
        for k,v in d.items():
            models[k] = model_from_dict(v)
        return cls(models)

def model_from_dict(d):
    if d['name'] == 'GPyModel':
        return GPyModel.from_dict(d)
    elif d['name'] == 'AnalyticalModel':
        return AnalyticalModel.from_dict(d)
    else:
        raise TypeError(f"Model Type {d['name']} is not valid.")
 
class GPyModel(BaseEstimator, RegressorMixin):
    ''' A Gaussian Process Regression model from GPy

    This is implemented as an alternative to the sklearn
    gaussian process because GPy offers several performance speed-ups. 
    
    Parameters
    ---------- 
    kernel: GPy.kern, optional
        A GPy kernel. Defaults to the Matern52 kernel with
        automatic relevance detection enabled.
    input_dim: int, optional
        The number of dimensions in the input. This must
        be specified if a kernel is not specified.
    noise_var: float, optional
        The noise variance for Gaussian likelhood, defaults to 1.
    optimizer: optional
        A custom optimizer. Defaults to GPy's internal optimizer.
    
    Notes
    -----
    For instructions on how to implement a custom optimizer, see
    here: https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/optimizer-implementation.ipynb
    
    ''' 
    def __init__(self, kernel=None, input_dim=None,noise_var=1.0, optimizer=None):
        if kernel:
            self._kernel = kernel
        else: 
            if not input_dim:
                raise ValueError('input_dim must be specified if no kernel is specified.')
            self.input_dim = input_dim
            self._kernel =  Matern52(input_dim = self.input_dim, ARD=True)
        self._noise_var = noise_var
        self._optimizer = optimizer
        self._model = None
        self.input_mean = []
        self.input_std = []
        self.output_mean = []
        self.output_std = []
    
    def fit(self, X, y, num_restarts=10, max_iters=2000, parallel=False):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : DataSet
            The data columns will be used as inputs for fitting the model
        y : DataSEt
            The data columns will be used as outputs for fitting the model
        num_restarts : int, optional (default=10)
            The number of random restarts of the optimizer.
        max_iters : int, optional (default=2000)
            The maximum number of iterations of the optimizer.
        parallel : bool (default=False)
            Use parallel computation for the optimization.

        Returns
        -------
        self : returns an instance of self.
        -----
        """ 
        #Standardize inputs and outputs
        if isinstance(X, DataSet):
            X_std, self.input_mean, self.input_std = X.standardize(return_mean=True, return_std=True)
        elif isinstance(X, np.ndarray):
            self.input_mean = np.mean(X,axis=0)
            self.input_std = np.std(X, axis=0)
            X_std = (X-self.input_mean)/self.input_std
            X_std[abs(X_std) < 1e-5] = 0.0

        if isinstance(y, DataSet):
            y_std, self.output_mean, self.output_std = y.standardize(return_mean=True, return_std=True)
        elif isinstance(y, np.ndarray):
            self.output_mean = np.mean(y,axis=0)
            self.output_std = np.std(y, axis=0)
            y_std = (y-self.output_mean)/self.output_std
            y_std[abs(y_std) < 1e-5] = 0.0

        #Initialize and fit model
        self._model = GPRegression(X_std,y_std, self._kernel, noise_var=self._noise_var)
        if self._optimizer:
            self._model.optimize_restarts(num_restarts = num_restarts, 
                                          verbose=False,
                                          max_iters=max_iters,
                                          optimizer=self._optimizer,
                                          parallel=parallel)
        else:
            self._model.optimize_restarts(num_restarts = num_restarts, 
                                          verbose=False,
                                          max_iters=max_iters,
                                          parallel=parallel)
        return self

    def predict(self, X, 
                return_cov: bool = False,
                return_std: bool = False):
        """Predict using the Gaussian process regression model

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        """
        if not self._model:
            raise ValueError('Fit must be called on the model prior to prediction')

        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        
        if isinstance(X, np.ndarray):
            X_std =  (X-self.input_mean)/self.input_std
            X_std[abs(X_std) < 1e-5] = 0.0
        elif isinstance(X, DataSet):
            X_std = X.standardize(mean=self.input_mean, std=self.input_std)
        else:
            raise TypeError("X must be a numpy array or summit DataSet.")

        m_std, v_std = self._model.predict(X_std)
        m = m_std*self.output_std + self.output_mean

        return m
    
    def to_dict(self):
        _model = self._model.to_dict() if self._model is not None else self._model
        return dict(name="GPyModel",
                    _model=_model,
                    kernel = self._kernel.to_dict(),
                    noise_var = self._noise_var,
                    input_mean=list(self.input_mean),
                    input_std=list(self.input_std),
                    output_mean=list(self.output_mean),
                    output_std=list(self.output_std))
    
    @classmethod
    def from_dict(cls, d):
        kernel = GPy.kern.Kern.from_dict(d['kernel'])
        m = cls(kernel=kernel, noise_var=d['noise_var'])
        if d['_model'] is not None:
            m._model = GPRegression.from_dict(d['_model'])
        m.input_mean = np.array(d['input_mean'])
        m.input_std = np.array(d['input_std'])
        m.output_mean = np.array(d['output_mean'])
        m.output_std = np.array(d['output_std'])
        return m

class AnalyticalModel(Model):
    ''' An analytical model instead of statistical model

    Use this for an objective that is a 
    known analytical function of the inputs 

    Parameters
    ---------- 
    function: callable
        An an analytical function that takes an input 
        array and returns the output
    '''

    def __init__(self, function: callable):
        self._function = function

    def fit(self, X, Y, **kwargs):
        '''This method is here because it is required.
           No fitting actually occurs'''
        pass

    def predict(self, X, **kwargs):
        """Predict using the analytical function

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        """

        return self.function(X, **kwargs)

    @property
    def function(self) -> callable:
        return self._function

