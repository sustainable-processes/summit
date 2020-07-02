from .dataset import DataSet
from abc import ABC, abstractmethod
import GPy
import pyrff
import numpy as np
from numpy import matlib
from numpy.random import default_rng
from GPy.models import GPRegression
from GPy.kern import Matern52
from scipy.stats import norm, invgamma
from scipy.stats.distributions import chi2
from .lhs import lhs
from sklearn.base import BaseEstimator, RegressorMixin


__all__ = ["Model", "ModelGroup", "GPyModel", "AnalyticalModel"]


class Model(ABC):
    """ Base class for model
    
    The model format is meant to reflect the sklearn API 
    """

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
            model.fit(X, y[[column_name]], **kwargs) 

    def predict(self, X, **kwargs):
        """
        Note
        -----
        This the make the assumption that each model returns a n_samples x 1 array
        from the predict method.
        """
        result = [model.predict(X, **kwargs)[:, 0] for model in self.models.values()]
        return np.array(result).T

    def __getitem__(self, key):
        return self.models[key]

    def to_dict(self):
        models = {}
        for k, v in self._models.items():
            models[k] = v.to_dict()
        return models

    @classmethod
    def from_dict(cls, d):
        models = {}
        for k, v in d.items():
            models[k] = model_from_dict(v)
        return cls(models)

def model_from_dict(d):
    if d["name"] == "GPyModel":
        return GPyModel.from_dict(d)
    elif d["name"] == "AnalyticalModel":
        return AnalyticalModel.from_dict(d)
    else:
        raise TypeError(f"Model Type {d['name']} is not valid.")


class GPyModel(BaseEstimator, RegressorMixin):
    """ A Gaussian Process Regression model from GPy

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
    
    """

    def __init__(self, kernel=None, input_dim=None, noise_var=1.0, optimizer=None):
        if kernel:
            self._kernel = kernel
        else:
            if not input_dim:
                raise ValueError(
                    "input_dim must be specified if no kernel is specified."
                )
            self.input_dim = input_dim
            self._kernel = Matern52(input_dim=self.input_dim, ARD=True)
        self._noise_var = noise_var
        self._optimizer = optimizer
        self._model = None
        self.input_mean = []
        self.input_std = []
        self.output_mean = []
        self.output_std = []

    def fit(self, X, y, **kwargs):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : DataSet
            The data columns will be used as inputs for fitting the model
        y : DataSet
            The data columns will be used as outputs for fitting the model
        num_restarts : int, optional (default=10)
            The number of random restarts of the optimizer.
        max_iters : int, optional (default=2000)
            The maximum number of iterations of the optimizer.
        parallel : bool (default=False)
            Use parallel computation for the optimization.
        spectral_sample: bool, optional
            Calculate the spectral sampled function. Defaults to False.

        Returns
        -------
        self : returns an instance of self.
        -----
        """

        num_restarts=kwargs.get('num_restarts',10)
        max_iters=kwargs.get('max_iters', 2000)
        parallel=kwargs.get('parallel',False)
        spectral_sample=kwargs.get('spectral_sample',False)
        verbose = kwargs.get('verbose', False)

        # Read in dataset or numpy array
        if isinstance(X, DataSet):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError("X must be a dataset or numpy array")

        if isinstance(y, DataSet):
            y = y.to_numpy()
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise TypeError("Y must be a DataSet or numpy array")

        # Initialize model
        self._model = GPRegression(
            X, y, self._kernel, noise_var=self._noise_var
        )

        # Set priors to constrain hyperparameters
        self._model.kern.lengthscale.set_prior(GPy.priors.Gamma(1, 0.1), warning=False)
        self._model.kern.variance.set_prior(GPy.priors.Gamma(1, 0.1), warning=False)
        self._model.Gaussian_noise.variance.set_prior(GPy.priors.Gamma(0.5,0.1), warning=False)

        # Fit model
        if self._optimizer:
            self._model.optimize_restarts(num_restarts = num_restarts, 
                                          verbose=verbose,
                                          max_iters=max_iters,
                                          optimizer=self._optimizer,
                                          parallel=parallel)
        else:
            self._model.optimize_restarts(num_restarts = num_restarts, 
                                          verbose=verbose,
                                          max_iters=max_iters,
                                          parallel=parallel)
        if spectral_sample:
            self.spectral_sample(X, y)

        return self


    def predict(self, X, **kwargs):
        """Predict using the Gaussian process regression model

        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        use_spectral_sample: bool, optional
            Use a spectral sample of the GP instead of the posterior prediction.
            Default is True.
        """
        if not self._model:
            raise ValueError("Fit must be called on the model prior to prediction")

        if isinstance(X, DataSet):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError("X must be a dataset or numpy array")

        # Use spectral sample when called. Otherwise use model directly
        use_spectral_sample = kwargs.get('use_spectral_sample', True)
        if use_spectral_sample and self.sampled_f is not None:
            return self.sampled_f(X)
        elif use_spectral_sample:
            raise ValueError("Spectral Sample must be called during fitting prior to prediction.")
        else:
            return self._model.predict(X)

    def spectral_sample(self, X, y, n_spectral_points=1500,
                        n_retries=10):
        '''Sample GP using spectral sampling

        Parameters
        ----------
        X: DataSet
            The data columns will be used as inputs for fitting the model
        y: DataSet
            The data columns will be used as outputs for fitting the model
        n_spectral_points: int, optional
            The number of points to use in spectral sampling. Defaults to 4000.
        n_retries: int, optional
            The number of retries for the spectral sampling code in the case
            the singular value decomposition fails.
        '''

        # Determine the degrees of freedom
        if type(self._model.kern) == GPy.kern.Exponential:
            matern_nu = 1
        elif type(self._model.kern) == GPy.kern.Matern32:
            matern_nu = 3
        elif type(self._model.kern) == GPy.kern.Matern52:
            matern_nu = 5
        elif type(self._model.kern) == GPy.kern.RBF:
            matern_nu = np.inf
        else:
            raise TypeError("Spectral sample currently only works with Matern type kernels, including RBF.")

        if isinstance(X, DataSet):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError("X must be a dataset or numpy array")
        
        if isinstance(y, DataSet):
            y = y.to_numpy()
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise TypeError("Y must be a DataSet or numpy array")
        
        # Spectral sampling. Clip values to match Matlab implementation
        noise = self._model.Gaussian_noise.variance.values[0]
        noise = np.clip(noise, 1e-6, 1)
        variance = self._model.kern.variance.values[0]
        variance = np.clip(variance, np.sqrt(1e-3), np.sqrt(1e3))
        lengthscales = self._model.kern.lengthscale.values
        lengthscales = np.clip(lengthscales, np.sqrt(1e-3), np.sqrt(1e3))
        for i in range(n_retries):
            try:
                sampled_f = pyrff.sample_rff(
                    lengthscales=self._model.kern.lengthscale.values,
                    scaling=self._model.kern.variance.values[0],
                    noise=noise,
                    kernel_nu=matern_nu,
                    X=X,
                    Y=y[:,0],
                    M=n_spectral_points,
                    )
                break
            except np.linalg.LinAlgError or ValueError:
                pass

        # Define function wrapper
        def f(x_new):
            y_s = sampled_f(x_new)
            return np.atleast_2d(y_s).T   
        self.sampled_f = f
        return f
    
    @property
    def hyperparameters(self):
        """Returns a tuple for the form legnthscales, variance, noise"""
        lengthscales = self._model.kern.lengthscale.values
        variance = self._model.kern.variance.values[0]
        noise = self._model.Gaussian_noise.variance.values[0]
        return lengthscales, variance, noise

    def to_dict(self):
        _model = self._model.to_dict() if self._model is not None else self._model
        return dict(
            name="GPyModel",
            _model=_model,
            kernel=self._kernel.to_dict(),
            noise_var=self._noise_var,
            input_mean=list(self.input_mean),
            input_std=list(self.input_std),
            output_mean=list(self.output_mean),
            output_std=list(self.output_std),
        )

    @classmethod
    def from_dict(cls, d):
        kernel = GPy.kern.Kern.from_dict(d["kernel"])
        m = cls(kernel=kernel, noise_var=d["noise_var"])
        if d["_model"] is not None:
            m._model = GPRegression.from_dict(d["_model"])
        m.input_mean = np.array(d["input_mean"])
        m.input_std = np.array(d["input_std"])
        m.output_mean = np.array(d["output_mean"])
        m.output_std = np.array(d["output_std"])
        return m

class AnalyticalModel(Model):
    """ An analytical model instead of statistical model

    Use this for an objective that is a 
    known analytical function of the inputs 

    Parameters
    ---------- 
    function: callable
        An an analytical function that takes an input 
        array and returns the output
    """

    def __init__(self, function: callable):
        self._function = function

    def fit(self, X, Y, **kwargs):
        """This method is here because it is required.
           No fitting actually occurs"""
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
