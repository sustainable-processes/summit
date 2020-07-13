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
import logging


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

    def __init__(self, kernel=None, input_dim=None, noise_var=1.0, optimizer=None, **kwargs):
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
        self.logger = kwargs.get('logger', logging.getLogger(__name__))

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
        max_iters=kwargs.get('max_iters', 10000)
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
        self._model.kern.lengthscale.constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3),warning=False)
        self._model.kern.lengthscale.set_prior(GPy.priors.LogGaussian(0, 10), warning=False)
        self._model.kern.variance.constrain_bounded(np.sqrt(1e-3), np.sqrt(1e3), warning=False)
        self._model.kern.variance.set_prior(GPy.priors.LogGaussian(-6, 10), warning=False)
        self._model.Gaussian_noise.constrain_bounded(np.exp(-6), 1, warning=False)
        
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
            self.spectral_sample(X, y, **kwargs)

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
            return self._model.predict(X)[0]

    def spectral_sample(self, X, y, n_spectral_points=1500,
                        n_retries=10, **kwargs):
        """Sample GP using spectral sampling

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
        """

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
        sampled_f = None
        for i in range(n_retries):
            try:
                sampled_f = pyrff.sample_rff(
                    lengthscales=self._model.kern.lengthscale.values,
                    scaling=np.sqrt(self._model.kern.variance.values[0]),
                    noise=np.sqrt(noise),
                    kernel_nu=matern_nu,
                    X=X,
                    Y=y[:,0],
                    M=n_spectral_points,
                    )
                break
            except np.linalg.LinAlgError as e:
                self.logger.error(e)
            except ValueError as e:
                self.logger.error(e)

        if sampled_f is None:
            raise RuntimeError(f"Spectral sampling failed after {n_retries} retries.")

        # Define function wrapper
        def f(x_new):
            y_s = sampled_f(x_new)
            return np.atleast_2d(y_s).T   
        self.sampled_f = f
        return self.sampled_f
    
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

def spectral_sample(lengthscales, scaling, noise, kernel_nu, X, Y, M):
        # Get variables from problem structure
        n, D = np.shape(X)
        ell = np.array(lengthscales) 
        sf2 = scaling
        sn2 = noise

        # Monte carlo samples of W and b
        sW = lhs(D, M, criterion='maximin')
        p = matlib.repmat(np.divide(1, ell), M, 1)
        if kernel_nu != np.inf:            
            inv = chi2.ppf(sW, kernel_nu)
            q = np.sqrt(np.divide(kernel_nu, inv)+1e-7)
            W = np.multiply(p, norm.ppf(sW))
            W = np.multiply(W, q)
        else:
            raise NotImplementedError("RBF not implemented yet!")

        b = 2*np.pi*lhs(1, M)

        # Calculate phi
        phi = np.sqrt(2*sf2/M)*np.cos(W@X.T +  matlib.repmat(b, 1, n))

        #Sampling of theta according to phi
        #For the matrix inverses, I defualt to Cholesky when possible
        A = phi@phi.T + sn2*np.identity(M)
        try:
            c = np.linalg.inv(np.linalg.cholesky(A))
            invA = np.dot(c.T,c)
        except np.linalg.LinAlgError:
            u,s, vh = np.linalg.svd(A)
            invA = vh.T@np.diag(1/s)@u.T
        if isinstance(Y, DataSet):
            Y = Y.data_to_numpy()
        mu_theta = invA@phi@Y
        cov_theta = sn2*invA
        #Add some noise to covariance to prevent issues
        cov_theta = 0.5*(cov_theta+cov_theta.T)+1e-4*np.identity(M)
        rng = default_rng()
        try:
            theta = rng.multivariate_normal(mu_theta, cov_theta,
                                           method='cholesky')
        except np.linalg.LinAlgError:
            theta = rng.multivariate_normal(mu_theta, cov_theta,
                                           method='svd')

        #Posterior sample according to theta
        def f(x):
            inputs, _ = np.shape(x)
            bprime = matlib.repmat(b, 1, inputs)
            output =  (theta.T*np.sqrt(2*sf2/M))@np.cos(W@x.T+bprime)
            return np.atleast_2d(output).T
        return f

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
