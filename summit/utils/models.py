
from .dataset import DataSet
from abc import ABC, abstractmethod
import GPy
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

class GPyModel:
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
        self.sampled_f = None
    
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
        """
        if not self._model:
            raise ValueError('Fit must be called on the model prior to prediction')

        use_spectral_sample = kwargs.get('use_spectral_sample', False)
        if use_spectral_sample and self.sampled_f is not None:
            return self.sampled_f(X)
        elif use_spectral_sample:
            raise ValueError("Spectral Sample must be called during fitting prior to prediction.")
        
        if isinstance(X, np.ndarray):
            X_std =  (X-self.input_mean)/self.input_std
            X_std[abs(X_std) < 1e-5] = 0.0
        elif isinstance(X, DataSet):
            X_std = X.standardize(mean=self.input_mean, std=self.input_std)

        m_std, v_std = self._model.predict(X_std)
        m = m_std*self.output_std + self.output_mean

        return m

    def spectral_sample(self, X, Y, n_spectral_points=1500):
        '''Sample GP using spectral sampling

        Parameters
        ----------
        X: DataSet
            The data columns will be used as inputs for fitting the model
        y: DataSet
            The data columns will be used as outputs for fitting the model
        n_spectral_points: float, optional
            The number of points to use in spectral sampling. Defaults to 4000.
        '''

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

        if isinstance(X, np.ndarray):
            X_std =  (X-self.input_mean)/self.input_std
            X_std[abs(X_std) < 1e-5] = 0.0
        elif isinstance(X, DataSet):
            X_std = X.standardize(mean=self.input_mean, 
                                  std=self.input_std)

        # Get variables from problem structure
        n, D = np.shape(X_std)
        ell = np.array(self._model.kern.lengthscale) 
        sf2 = np.array(self._model.kern.variance)
        sn2 = np.array(self._model.Gaussian_noise.variance)

        # Monte carlo samples of W and b
        # W = p*norminv(sW1)*q
        sW = lhs(D, n_spectral_points)
        p = matlib.repmat(np.divide(1, ell), n_spectral_points, 1)
        if matern_nu != np.inf:            
            inv = chi2.ppf(sW, matern_nu)
            q = np.sqrt(np.divide(matern_nu, inv)+1e-7)
            W = np.multiply(p, norm.ppf(sW))
            W = np.multiply(W, q)
        else:
            raise NotImplementedError("RBF not implemented yet!")

        b = 2*np.pi*lhs(1, n_spectral_points)

        # Calculate phi
        phi = np.sqrt(2*sf2/n_spectral_points)*np.cos(W@X_std.T +  matlib.repmat(b, 1, n))

        #Sampling of theta according to phi
        #For the matrix inverses, I defualt to Cholesky when possible
        A = phi@phi.T + sn2*np.identity(n_spectral_points)
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
        cov_theta = 0.5*(cov_theta+cov_theta.T)+1e-4*np.identity(n_spectral_points)
        rng = default_rng()
        try:
            theta = rng.multivariate_normal(mu_theta[:, 0], cov_theta,
                                           method='cholesky')
        except np.linalg.LinAlgError:
            theta = rng.multivariate_normal(mu_theta[:, 0], cov_theta,
                                           method='svd')

        # theta = np.random.multivariate_normal(mu_theta[:, 0], cov_theta)

        #Posterior sample according to theta
        def f(x):
            if isinstance(x, np.ndarray):
                x =  (x-self.input_mean)/self.input_std
                x[abs(x) < 1e-5] = 0.0
            elif isinstance(X, DataSet):
                x = x.standardize(mean=self.input_mean, 
                                  std=self.input_std)
            inputs, _ = np.shape(x)
            bprime = matlib.repmat(b, 1, inputs)
            output =  (theta.T*np.sqrt(2*sf2/n_spectral_points))@np.cos(W@x.T+bprime)
            return np.atleast_2d(output).T
        self.sampled_f = f
        return f
    
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

