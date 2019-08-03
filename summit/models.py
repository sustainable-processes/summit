from summit.initial_design.latin_designer import lhs

from GPy.models import GPRegression
from GPy.kern import Matern52
import numpy as np
from numpy import matlib
import scipy

from abc import ABC, abstractmethod


class Model(ABC):
    
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class GPyModel(Model):
    def __init__(self, kernel=None, noise_var=1.0, optimizer=None):
        if kernel:
            self._kernel = kernel
        else:
            pass
            # input_dim = self.domain.num_continuous_dimensions() + self.domain.num_discrete_variables(), 
            # self._kernel =  Matern52(input_dim = input_dim, ARD=True)
        self._noise_var = noise_var
        self._optimizer = optimizer
        self._model = None
    
    def fit(self, X, Y, num_restarts=10, max_iters=2000, parallel=False):
        self._model = GPRegression(X,Y, self._kernel, noise_var=self._noise_var)
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

    def predict(self, X):
        m, v = self._model.predict(X)
        return m,v 
    
    def spectral_posterior_sample(self, n_spectral_points): 
        '''Take sample function from posterior GP'''

        Xnew = self._model.X
        Ynew = self._model.Y

        # Get variables from problem structure
        n, D = np.shape(Xnew)
        ell = self._model.kern.lengthscale.values
        sf2 = self._model.kern.variance.values[0]
        sn2 = self._model.Gaussian_noise.variance.values[0]

        # Monte carlo samples of W and b
        sW1 = lhs(D, n_spectral_points)
        sW2 = lhs(D, n_spectral_points)

        p = matlib.repmat(np.divide(1, ell), n_spectral_points, 1)
        q = np.sqrt(np.divide(2.5, chi2inv(sW2, 2.5)+1e-7)) #Add padding to prevent /0 errors
        # q.shape = (n_spectral_points, 1)
        W = np.multiply(p, scipy.stats.norm.ppf(sW1))
        W = np.multiply(W, q)

        b = lhs(1, n_spectral_points)
        b = 2*np.pi*b

        # Calculate phi
        phi = np.sqrt(2*sf2/n_spectral_points)*np.cos(W@Xnew.transpose() +  matlib.repmat(b, 1, n))

        #Sampling of theta according to phi
        A = phi@phi.transpose() + sn2*np.identity(n_spectral_points)
        invA = inv_cholesky(A)
        mu_theta = invA@phi@Ynew
        cov_theta = sn2*invA
        cov_theta = 0.5*(cov_theta+cov_theta.transpose())
        normal = scipy.stats.multivariate_normal(mu_theta[:, 0], cov_theta)
        theta = np.array([normal.rvs() for i in range(n_spectral_points)])

        #Posterior sample according to theta
        def f(x):
            import ipdb; ipdb.set_trace()
            inputs, _ = np.shape(x)
            x = x.astype(np.float64)
            bprime = matlib.repmat(b, 1, inputs)
            output =  (theta.T*np.sqrt(2*sf2/n_spectral_points))@np.cos(W*x.transpose()+ bprime)
            return output

        return f

def chi2inv(p, v):
    ''' Inverse chi-squared distribution
    '''
    output = scipy.stats.invgamma.cdf(p, v)
    # l, _ = np.shape(output)
    # output.shape = (l,)
    return output

def inv_cholesky(A):
    _, n = np.shape(A)
    chol = np.linalg.cholesky(A)
    x = np.linalg.solve(chol.transpose(), np.identity(n))
    y = np.linalg.solve(chol, x)
    return y