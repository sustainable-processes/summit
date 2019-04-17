"""
A Direct Search Strategy for optimizing GP hyperparameters
"""
import scipydirect 
import scipy.optimize as so
from GPy.inference.optimization import Optimizer
import GPy
import numpy as np
import matplotlib.pyplot as plt
from lxml import html

class DirectOpt(Optimizer):
    """Combined global and local optimization"""
    def __init__(self, maxT=6000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxT = maxT
        
    def opt(self, x_init, f_fp=None, f=None, fp=None):
        nvar = x_init.shape[0]
        #Global optimization
        res1 = scipydirect.minimize(f, nvar=nvar, maxT=self.maxT)
        
        #Local gradient optimization
        res2 = so.minimize(f, res1.x)
        
        self.x_opt = res2.x
        
        
def plot_3d(f, m):
    """Plot model and objective function in 3d"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    gridX, gridY = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-1, 1, 0.1))
    Z = np.zeros_like(gridX)
    Zpredict = np.zeros_like(gridX)
    flattened = np.array([gridX.flatten(), gridY.flatten()]).T
    values = f(flattened)
    mean, var = m.predict(flattened)
    for i in range(values.shape[0]):
        Z.ravel()[i] = values[i, 0]
        Zpredict.ravel()[i] = mean[i]
    
    ax.plot_wireframe(gridX, gridY, Z, rstride=1, cstride=1, label='Data')
    ax.plot_wireframe(gridX, gridY, Zpredict, rstride=2, cstride=2, color='y', label='Model')
    ax.legend()
    
def loo_error(x, y, optimizer=None, kernel=None, max_iters=1000, num_restarts=10):
    """Calculate the the leave-one-out cross-validation errror"""
    n = x.shape[0]
    sq_errors = np.zeros(n)
    for i, point in enumerate(x):
        kern = kernel if kernel else GPy.kern.Matern52(input_dim=x.shape[1], ARD=True)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = GPy.models.GPRegression(x[mask, :], y[mask, :], kernel=kernel)
        if optimizer:
            m.optimize_restarts(num_restarts = num_restarts, 
                                verbose=False,
                                max_iters=max_iters,
                                optimizer=self._optimizer)
        else:
            m.optimize_restarts(num_restarts = num_restarts, 
                                verbose=False,
                                max_iters=max_iters)
        pred, _ = m.predict(np.atleast_2d(x[i, :]))
        sq_errors[i] = (pred-y[i, :])**2
    range_y = np.max(y) - np.min(y)
    avg_err = np.sqrt(1/n*np.sum(sq_errors))/range_y
    return avg_err

def plot_3d_model(ax, m):
    """Plot model and objective function in 3d"""
    gridX, gridY = np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))
    Zpredict = np.zeros_like(gridX)
    flattened = np.array([gridX.flatten(), gridY.flatten()]).T
    mean, var = m.predict(flattened)
    for i in range(mean.shape[0]):
        Zpredict.ravel()[i] = mean[i]
    ax.plot_wireframe(gridX, gridY, Zpredict, rstride=2, cstride=2, color='g',alpha=0.2, label='Model')
    


def get_cas_number(http, name):
    name = name.replace(' ', '+')
    r = http.request('GET', f'http://www.commonchemistry.org/search.aspx?terms={name}')
    tree = html.fromstring(r.data)
    try: 
        search_results = tree.xpath('//div[@id="searchresults"]')
        h2 = [v for v in search_results[0].iter('h2')]
        cas_number = [v for v in h2[0].iter('a')][0].text
        cas_number = cas_number.lstrip('\r\n                                ')
    except IndexError:
        search_results = tree.xpath('//div[@id="chemicalDetails"]')
        h2 = [v for v in search_results[0].iter('h2')]
        cas_number = [v for v in h2[0].iter('span')][0].text
    return cas_number
