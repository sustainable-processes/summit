"""
Helper functions for GPy model training and analysis
"""
# import scipydirect 
# import scipy.optimize as so
from GPy.inference.optimization import Optimizer
import GPy
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_model(ax, m, f=None,
                  x_bounds=[-2, 2], 
                  y_bounds=[-2,2]):
    ''' Plot model in 3d 
    
    Parameters
    ---------- 
    ax: matplotlib axes object
        Axes on which to plot the model; this should have 3d projection enabled.
    m: GPy model
        Model to plot. Must have a 2 dimensional input space
    f: callable, optional
        Function to plot. Must accept an array of x, y values.
    x_bounds: array_like, optional
        The bounds of the x axis. Defaults to [-2, 2]
    y_bounds: array_like, optional
        The bounds of the y_axis. Defaults to [-2,2]
    ''' 
    gridX, gridY = np.meshgrid(np.arange(x_bounds[0], x_bounds[1], (x_bounds[1]-x_bounds[0])/50),
                               np.arange(y_bounds[0], y_bounds[1], (y_bounds[1]-y_bounds[0])/50))
    Zpredict = np.zeros_like(gridX)
    flattened = np.array([gridX.flatten(), gridY.flatten()]).T
    mean, var = m.predict(flattened)
    for i in range(mean.shape[0]):
        Zpredict.ravel()[i] = mean[i]
    ax.plot_wireframe(gridX, gridY, Zpredict, rstride=2, cstride=2, color='g',alpha=0.2, label='Model')
    if f is not None:
        Z = np.zeros_like(gridX)
        values = f(flattened)
        for i in range(values.shape[0]):
            Z.ravel()[i] = values[i, 0]
        ax.plot_wireframe(gridX, gridY, Z, rstride=3, alpha=0.2, cstride=3, label='Data')
        ax.legend()
    
def loo_error(x, y, kernel=None, optimizer=None, max_iters=1000, num_restarts=10):
    ''' Calculate the the leave-one-out cross-validation errror for a set of data
    
    Parameters
    ---------- 
    x: array_like
        Numpy array of the inputs. Must be 2 dimensional
    y: array_like
        Numpy array of the outputs. Must be 2 dimensional
    kernel: optional
        A kernel from `GPy.kern`. By default, the Matern52 kernel is used
    optimizer: `GPy.inference.optimization.Optimizer`, optional
        A custom optimizer to use in the hyperparameter inference. 
        See the DirectOpt optimizer below for an example.
        By default, the standard lbfgs optimizer from GPy will be used.
    max_iters: int, optional
        The maximum number of iterations used in the hyperparameter inference optimization
        Defaults to 1000
    num_restarts: int, optional
        The number of random restarts to use in the optimization. Defaults to 10
    
    Returns
    -------
    avg_error: float
        The leave-one-out cross validation error
    
    ''' 
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
                                optimizer=optimizer)
        else:
            m.optimize_restarts(num_restarts = num_restarts, 
                                verbose=False,
                                max_iters=max_iters)
        pred, _ = m.predict(np.atleast_2d(x[i, :]))
        sq_errors[i] = (pred-y[i, :])**2
    range_y = np.max(y) - np.min(y)
    avg_err = np.sqrt(1/n*np.sum(sq_errors))/range_y
    return avg_err

# class DirectOpt(Optimizer):
#     """Combined global and local optimization of model hyperparameters"""
#     def __init__(self, bounds, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.bounds = bounds
        
#     def opt(self, x_init, f_fp=None, f=None, fp=None):
        
#         #Global optimization
#         res1 = scipydirect.minimize(f, self.bounds)
        
#         #Local gradient optimization
#         res2 = so.minimize(f, res1.x)
        
#         self.x_opt = res2.x
