"""
A Direct Search Strategy for optimizing GP hyperparameters
"""
import scipydirect 
import scipy.optimize as so
from GPy.inference.optimization import Optimizer

class DirectOpt(Optimizer):
    def __init__(self, bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds = bounds
        
    def opt(self, x_init, f_fp=None, f=None, fp=None):
        
        #Global optimization
        res1 = scipydirect.minimize(f, self.bounds)
        
        #Local gradient optimization
        res2 = so.minimize(f, res1.x)
        
        self.x_opt = res2.x