from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet

import numpy as np
import pandas as pd

class NelderMead(Strategy):
    ''' A reimplementation of the Nelder-Mead Simplex method

    Parameters
    ----------
    domain: `summit.domain.Domain`
        A summit domain object


    Examples
    -------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import NelderMead
    >>> from summit.utils.dataset import DataSet
    >>> import numpy as np
    >>> import scipy.optimize
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> d = {'temperature': [0.5], 'flowrate_a': [0.6], 'yield': 1}
    >>> df = pd.DataFrame(data=d)
    >>> previous = DataSet.from_df(df)
    >>> strategy = NelderMead(domain)
    >>> strategy.suggest_experiments(prev_res = previous)
    NAME  temperature  flowrate_a  flowrate_b
    0       77.539895    0.458517    0.111950
    1       85.407391    0.150234    0.282733
    2       64.545237    0.182897    0.359658
    3       75.541380    0.120587    0.211395
    4       94.647348    0.276324    0.370502
    '''

    def __init__(self, domain: Domain, **kwargs):
        Strategy.__init__(self, domain)

        self._adaptive = kwargs.get('adaptive', False)

    def suggest_experiments(self, prev_res: DataSet=None, prev_param=None):
        """ Suggest experiments using Nelder-Mead Simplex method

        Parameters
        ----------
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, the SNOBFIT optimization algorithm
            will be initialized and suggest initial experiments.
        prev_param: file.txt TODO: how to handle this?
            File with parameters of SNOBFIT algorithm from previous
            iterations of a optimization problem.
            If no data is passed, the SNOBFIT optimization algorithm
            will be initialized.

        Returns
        -------
        next_experiments: DataSet
            A `Dataset` object with the suggested experiments by SNOBFIT algorithm
        xbest: list
            List with variable settings of experiment with best outcome
        fbest: float
            Objective value at xbest
        param: list
            List with parameters and prev_param of SNOBFIT algorithm (required for next iteration)
        """

        # Extract dimension of input domain
        dim = self.domain.num_continuous_dimensions()

        # Get bounds of input variables
        bounds = []
        for v in self.domain.variables:
            if not v.is_objective:
                bounds.append(v.bounds)
        bounds = np.asarray(bounds, dtype=float)

        # Initialization
        x0 = []
        y0 = []

        # Get previous results
        if prev_res is not None:
            inputs, outputs = self.get_inputs_outputs(prev_res)
            x0 = inputs.data_to_numpy()
            y0 = outputs.data_to_numpy()

        elif prev_param is not None:
            raise ValueError('Parameter from previous optimization iteration are given but previous results are '
                             'missing!')

        # if no previous results are given initialize center point as zero-vector
        if not len(x0):
            x0 = np.zeros((1,len(bounds)))


        ''' Set Nelder-Mead parameters, i.e., initialize or include data from previous iterations
            --------
            prev_sim: 
                variable coordinates of simplex from previous run
            prev_fsim: 
                function evaluations corresponding to points of simplex from previous run
            x_iter: 
                variable coordinates and corresponding function evaluations of potential new 
                simplex points determined in one iteration of the NMS algorithm; note that 
                within one iteration multiple points need to be evaluated; that's why we have
                to store the points of an unfinished iteration (start iteration -> request point
                -> run experiment -> restart same iteration with results of experiment 
                -> request point -> run experiment ... -> finish iteration)
                    
                    
        '''
        prev_sim, prev_fsim, x_iter = None, None, None
        if prev_param:
            prev_sim= prev_param[0]
            if prev_param[1] is not None:
                prev_fsim = prev_param[1]
                x_iter = prev_param[2]
                for key, value in x_iter.items():
                    if value is not None:
                        if key == 'x_shrink':
                            for k in range(len(x0)):
                                for j in range(len(value)):
                                    if np.array_equal(value[j][0], np.asarray(x0[k])):
                                        x_iter[key][j][1] = y0[k]
                        for k in range(len(x0)):
                            if np.array_equal(value[0], np.asarray(x0[k])):
                                x_iter[key][1] = y0[k]
                                continue
            else:
                prev_fsim = y0
        elif prev_res is not None:
            prev_sim = x0
            prev_fsim = y0

        # Run Nelder-Mead Simplex algorithm for one iteration
        request, sim, fsim, x_iter = self.minimize_neldermead(x0=x0[0], x=x_iter, f=prev_fsim, prev_sim=prev_sim)
        res = [sim, fsim, x_iter]

        # Generate DataSet object with variable values of next experiments
        next_experiments = {}
        for i, v in enumerate(self.domain.variables):
            if not v.is_objective:
                next_experiments[v.name] = request[:,i]
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
        next_experiments[('strategy', 'METADATA')] = ['Nelder-Mead Simplex']*len(request)
        #print("\n ITER")
        #print(next_experiments)
        #print(res)
        return next_experiments, 0, 0, res


    # https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
    def minimize_neldermead(self, x0, x=None, f=None, prev_sim= None, initial_simplex= None, adaptive=False,
                             **unknown_options):
        """
        Minimization of scalar function of one or more variables using the
        Nelder-Mead algorithm.
        Options
        -------
        disp : bool
            Set to True to print convergence messages.
        maxiter, maxfev : int
            Maximum allowed number of iterations and function evaluations.
            Will default to ``N*200``, where ``N`` is the number of
            variables, if neither `maxiter` or `maxfev` is set. If both
            `maxiter` and `maxfev` are set, minimization will stop at the
            first reached.
        return_all : bool, optional
            Set to True to return a list of the best solution at each of the
            iterations.
        initial_simplex : array_like of shape (N + 1, N)
            Initial simplex. If given, overrides `x0`.
            ``initial_simplex[j,:]`` should contain the coordinates of
            the jth vertex of the ``N+1`` vertices in the simplex, where
            ``N`` is the dimension.
        xatol : float, optional
            Absolute error in xopt between iterations that is acceptable for
            convergence.
        fatol : number, optional
            Absolute error in func(xopt) between iterations that is acceptable for
            convergence.
        adaptive : bool, optional
            Adapt algorithm parameters to dimensionality of problem. Useful for
            high-dimensional minimization [1]_.
        References
        ----------
        .. [1] Gao, F. and Han, L.
           Implementing the Nelder-Mead simplex algorithm with adaptive
           parameters. 2012. Computational Optimization and Applications.
           51:1, pp. 259-277
        """


        if adaptive:
            dim = float(len(x0))
            rho = 1
            chi = 1 + 2 / dim
            psi = 0.75 - 1 / (2 * dim)
            sigma = 1 - 1 / dim
        else:
            rho = 1
            chi = 2
            psi = 0.5
            sigma = 0.5

        nonzdelt = 0.05
        zdelt = 0.00025

        print(x0)
        x0 = np.asfarray(x0).flatten()
        N = len(x0)

        if initial_simplex is None and prev_sim is None:
            print(1)
            sim = np.zeros((N + 1, N), dtype=x0.dtype)
            sim[0] = x0
            for k in range(N):
                y = np.array(x0, copy=True)
                if y[k] != 0:
                    y[k] = (1 + nonzdelt) * y[k]
                else:
                    y[k] = zdelt
                sim[k + 1] = y
            return sim, sim, None, None
        elif prev_sim is None:
            print(2)
            sim = np.asfarray(initial_simplex).copy()
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
            N = sim.shape[1]
        else:
            print(3)
            sim = np.asfarray(prev_sim).copy()
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
            N = sim.shape[1]

        one2np1 = list(range(1, N + 1))
        fsim = np.zeros((N + 1,), float)

        for k in range(N + 1):
            fsim[k] = f[k]

        print(f)

        ind = np.argsort(fsim)
        fsim = np.take(fsim, ind, 0)
        # sort so sim[0,:] has the lowest function value
        sim = np.take(sim, ind, 0)

        # Catch information on previous experiment
        if x:
            x_iter = x
        else:
            x_iter = {'xbar': None, 'xr': None, 'xe': None, 'xc': None, 'xcc': None, 'x_shrink': None}

        # Iteration
        while 1:
            print(x_iter)
            if not x_iter['xr']:
                # Centroid point: xbar
                xbar = np.add.reduce(sim[:-1], 0) / N
                # Reflection point xr
                xr = (1 + rho) * xbar - rho * sim[-1]
                x_iter['xbar'] = xbar
                x_iter['xr'] = [xr, None]
                return np.asarray([xr]), sim, fsim, x_iter
            xr = x_iter['xr'][0]
            fxr = x_iter['xr'][1]
            doshrink = 0

            # if function value of reflected point is better than best point of simplex, determine expansion point
            if fxr < fsim[0]:
                if not x_iter['xe']:
                    # expansion point: xe
                    xbar = x_iter['xbar']
                    xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                    x_iter['xe'] = [xe, None]
                    return np.asarray([xe]), sim, fsim, x_iter
                xe = x_iter['xe'][0]
                fxe = x_iter['xe'][1]
                # if expansion point is better than reflected point,
                # replace worst point of simplex by expansion point
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                # if reflected point is better than expansion point,
                # replace worst point of simplex by reflected point
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            # if function value of reflected point is not better than best point of simplex...
            else:  # fsim[0] <= fxr
                # ... but better than second worst point,
                # replace worst point of simplex by reflected point
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                # ... and not better than second worst point
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    # if reflected point is better than worst point
                    if fxr < fsim[-1]:
                        # contracted point: xc
                        if not x_iter['xc']:
                            xbar = x_iter['xbar']
                            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                            x_iter['xc'] = [xc, None]
                            return np.asarray([xc]), sim, fsim, x_iter
                        xc = x_iter['xc'][0]
                        fxc = x_iter['xc'][1]

                        # if contracted point is better than reflected point
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    # if reflected point is better than worst point
                    else:
                        # Perform an inside contraction
                        if not x_iter['xcc']:
                            xbar = x_iter['xbar']
                            xcc = (1 - psi) * xbar + psi * sim[-1]
                            x_iter['xcc'] = [xcc, None]
                            return np.asarray([xcc]), sim, fsim, x_iter
                        xcc = x_iter['xcc'][0]
                        fxcc = x_iter['xcc'][1]

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    # shrink simplex for all x
                    if doshrink:
                        x_shrink = []
                        x_shrink_f = []
                        if not x_iter['x_shrink']:
                            iteration_stop_point = 5
                            for j in one2np1:
                                sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                                xj = sim[j]
                                x_shrink.append(xj)
                                x_shrink_f.append([xj, None])
                            x_iter['x_shrink'] = x_shrink_f
                            return np.asarray(x_shrink), sim, fsim, x_iter
                        for j in one2np1:
                            print(x_iter["x_shrink"])
                            sim[j] = x_iter['x_shrink'][j-1][0]
                            fsim[j] = x_iter['x_shrink'][j-1][1]
            x_iter = {'xbar': None, 'xr': None, 'xe': None, 'xc': None, 'xcc': None, 'x_shrink': None}
        # end of iteration
        iteration_stop_point = 0

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        #iterations += 1

        x = sim[0]
        return None, sim, fsim, None



