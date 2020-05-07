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
            If no data is passed, the Nelder-Mead optimization algorithm
            will be initialized and suggest initial experiments.
        prev_param: file.txt TODO: how to handle this?
            File with parameters of Nelder-Mead algorithm from previous
            iterations of a optimization problem.
            If no data is passed, the Nelder-Mead optimization algorithm
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
        prev_sim, prev_fsim, x_iter, red_dim, red_sim, red_fsim, rec_dim, memory = None, None, None, None, None, None, None, [[-100,-100]]

        if prev_param:
            prev_sim= prev_param[0]
            red_dim = prev_param[3]
            red_sim = prev_param[4]
            red_fsim = prev_param[5]
            rec_dim = prev_param[6]
            memory = prev_param[7]
            print(prev_param)
            if rec_dim:
                flat_y0 = [y0[i] for i in range(len(y0))]
                prev_fsim = prev_param[1]
                ##print(prev_fsim)
                prev_fsim[:-1] = flat_y0
                rec_dim = False
                ##print(prev_fsim)
            elif prev_param[1] is not None:
                prev_fsim = prev_param[1]
                x_iter = prev_param[2]
                print(x_iter)
                for key, value in x_iter.items():
                    if value is not None:
                        if key == 'x_shrink':
                            for k in range(len(x0)):
                                for j in range(len(value)):
                                    if np.array_equal(value[j][0], np.asarray(x0[k])):
                                        x_iter[key][j][1] = y0[k]
                        else:
                            for k in range(len(x0)):
                                if np.array_equal(value[0], np.asarray(x0[k])):
                                    x_iter[key][1] = y0[k]
                                    break
                print(x_iter)
            else:
                prev_fsim = y0
        elif prev_res is not None:
            prev_sim = x0
            prev_fsim = y0
            red_dim =  None
            red_sim = None
            red_fsim = None
            rec_dim = None
            memory = [[-100,-100]]

        # Run Nelder-Mead Simplex algorithm for one iteration
        if not red_dim:
            print("HERE")
            print(x_iter)
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=x0[0], bounds=bounds, x=x_iter, f=prev_fsim, sim=prev_sim)
            test_sim = np.asarray(sim[:-1])
            overfull_sim_dim = np.all(test_sim == test_sim[0, :], axis=0)
            overfull = False
            print(x_iter)
            #print(request)
            for i in range(len(overfull_sim_dim)):
                if overfull_sim_dim[i]:
                    ##print("MARKER")
                    ##print(request[0][i])
                    ##print(test_sim[0][i])
                    if request[0][i] == test_sim[0][i]:
                        overfull_dim = i
                        overfull = True
                        #print("OVER")

                        prev_sim = sim[:-1]
                        prev_fsim = fsim[:-1]
                        red_sim = sim
                        red_fsim = fsim
                        break
        # reduce dimension if n+1 points are located in n-1 dimensions
        if red_dim or overfull:
            print("HALLO")
            if red_dim:
                print(prev_sim)
                overfull_sim_dim = np.all(prev_sim == prev_sim[0, :], axis=0)
                overfull_dim = np.where(overfull_sim_dim)[0][0]
                print(overfull_sim_dim)
                print(overfull_dim)
                #print(x_iter)
                if x_iter:
                    for key, value in x_iter.items():
                        if value is not None:
                            if key is 'xbar':
                                x_iter[key] = np.delete(value,overfull_dim)
                                continue
                            if key is 'x_shrink':
                                for v in range(len(value)):
                                    x_iter[key][v] = [np.delete(value[v][0], overfull_dim), value[v][1]]
                                continue
                            ##print(x_iter[key])
                            x_iter[key] = [np.delete(value[0], overfull_dim), value[1]]
                    #print(x_iter)
            else:
                ##print("tRE")
                x_iter = None
            #overfull_dim = np.where(overfull_sim_dim)[0][0]
            save_dim = prev_sim[0][overfull_dim]
            print(save_dim)
            new_prev_sim = np.delete(prev_sim, overfull_dim, 1)
            ##print(x_iter)
            ##print(prev_fsim)
            new_prev_fsim = prev_fsim
            new_bounds = np.delete(bounds, overfull_dim,0)
            ##print(new_prev_sim)
            ##print(new_bounds)
            ##print("STOP - DIM")
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=new_prev_sim[0], x=x_iter, bounds=new_bounds, f=new_prev_fsim,
                                                                  sim=new_prev_sim)
            #print(request)
            request = np.insert(request, overfull_dim, save_dim, 1)
            sim = np.insert(sim, overfull_dim, save_dim, 1)
            #print(x_iter)
            for key, value in x_iter.items():
                if value is not None:
                    if key is 'xbar':
                        x_iter[key] = np.insert(value, overfull_dim, save_dim)
                        continue
                    if key is 'x_shrink':
                        for v in range(len(value)):
                            x_iter[key][v] = [np.insert(value[v][0], overfull_dim, save_dim), value[v][1]]
                        continue
                    ##print(x_iter[key])
                    x_iter[key] = [np.insert(value[0], overfull_dim, save_dim), value[1]]
            #print(request)
            #print(x_iter)
            red_dim = True
            ##print("STOP - DIM")
            red_dim = True
        else:
            red_dim = False
        #red_dim = False
        ##print(rec_dim)
        print("RE")
        print(sim)
        print(request)
        print(memory)
        if red_dim and any(np.equal(np.asarray(memory), request).all(1)):
            # recover dimension
            ##print(sim[0])
            ##print(red_sim[-1])
            ##print(red_sim)
            xr_red_dim = (red_sim[-1][overfull_dim] - red_sim[0][overfull_dim])
            ##print(xr_red_dim)
            ##print(red_sim[:-1][:,[0]]+1)
            new_sim = red_sim.copy()
            new_sim[:-1][:,[overfull_dim]] = red_sim[:-1][:,[overfull_dim]] + xr_red_dim
            for dim in range(len(red_sim[0])):
                if dim == overfull_dim:
                    continue
                else:
                    xt_red_dim = (red_sim[-1][dim] - sim[0][dim])
                    ##print(xt_red_dim)
                    for s in range(len(new_sim[:-1])):
                        xs = red_sim[s][dim] - xt_red_dim
                        if bounds[dim][0] > xs:
                            xs = bounds[dim][0]
                        elif bounds[dim][1] < xs:
                            xs = bounds[dim][1]
                        new_sim[s][dim] = xs
            new_sim[-1] = sim[0]
            ##print(new_sim)
            red_dim = False
            rec_dim = True
            sim = new_sim
            request = sim[:-1]
            fsim = red_fsim
            fsim[-1] = fsim[0]
            ##print("STOP STOP")
        memory.append(request.tolist()[0])
        print(memory)
        print(x_iter)
        res = [sim, fsim, x_iter, red_dim, red_sim, red_fsim, rec_dim, memory]


        # Generate DataSet object with variable values of next experiments
        next_experiments = {}
        for i, v in enumerate(self.domain.variables):
            if not v.is_objective:
                next_experiments[v.name] = request[:,i]
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
        next_experiments[('strategy', 'METADATA')] = ['Nelder-Mead Simplex']*len(request)
        ###print("\n ITER")
        ###print(next_experiments)
        ###print(res)
        return next_experiments, sim[0], fsim[0], res


    # https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
    def minimize_neldermead(self, x0, bounds, x=None, f=None, sim= None, initial_simplex= None, adaptive=False,
                             **unknown_options):
        """
        Minimization of scalar function of one or more variables using the
        Nelder-Mead algorithm.
        Options
        -------
        disp : bool
            Set to True to ##print convergence messages.
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

        x0 = np.asfarray(x0).flatten()
        N = len(x0)

        if initial_simplex is None and sim is None:
            ##print(1)
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
        elif sim is None:
            ##print(2)
            sim = np.asfarray(initial_simplex).copy()
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
            N = sim.shape[1]
        else:
            ##print(3)
            sim = np.asfarray(sim).copy()
            ##print(sim)
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
            N = sim.shape[1]

        one2np1 = list(range(1, N + 1))
        fsim = np.zeros((N + 1,), float)

        for k in range(N + 1):
            fsim[k] = f[k]

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
            ###print(x_iter)
            ###print(sim)
            ##print("HALLO")
            if not x_iter['xr']:
                # Centroid point: xbar
                xbar = np.add.reduce(sim[:-1], 0) / N
                x_iter['xbar'] = xbar
                # Reflection point xr
                xr = (1 + rho) * xbar - rho * sim[-1]
                ##print(sim)
                ##print(x_iter)
                for l in range(len(bounds)):
                    ##print(xr)
                    _bool, i, b = self.check_bounds(xr, bounds)
                    ##print(_bool)
                    ##print(i)
                    ##print(b)
                    if _bool:
                        break
                    else:
                        tmp_rho = (bounds[i][b] - xbar[i])/(xbar[i] - sim[-1][i])
                        xr = (1 + tmp_rho) * xbar - tmp_rho * sim[-1]
                x_iter['xr'] = [xr, None]
                return np.asarray([xr]), sim, fsim, x_iter
            xr = x_iter['xr'][0]
            fxr = x_iter['xr'][1]
            doshrink = 0

            # if function value of reflected point is better than best point of simplex, determine expansion point
            ##print(fxr)
            ##print(fsim[0])
            if fxr < fsim[0]:
                if not x_iter['xe']:
                    # expansion point: xe
                    xbar = x_iter['xbar']
                    #xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                    xe = xr + chi * xbar - chi * sim[-1]
                    ##print(sim)
                    ##print(x_iter)
                    for l in range(len(bounds)):
                        ##print(xe)
                        _bool, i, b = self.check_bounds(xe, bounds)
                        ##print(_bool)
                        ##print(i)
                        ##print(b)
                        if _bool:
                            break
                        else:
                            tmp_chi = (bounds[i][b] - xr[i])/(xbar[i] - sim[-1][i])
                            xe = xr + tmp_chi * xbar - tmp_chi * sim[-1]
                    if np.array_equal(xe,xr):
                        x_iter['xe'] = [xe, float("inf")]
                    else:
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
                            rho = np.min((xr- xbar)/(xbar - sim[-1]))
                            ##print("RHO")
                            ##print(rho)
                            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                            for l in range(len(bounds)):
                                _bool, i, b = self.check_bounds(xc, bounds)
                                if _bool:
                                    break
                                else:
                                    tmp_psi = (bounds[i][b] - xr[i]) / (xbar[i] - sim[-1][i])
                                    xc = (1 + tmp_psi * rho) * xbar - tmp_psi * rho * sim[-1]
                            if np.array_equal(xc,xr):
                                x_iter['xc'] = [xc, float("inf")]
                            else:
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
                            for l in range(len(bounds)):
                                _bool, i, b = self.check_bounds(xcc, bounds)
                                if _bool:
                                    break
                                else:
                                    tmp_psi = (bounds[i][b] - xbar[i])/(sim[-1][i] - xbar[i])
                                    xcc = (1 - tmp_psi) * xbar + tmp_psi * sim[-1]
                            if np.array_equal(xcc,xr):
                                x_iter['xcc'] = [xcc, None]
                            else:
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
                            ##print(x_iter["x_shrink"])
                            sim[j] = x_iter['x_shrink'][j-1][0]
                            fsim[j] = x_iter['x_shrink'][j-1][1]
            x_iter = {'xbar': None, 'xr': None, 'xe': None, 'xc': None, 'xcc': None, 'x_shrink': None}
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
        # end of iteration

        x = sim[0]
        return None, sim, fsim, None

    def check_bounds(self, x, bounds):
        for i, b in enumerate(bounds):
            upper_b = b[1] < x[i]
            lower_b = b[0] > x[i]
            if upper_b or lower_b:
                ##print(x)
                ##print("Point violated bound constraints")
                if lower_b:
                    return False, i, 0
                else:
                    return False, i, 1
        return True, None, None


