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
    x_start: array_like of shape (1, N), optional
        Initial center point of simplex
        Default: empty list that will initialize generation of x_start as geoemetrical center point of bounds
        Note that x_start is ignored when initial call of suggest_exp contains prev_res and/or prev_param

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

        self._x_start = kwargs.get('x_start', [])
        self._adaptive = kwargs.get('adaptive', False)

    def suggest_experiments(self, prev_res: DataSet=None, prev_param=None):
        """ Suggest experiments using Nelder-Mead Simplex method

        Parameters
        ----------
        x_start: np.array of size 1xdim, optional
            Initial center point for simplex
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
        initial_run = True
        x0 = [self._x_start]
        y0 = []

        # Get previous results
        if prev_res is not None:
            initial_run = False
            inputs, outputs = self.get_inputs_outputs(prev_res)

            # Set up maximization and minimization
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]

            x0 = inputs.data_to_numpy()
            y0 = outputs.data_to_numpy()

        elif prev_param is not None:
            raise ValueError('Parameter from previous optimization iteration are given but previous results are '
                             'missing!')

        # if no previous results are given initialize center point as geometrical middle point of bounds
        if not len(x0):
            x0 = np.ones((1,len(bounds)))*1/2*((bounds[:,1] + bounds[:,0]).T)

        ''' Set Nelder-Mead parameters, i.e., initialize or include data from previous iterations
            --------
            prev_sim: array-like
                variable coordinates (points) of simplex from previous run
            prev_fsim: array-like
                function values corresponding to points of simplex from previous run
            x_iter: array-like
                variable coordinates and corresponding function values of potential new 
                simplex points determined in one iteration of the NMS algorithm; note that 
                within one iteration multiple points need to be evaluated; that's why we have
                to store the points of an unfinished iteration (start iteration -> request point
                -> run experiment -> restart same iteration with results of experiment 
                -> request point -> run experiment ... -> finish iteration)
            red_dim: boolean
                True if dimension was reduced in one of the previous iteration and has not been recovered yet
            red_sim: array-like
                variable coordinates (points) of simplex before dimension was reduced
            red_fsim: array-like
                function values of points corresponding to simplex before dimension was reduced
            rec_dim: boolean
                True if dimension was revocered in last iteration
            memory: array-like
                list of all points for which the function was evaluated
        '''

        prev_sim, prev_fsim, x_iter, red_dim, red_sim, red_fsim, rec_dim, memory = \
            None, None, None, None, None, None, None, [[float("-inf"),float("inf")]]

        # if this is not the first iteration of the Nelder-Mead algorithm, get parameters from previous iteration
        if prev_param:
            prev_sim= prev_param[0]
            red_dim = prev_param[3]
            red_sim = prev_param[4]
            red_fsim = prev_param[5]
            rec_dim = prev_param[6]
            memory = prev_param[7]

            # if dimension was recovered in last iteration, N functions evaluations were requested
            # that need to be assigned to the respective points in the simplex
            if rec_dim:
                flat_y0 = [y0[i] for i in range(len(y0))]
                prev_fsim = prev_param[1]
                prev_fsim[:-1] = flat_y0
                rec_dim = False
            # assign function values to respective points
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
            else:
                prev_fsim = y0
                print("TEST")

        # initialize with given simplex points (including function evaluations) for initialization
        elif prev_res is not None:
            prev_sim = x0
            prev_fsim = y0

        # Run Nelder-Mead Simplex algorithm for one iteration
        overfull_simplex = False
        if not red_dim:
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=x0[0], bounds=bounds, x_iter=x_iter, f=prev_fsim, sim=prev_sim)
            '''if not initial_run:
                test_sim = np.asarray(sim[:-1])
                overfull_sim_dim = np.all(test_sim == test_sim[0, :], axis=0)
                for i in range(len(overfull_sim_dim)):
                    if overfull_sim_dim[i]:
                        if request[0][i] == test_sim[0][i]:
                            overfull_dim = i
                            overfull = True
                            prev_sim = sim[:-1]
                            prev_fsim = fsim[:-1]
                            red_sim = sim
                            red_fsim = fsim
                            break'''
            if not initial_run:
                overfull_simplex, prev_sim, prev_fsim, red_sim, red_fsim, overfull_dim = self.check_overfull(request, sim, fsim)

        # reduce dimension if n+1 points are located in n-1 dimensions
        if red_dim or overfull_simplex:
            if red_dim:
                overfull_sim_dim = np.all(prev_sim == prev_sim[0, :], axis=0)
                overfull_dim = np.where(overfull_sim_dim)[0][0]
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
                            x_iter[key] = [np.delete(value[0], overfull_dim), value[1]]
            else:
                x_iter = None

            save_dim = prev_sim[0][overfull_dim]
            new_prev_sim = np.delete(prev_sim, overfull_dim, 1)
            new_prev_fsim = prev_fsim
            new_bounds = np.delete(bounds, overfull_dim,0)
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=new_prev_sim[0], x_iter=x_iter, bounds=new_bounds, f=new_prev_fsim,
                                                                  sim=new_prev_sim)
            request = np.insert(request, overfull_dim, save_dim, 1)
            sim = np.insert(sim, overfull_dim, save_dim, 1)

            for key, value in x_iter.items():
                if value is not None:
                    if key is 'xbar':
                        x_iter[key] = np.insert(value, overfull_dim, save_dim)
                        continue
                    if key is 'x_shrink':
                        for v in range(len(value)):
                            x_iter[key][v] = [np.insert(value[v][0], overfull_dim, save_dim), value[v][1]]
                        continue
                    x_iter[key] = [np.insert(value[0], overfull_dim, save_dim), value[1]]
            red_dim = True
        else:
            red_dim = False

        if red_dim and any(np.equal(np.asarray(memory), request).all(1)):
            # recover dimension
            xr_red_dim = (red_sim[-1][overfull_dim] - red_sim[0][overfull_dim])
            new_sim = red_sim.copy()
            new_sim[:-1][:,[overfull_dim]] = red_sim[:-1][:,[overfull_dim]] + xr_red_dim
            for dim in range(len(red_sim[0])):
                if dim == overfull_dim:
                    continue
                else:
                    xt_red_dim = (red_sim[-1][dim] - sim[0][dim])
                    for s in range(len(new_sim[:-1])):
                        xs = red_sim[s][dim] - xt_red_dim
                        if bounds[dim][0] > xs:
                            xs = bounds[dim][0]
                        elif bounds[dim][1] < xs:
                            xs = bounds[dim][1]
                        new_sim[s][dim] = xs
            new_sim[-1] = sim[0]
            red_dim = False
            rec_dim = True
            sim = new_sim
            request = sim[:-1]
            fsim = red_fsim
            fsim[-1] = fsim[0]
        memory.append(request.tolist()[0])

        # store parameters of iteration as parameter array
        param = [sim, fsim, x_iter, red_dim, red_sim, red_fsim, rec_dim, memory]


        # Generate DataSet object with variable values of next experiments
        next_experiments = {}
        for i, v in enumerate(self.domain.variables):
            if not v.is_objective:
                next_experiments[v.name] = request[:,i]
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))
        next_experiments[('strategy', 'METADATA')] = ['Nelder-Mead Simplex']*len(request)
        return next_experiments, 0, 0, param


    # https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
    def minimize_neldermead(self, x0, bounds, x_iter=None, f=None, sim=None, initial_simplex=None, adaptive=False,
                             **unknown_options):
        """
        Minimization of scalar function of one or more variables using the
        Nelder-Mead algorithm.
        Options
        -------
        x0: array_like of shape (1, N)
        x_iter:
        f:
        sim:

        initial_simplex : array_like of shape (N + 1, N)
            Initial simplex. If given, overrides `x0`.
            ``initial_simplex[j,:]`` should contain the coordinates of
            the jth vertex of the ``N+1`` vertices in the simplex, where
            ``N`` is the dimension.
        adaptive : bool, optional
            Adapt algorithm parameters to dimensionality of problem. Useful for
            high-dimensional minimization [1].
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

        # TODO: discuss hyperparameter, find literature
        zdelt = 0.25

        x0 = np.asfarray(x0).flatten()
        N = len(x0)

        # generate N points based on center point, each point varying in
        # one different variable compared to center point -> initial simplex with N+1 points
        if initial_simplex is None and sim is None:
            sim = np.zeros((N + 1, N), dtype=x0.dtype)
            sim[0] = x0
            for k in range(N):
                y = np.array(x0, copy=True)
                y[k] = y[k] + zdelt * 1 / 2 * (bounds[k,1] - bounds[k,0])
                bool, _, _ = self.check_bounds(y, bounds)
                # if point violates bound restriction, change variable in opposite direction
                if not bool:
                    y[k] = y[k] - zdelt * (bounds[k, 1] - bounds[k, 0])
                # if point violates constraint, try opposite direction
                # if point violates other constraint or bound, calculate max zdelt <zdelt_mod> that meets
                # constraint for both direction and choose direction with greater zdelt_mod
                # TODO: check constraints
                sim[k + 1] = y
            return sim, sim, None, None
        elif sim is None:
            sim = np.asfarray(initial_simplex).copy()
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
                raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
            if len(x0) != sim.shape[1]:
                raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
            N = sim.shape[1]
        else:
            sim = np.asfarray(sim).copy()
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
        sim = np.take(sim, ind, 0)

        # Catch information on previous experiment
        if not x_iter:
            x_iter = {'xbar': None, 'xr': None, 'xe': None, 'xc': None, 'xcc': None, 'x_shrink': None}

        # Iteration
        while 1:
            if not x_iter['xr']:
                # Centroid point: xbar
                xbar = np.add.reduce(sim[:-1], 0) / N
                x_iter['xbar'] = xbar
                # Reflection point xr
                xr = (1 + rho) * xbar - rho * sim[-1]
                for l in range(len(bounds)):
                    _bool, i, b = self.check_bounds(xr, bounds)
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
            if fxr < fsim[0]:
                if not x_iter['xe']:
                    # expansion point: xe
                    xbar = x_iter['xbar']
                    xe = xr + chi * xbar - chi * sim[-1]
                    for l in range(len(bounds)):
                        _bool, i, b = self.check_bounds(xe, bounds)
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
                            # avoid division with zero (some coordinates of xbar, xr, and sim[-1] may be identical)
                            # by applying np.max and np.min
                            rho = np.min(np.max(xr- xbar)/np.max((xbar - sim[-1])))
                            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                            for l in range(len(bounds)):
                                _bool, i, b = self.check_bounds(xc, bounds)
                                if _bool:
                                    break
                                else:
                                    tmp_psi = (bounds[i][b] - xr[i]) / (xbar[i] - sim[-1][i])
                                    print(tmp_psi)
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
                            for j in one2np1:
                                sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                                xj = sim[j]
                                x_shrink.append(xj)
                                x_shrink_f.append([xj, None])
                            x_iter['x_shrink'] = x_shrink_f
                            return np.asarray(x_shrink), sim, fsim, x_iter
                        for j in one2np1:
                            sim[j] = x_iter['x_shrink'][j-1][0]
                            fsim[j] = x_iter['x_shrink'][j-1][1]
            x_iter = {'xbar': None, 'xr': None, 'xe': None, 'xc': None, 'xcc': None, 'x_shrink': None}
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)


    # Function to check whether a point x lies within the variable bounds of the domain
    def check_bounds(self, x, bounds):
        for i, b in enumerate(bounds):
            upper_b = b[1] < x[i]
            lower_b = b[0] > x[i]
            # Point violated bound constraints
            if upper_b or lower_b:
                if lower_b:
                    return False, i, 0
                else:
                    return False, i, 1
        return True, None, None


    # Function to check whether a point meets the constraints of the domain
    def check_constraints(self, x, constraints):
        raise NotImplementedError("Constraints not implemented yet")
        return True, None, None

    def check_overfull(self, tmp_request, tmp_sim, tmp_fsim):
        test_sim = np.asarray(tmp_sim[:-1])
        overfull_sim_dim = np.all(test_sim == test_sim[0, :], axis=0)
        for i in range(len(overfull_sim_dim)):
            if overfull_sim_dim[i]:
                if tmp_request[0][i] == test_sim[0][i]:
                    overfull_dim = i
                    prev_sim = tmp_sim[:-1]
                    prev_fsim = tmp_fsim[:-1]
                    red_sim = tmp_sim
                    red_fsim = tmp_fsim
                    return True, prev_sim, prev_fsim, red_sim, red_fsim, overfull_dim
                    break
        return False, None, None, None, None, None



