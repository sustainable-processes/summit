from .base import Strategy
from summit.domain import Domain, DomainError
from summit.utils.dataset import DataSet

import numpy as np
import pandas as pd

class NelderMead(Strategy):
    ''' A reimplementation of the Nelder-Mead Simplex method adapted for sequential calls.
    This includes adaptions in terms of reflecting points, dimension reduction and dimension recovery
    proposed by Cortes-Borda et al. [1].

    Parameters
    ----------
    domain: `summit.domain.Domain`
        A summit domain object
    x_start: array_like of shape (1, N), optional
        Initial center point of simplex
        Default: empty list that will initialize generation of x_start as geoemetrical center point of bounds
        Note that x_start is ignored when initial call of suggest_exp contains prev_res and/or prev_param
    dx: float, optional
        Parameter for stopping criterion: two points are considered
        to be different if they differ by at least dx(i) in at least one
        coordinate i.
        Default is 1E-5.
    df: float, optional
        Parameter for stopping criterion: two function values are considered
        to be different if they differ by at most df.
        Default is 1E-5.

    Notes
    ----------
    Implementation partly follows the Nelder-Mead Simplex implementation in scipy-optimize:
    https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py

    References
    ----------
    .. [1] Cortés-Borda, D.; Kutonova, K. V.; Jamet, C.; Trusova, M. E.; Zammattio, F.;
    Truchet, C.; Rodriguez-Zubiri, M.; Felpin, F.-X. Optimizing the Heck–Matsuda Reaction
    in Flow with a Constraint-Adapted Direct Search Algorithm.
    Organic ProcessResearch & Development 2016,20, 1979–1987

    Examples
    -------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import NelderMead
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> strategy = NelderMead(domain)
    >>> next_experiments, xbest, fbest, param = strategy.suggest_experiments()
    >>> print(next_experiments)
    NAME  temperature  flowrate_a             strategy
    0           0.500       0.500  Nelder-Mead Simplex
    1           0.625       0.500  Nelder-Mead Simplex
    2           0.500       0.625  Nelder-Mead Simplex

    '''

    def __init__(self, domain: Domain, **kwargs):
        Strategy.__init__(self, domain)

        self.domain = domain
        self._x_start = kwargs.get('x_start', [])
        self._dx = kwargs.get('dx', 1E-5)
        self._df = kwargs.get('df', 1E-5)
        self._adaptive = kwargs.get('adaptive', False)

    def suggest_experiments(self, prev_res: DataSet=None, prev_param=None):

        # get objective name and whether optimization is maximization problem
        obj_name = None
        obj_maximize = False
        for v in self.domain.variables:
            i = 0
            if v.is_objective:
                i += 1
                if i > 1:
                    raise ValueError("Nelder-Mead is not able to optimize multiple objectives.")
                obj_name = v.name
                if v.maximize:
                    obj_maximize = True

        # get results from conducted experiments
        if prev_res is not None:
            prev_res = prev_res

        # get parameters from previous iterations
        inner_prev_param = None
        if prev_param is not None:
            # get parameters for Nelder-Mead from previous iterations
            inner_prev_param = prev_param[0]
            # recover invalid experiments from previous iteration
            if prev_param[1] is not None:
                invalid_res = prev_param[1][0].drop(('constraint','DATA'),1)
                prev_res = pd.concat([prev_res,invalid_res])

        ## Generation of new suggested experiments.
        # An inner function is called loop-wise to get valid experiments and
        # avoid suggestions of experiments that violate constraints.
        # If no valid experiment is found after #<inner_iter_tol>, an error is raised.
        inner_iter_tol = 5
        c_iter = 0
        valid_next_experiments = False
        next_experiments = None
        while not valid_next_experiments and c_iter < inner_iter_tol:
            valid_next_experiments = False
            next_experiments, xbest, fbest, param = self.inner_suggest_experiments(prev_res=prev_res, prev_param=inner_prev_param)
            invalid_experiments = next_experiments.loc[next_experiments[('constraint','DATA')] == False]
            next_experiments = next_experiments.loc[next_experiments[('constraint','DATA')] != False]
            prev_res = prev_res
            if len(next_experiments) and len(invalid_experiments):
                valid_next_experiments = True
                if obj_maximize:
                    invalid_experiments[(obj_name, 'DATA')] = float("-inf")
                else:
                    invalid_experiments[(obj_name, 'DATA')] = float("inf")
            #
            elif len(invalid_experiments):
                if obj_maximize:
                    invalid_experiments[(obj_name, 'DATA')] = float("-inf")
                else:
                    invalid_experiments[(obj_name, 'DATA')] = float("inf")
                prev_res = invalid_experiments
            else:
                valid_next_experiments = True
            inner_prev_param = param
            param = [param, [invalid_experiments]]
            c_iter += 1

        # return only valid experiments (invalid experiments are stored in param[1])
        next_experiments = next_experiments.drop(('constraint', 'DATA'), 1)
        return next_experiments, xbest, fbest, param

    def inner_suggest_experiments(self, prev_res: DataSet=None, prev_param=None):
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

        # intern
        stay_inner = False

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
        if not len(x0[0]):
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
            None, None, None, None, None, None, None, [np.ones(dim)*float("inf")]

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
                prev_fsim = prev_param[1]
                for k in range(len(x0)):
                    for s in range(len(prev_sim)):
                        if np.array_equal(prev_sim[s], x0[k]):
                            prev_fsim[s] = y0[k]
                rec_dim = False
            # assign function values to respective points
            elif prev_param[1] is not None:
                prev_fsim = prev_param[1]
                x_iter = prev_param[2]
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
        # initialize with given simplex points (including function evaluations) for initialization
        elif prev_res is not None:
            prev_sim = x0
            prev_fsim = y0
            for p in x0.astype(float).tolist():
                memory.append(p)

        # Run Nelder-Mead Simplex algorithm for one iteration
        overfull_simplex = False
        if not red_dim:
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=x0[0], bounds=bounds, x_iter=x_iter, f=prev_fsim, sim=prev_sim,adaptive=self._adaptive)
            if not initial_run:
                overfull_simplex, prev_sim, prev_fsim, red_sim, red_fsim, overfull_dim = self.check_overfull(request, sim, fsim, bounds)

        ## Reduce dimension if n+1 points are located in n-1 dimensions (if either red_dim = True, i.e.,
        # optimization in the reduced dimension space was not finished in the last iteration, or overfull_simplex, i.e.,
        # last Nelder-Mead call (with red_dim = False) lead to an overfull simplex).
        ## Note that in order to not loose any information, the simplex without dimension reduction is returned even
        # if the optimization in the reduced dimension space is not finished.
        ## If the optimization in the reduced dimension space was not finished in the last iteration (red_dim = True),
        # the simplex will automatically be reduced again.
        if red_dim or overfull_simplex:
            # prepare dimension reduction
            if red_dim:
                x_iter, overfull_dim = self.upstream_simplex_dim_red(prev_sim, x_iter)
            else:
                x_iter = None

            # save value of dimension reduced
            save_dim = prev_sim[0][overfull_dim]
            # delete overfull dimension
            new_prev_sim = np.delete(prev_sim, overfull_dim, 1)
            # delete bounds for overfull dimension
            new_bounds = np.delete(bounds, overfull_dim,0)

            # Run one iteration of Nelder-Mead Simplex algorithm for reduced simplex
            request, sim, fsim, x_iter = self.minimize_neldermead(x0=new_prev_sim[0], x_iter=x_iter, bounds=new_bounds, f=prev_fsim,
                                                                  sim=new_prev_sim, adaptive=self._adaptive)

            overfull_simplex, _, _, _, _, _ = self.check_overfull(request, sim, fsim, bounds)
            if overfull_simplex:
                raise NotImplementedError("Recursive dimension reduction not implemented yet.")

            # recover dimension after Nelder-Mead Simplex run (to return full request for experiment)
            request = np.insert(request, overfull_dim, save_dim, 1)
            sim = np.insert(sim, overfull_dim, save_dim, 1)

            # follow-up of dimension reduction
            x_iter = self.downstream_simplex_dim_red(x_iter, overfull_dim, save_dim)

            red_dim = True

        # if not overfull and no reduced dimension from previous iteration
        else:
            red_dim = False

        # Circle (suggested point that already has been investigated)
        if any((memory == x).all(1).any() for x in request):
            ## if dimension is reduced and requested point has already been evaluated, recover dimension with
            # reflected and translated simplex before dimension reduction
            if red_dim:
                sim, fsim, request = self.recover_simplex_dim(sim, red_sim, red_fsim, overfull_dim, bounds, memory, self._dx)
                red_dim = False
                rec_dim = True
            # raise error
            else:
                stay_inner = True
                #raise NotImplementedError("Circle - point has already been investigated.")


        ## Only little changes in requested points, xatol = tolerance for changes in x,
        # or in function values, fatol = tolerance for changes in f
        ## TODO: add extra threshold to stop reduced dimension problem and recover dimension
        if not initial_run:
            xatol = (bounds[:, 1] - bounds[:, 0]) * self._dx
            fatol = self._df
            if (np.max(np.abs(sim[1:] - sim[0]),0) <= xatol).all() or (np.max(np.abs(fsim[0] - fsim[1:])) <= fatol).any():
                if red_dim:
                    sim, fsim, request = self.recover_simplex_dim(sim, red_sim, red_fsim, overfull_dim, bounds, memory, self._dx)
                    red_dim = False
                    rec_dim = True
                else:
                    stopping_error = 'Stopping criterion is reached.'
                    raise ValueError(stopping_error)

        # add requested points to memory
        for p in request.astype(float).tolist():
            memory.append(p)

        # store parameters of iteration as parameter array
        param = [sim, fsim, x_iter, red_dim, red_sim, red_fsim, rec_dim, memory]

        # Generate DataSet object with variable values of next experiments
        next_experiments = {}
        for i, v in enumerate(self.domain.variables):
            if not v.is_objective:
                next_experiments[v.name] = request[:,i]
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))

        # Violate constraint
        mask_valid_next_experiments = self.check_constraints(next_experiments)
        if initial_run and not all(mask_valid_next_experiments):
            raise ValueError("Default initialization failed due to constraints. Please enter an initial simplex with feasible points")
        if not any(mask_valid_next_experiments):
            stay_inner = True

        if stay_inner:
            # add infinity as
            next_experiments[('constraint', 'DATA')] = False
        else:
            # add optimization strategy
            next_experiments[('constraint', 'DATA')] = mask_valid_next_experiments
            next_experiments[('strategy', 'METADATA')] = ['Nelder-Mead Simplex'] * len(request)

        x_best = None
        f_best = float("inf")
        if not initial_run:
            x_best = sim[0]
            f_best = fsim[0]
            x_best = self.round(x_best, bounds, self._dx)
            #f_best = np.around(f_best, decimals=self._dx)
        #next_experiments = np.around(next_experiments, decimals=self._dx)

        return next_experiments, x_best, f_best, param


    # implementation partly follows: https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
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
                xbar = self.round(xbar, bounds, self._dx)
                x_iter['xbar'] = xbar
                # Reflection point xr
                xr = (1 + rho) * xbar - rho * sim[-1]
                for l in range(len(bounds)):
                    _bool, i, b = self.check_bounds(xr, bounds)
                    if _bool:
                        break
                    else:
                        tmp_rho = np.min(np.max(np.abs((bounds[i][b] - xbar[i])))/np.max(np.abs((xbar[i] - sim[-1][i]))))
                        xr = (1 + tmp_rho) * xbar - tmp_rho * sim[-1]
                xr = self.round(xr, bounds, self._dx)
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
                            tmp_chi = np.min(np.max(np.abs((bounds[i][b] - xr[i])))/np.max(np.abs((xbar[i] - sim[-1][i]))))
                            xe = xr + tmp_chi * xbar - tmp_chi * sim[-1]
                    xe = self.round(xe, bounds, self._dx)
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
                            rho = np.min(np.max(np.abs(xr- xbar))/np.max(np.abs((xbar - sim[-1]))))
                            xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                            for l in range(len(bounds)):
                                _bool, i, b = self.check_bounds(xc, bounds)
                                if _bool:
                                    break
                                else:
                                    tmp_psi = np.min(np.max(np.abs((bounds[i][b] - xr[i]))) / np.max(np.abs((xbar[i] - sim[-1][i]))))
                                    xc = (1 + tmp_psi * rho) * xbar - tmp_psi * rho * sim[-1]
                            xc = self.round(xc, bounds, self._dx)
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
                            xcc = self.round(xcc, bounds, self._dx)
                            for l in range(len(bounds)):
                                _bool, i, b = self.check_bounds(xcc, bounds)
                                if _bool:
                                    break
                                else:
                                    tmp_psi = np.min(np.max(np.abs((bounds[i][b] - xbar[i])))/np.max(np.abs((sim[-1][i] - xbar[i]))))
                                    xcc = (1 - tmp_psi) * xbar + tmp_psi * sim[-1]
                            xcc = self.round(xcc, bounds, self._dx)
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
                                xj = self.round(xj, bounds, self._dx)
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
    def check_constraints(self, tmp_next_experiments):
        input_columns = [v.name for  v in self.domain.variables if not v.is_objective]
        tmp_next_experiments = DataSet(np.atleast_2d(tmp_next_experiments), columns=input_columns)
        constr_mask = np.asarray([True]*len(tmp_next_experiments)).T
        if self.domain.constraints:
            constr = [c.constraint_type + "0" for c in self.domain.constraints]
            constr_mask = [tmp_next_experiments.eval(c.lhs + constr[i], resolvers=[tmp_next_experiments]) for i, c in enumerate(self.domain.constraints)]
            constr_mask = [c.tolist() for c in constr_mask][0]
        return constr_mask

    # Function to check whether a simplex contains only points that are identical in one dimension and the
    # the variable value fo this dimension corresponds to the bound value
    def check_overfull(self, tmp_request, tmp_sim, tmp_fsim, bounds):
        test_sim = np.asarray(tmp_sim[:-1])
        overfull_sim_dim = np.all(test_sim == test_sim[0, :], axis=0)
        #print(tmp_request)
        #print(tmp_sim)
        for i in range(len(overfull_sim_dim)):
            if overfull_sim_dim[i]:
                #print(i)
                if tmp_request[0][i] == test_sim[0][i]:
                    if any(bounds[i] == test_sim[0][i]):
                        overfull_dim = i
                        prev_sim = tmp_sim[:-1]
                        prev_fsim = tmp_fsim[:-1]
                        red_sim = tmp_sim
                        red_fsim = tmp_fsim
                        return True, prev_sim, prev_fsim, red_sim, red_fsim, overfull_dim
                    else:
                        raise ValueError("Simplex is overfull in one dimension. Please increase threshold for stopping.")
        return False, None, None, None, None, None

    # Prepare Nelder-Mead parameters and previous results for dimension reduction by removing overfull dimension
    def upstream_simplex_dim_red(self, tmp_prev_sim, tmp_x_iter):
        tmp_x_iter = tmp_x_iter
        overfull_sim_dim = np.all(tmp_prev_sim == tmp_prev_sim[0, :], axis=0)
        overfull_dim = np.where(overfull_sim_dim)[0][0]
        if tmp_x_iter:
            for key, value in tmp_x_iter.items():
                if value is not None:
                    if key is 'xbar':
                        tmp_x_iter[key] = np.delete(value, overfull_dim)
                        continue
                    if key is 'x_shrink':
                        for v in range(len(value)):
                            tmp_x_iter[key][v] = [np.delete(value[v][0], overfull_dim), value[v][1]]
                        continue
                    tmp_x_iter[key] = [np.delete(value[0], overfull_dim), value[1]]
            return tmp_x_iter, overfull_dim
        else:
            return None, overfull_dim

    # Restore simplex after one call of Nelder-Mead with reduced dimension by adding overfull dimension.
    ## Note that if dimension reduction process is not finished, the simplex will reduced in the
    #  next Nelder-Mead call again.
    def downstream_simplex_dim_red(self, tmp_x_iter, overfull_dim, save_dim):
        for key, value in tmp_x_iter.items():
            if value is not None:
                if key is 'xbar':
                    tmp_x_iter[key] = np.insert(value, overfull_dim, save_dim)
                    continue
                if key is 'x_shrink':
                    for v in range(len(value)):
                        tmp_x_iter[key][v] = [np.insert(value[v][0], overfull_dim, save_dim), value[v][1]]
                    continue
                tmp_x_iter[key] = [np.insert(value[0], overfull_dim, save_dim), value[1]]
        return tmp_x_iter

    ## Reflect and translate simplex from iteration before dimension with respect to the point that was found in the
    #  reduced dimension problem.
    def recover_simplex_dim(self, tmp_sim, tmp_red_sim, tmp_red_fsim, overfull_dim, bounds, memory, dx):
        ## Translate all points of the simplex before the reduction along the axis of the reduced dimension
        # but the one, that caused dimension reduction (translation distance corresponds to distance of point, that
        # caused the dimension reduction, to the values of all other points at axis of the reduced dimension)
        xr_red_dim = (tmp_red_sim[-1][overfull_dim] - tmp_red_sim[0][overfull_dim])
        xr_red_dim = self.round(xr_red_dim, np.asarray([len(bounds)*[float("-inf"),float("inf")]]), dx)
        new_sim = tmp_red_sim.copy()
        new_sim[:-1][:, [overfull_dim]] = tmp_red_sim[:-1][:, [overfull_dim]] + xr_red_dim

        ## Translate all points of the simplex before the reduction along the remaining axes but the one, that caused
        # dimension reduction (translation distance corresponds to distance of point, that caused the dimension
        # reduction, to optimal point found in reduced space optimization)
        for dim in range(len(tmp_red_sim[0])):
            if dim == overfull_dim:
                continue
            else:
                xt_red_dim = (tmp_red_sim[-1][dim] - tmp_sim[0][dim])
                xt_red_dim = self.round(xt_red_dim, np.asarray([len(bounds)*[float("-inf"),float("inf")]]), dx)
                for s in range(len(new_sim[:-1])):
                    xs = tmp_red_sim[s][dim] - xt_red_dim
                    # TODO: check bounds here, what happens if more points violate bound constraints)
                    if bounds[dim][0] > xs:
                        xs = bounds[dim][0]
                    elif bounds[dim][1] < xs:
                        xs = bounds[dim][1]
                    new_sim[s][dim] = xs
        # Alter simplex in case one point is twice into recovered simplex due to bound constraints
        p = 0
        c_i = 0
        while p < len(new_sim) and c_i < len(new_sim):
            l_new_sim = new_sim.tolist()
            x = l_new_sim.count(l_new_sim[p])
            if x > 1:
                t_x = l_new_sim[p]
                for dim in range(len(t_x)):
                    if t_x[dim] == bounds[dim,0]:
                        new_sim[p][dim] = new_sim[p][dim] + 0.25 * 1 / 2 * (bounds[dim,1] - bounds[dim,0])
                        new_sim[p] = self.round(new_sim[p], bounds, self._dx)

                        p = 0
                        c_i += 1
                    elif t_x[dim] == bounds[dim,1]:
                        new_sim[p][dim] = new_sim[p][dim] - 0.25 * 1 / 2 * (bounds[dim, 1] - bounds[dim, 0])
                        new_sim[p] = self.round(new_sim[p], bounds, self._dx)
                        p = 0
                        c_i += 1
                    else:
                        c_i += 1
            else:
                p += 1

        new_sim[-1] = tmp_sim[0]
        sim = new_sim
        fsim = tmp_red_fsim
        fsim[-1] = fsim[0]
        request = sim[:-1]
        if any((memory == x).all(1).any() for x in request):
            len_req = len(request)
            len_req_mod = len_req
            i = 0
            while i < len_req_mod:
                if (memory == request[i]).all(1).any():
                    fsim[i + len_req - len_req_mod] = float("inf")
                    request = np.delete(request, i, 0)
                    len_req_mod -= 1
                else:
                    i += 1
            if len_req_mod == 0:
                raise ValueError("Recovering dimension failed due to error in generating new points. " \
                      "Please increase threshold for stopping.")
        return sim, fsim, request

    # adapted from the SQSnobFit package
    def round(self, x, bounds, dx):
        """
          function x = round(x, bounds, dx)

          A point x is projected into the interior of [u, v] and x[i] is
          rounded to the nearest integer multiple of dx[i].

          Input:
          x         vector of length n
          bounds    matrix of length nx2 such that bounds[:,0] < bounds[:,1]
          dx        float

          Output:
          x         projected and rounded version of x
        """
        u = bounds[:,0]
        v = bounds[:,1]

        x = np.minimum(np.maximum(x, u), v)
        x = np.round(x / dx) * dx
        i1 = self.find(x < u)

        if i1.size > 0:
            x[i1] = x[i1] + dx

        i2 = self.find(x > v)
        if i2.size > 0:
            x[i2] = x[i2] - dx

        return x

    # adapted from the SQSnobFit package
    def find(self, cond_array):
        return (np.transpose(np.nonzero(cond_array.flatten()))).astype(int)