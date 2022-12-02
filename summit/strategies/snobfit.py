import pdb
from .base import Strategy, Transform
from summit.domain import *
from summit.utils.dataset import DataSet

import math
import numpy
from copy import deepcopy
import pandas as pd
import warnings


class SNOBFIT(Strategy):
    """Stable Noisy Optimization by Branch and Fit (SNOBFIT)

    SNOBFIT is designed to quickly optimise noisy functions.

    Parameters
    ----------
    domain : :class:`~summit.domain.Domain`
        The domain of the optimization
    transform : :class:`~summit.strategies.base.Transform`, optional
        A transform object. By default no transformation will be done
        on the input variables or objectives.
    probability_p: float, optional
        The probability p that a point of class 4 is generated, i.e., higher p
        leads to more exploration.
        Default is 0.5.
    dx_dim: float, optional
        only used for the definition of a new problem: two points are considered
        to be different if they differ by at least dx(i) in at least one
        coordinate i.
        Default is 1E-5.


    Examples
    --------

    >>> from summit.domain import Domain, ContinuousVariable
    >>> from summit.strategies import SNOBFIT
    >>> from summit.utils.dataset import DataSet
    >>> import pandas as pd
    >>> domain = Domain()
    >>> domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 100])
    >>> domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    >>> domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.9])
    >>> domain += ContinuousVariable(name="yld", description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    >>> d = {'temperature': [50,40,70,30], 'flowrate_a': [0.6,0.3,0.2,0.1], 'flowrate_b': [0.1,0.3,0.2,0.1], 'yld': [0.7,0.6,0.3,0.1]}
    >>> df = pd.DataFrame(data=d)
    >>> initial = DataSet.from_df(df)
    >>> strategy = SNOBFIT(domain)
    >>> next_experiments = strategy.suggest_experiments(5, initial)

    Notes
    ------

    SNOBFIT was created by [Huyer]_ et al. This implementation is based on the python reimplementation [SQSnobFit]_
    of the original MATLAB code by [Neumaier]_.


    Note that SNOBFIT sometimes returns more experiments than requested when the number of experiments
    request is small (i.e., 1 or 2). This seems to be a general issue with the algorithm
    instead of the specific implementation used here.


    References
    ----------

    .. [Huyer] W. Huyer et al., ACM Trans. Math. Softw., 2008, 35, 1–25.
       DOI: `10.1145/1377612.1377613 <https://doi.org/10.1145/1377612.1377613>`_.

    .. [SQSnobFit] Lavrijsen, W. SQSnobFit `<https://pypi.org/project/SQSnobFit/>`_

    .. [Neumaier] `<https://www.mat.univie.ac.at/~neum/software/snobfit/>`_

    """

    def __init__(self, domain: Domain, **kwargs):
        Strategy.__init__(self, domain, **kwargs)

        self._p = kwargs.get("probability_p", 0.5)
        self._dx_dim = kwargs.get("dx_dim", 1e-5)
        self.prev_param = None

    def suggest_experiments(
        self, num_experiments=1, prev_res: DataSet = None, **kwargs
    ):
        """Suggest experiments using the SNOBFIT method

        Parameters
        ----------
        num_experiments: int, optional
            The number of experiments (i.e., samples) to generate. Default is 1.
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, the SNOBFIT optimization algorithm
            will be initialized and suggest initial experiments.
        Returns
        -------
        next_experiments: DataSet
            A `Dataset` object with the suggested experiments by SNOBFIT algorithm

        """
        silence_warnings = kwargs.get("silence_warnings", True)
        if silence_warnings:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        # get objective name and whether optimization is maximization problem
        obj_name = None
        obj_maximize = False
        for v in self.domain.variables:
            i = 0
            if v.is_objective:
                i += 1
                if i > 1:
                    raise ValueError(
                        "SNOBFIT is not able to optimize multiple objectives, please use transform."
                    )
                obj_name = v.name
                if v.maximize:
                    obj_maximize = True

        # get parameters from previous iterations
        inner_prev_param = None
        if self.prev_param is not None:
            # get parameters for Nelder-Mead from previous iterations
            inner_prev_param = self.prev_param[0]
            # recover invalid experiments from previous iteration
            if self.prev_param[1] is not None:
                invalid_res = self.prev_param[1][0].drop(("constraint", "DATA"), axis=1)
                prev_res = pd.concat([prev_res, invalid_res])

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
            next_experiments, xbest, fbest, param = self._inner_suggest_experiments(
                num_experiments=num_experiments,
                prev_res=prev_res,
                prev_param=inner_prev_param,
            )
            # Invalid experiments hidden from data returned to user but stored internally elswehere
            invalid_experiments = next_experiments.loc[
                next_experiments[("constraint", "DATA")] == False
            ]
            next_experiments = next_experiments.loc[
                next_experiments[("constraint", "DATA")] != False
            ]
            prev_res = prev_res
            if len(next_experiments) and len(invalid_experiments):
                valid_next_experiments = True
                # pass NaN if at least one constraint is violated
                invalid_experiments[(obj_name, "DATA")] = numpy.nan
            elif len(invalid_experiments):
                # pass NaN if at least one constraint is violated
                invalid_experiments[(obj_name, "DATA")] = numpy.nan
                prev_res = invalid_experiments
            else:
                valid_next_experiments = True
            inner_prev_param = param
            param = [param, [invalid_experiments]]
            c_iter += 1

        if c_iter >= inner_iter_tol:
            warnings.warn(
                "No new points found. Internal stopping criterion is reached."
            )

        # return only valid experiments (invalid experiments are stored in param[1])
        next_experiments = next_experiments.drop(("constraint", "DATA"), axis=1)
        objective_dir = -1.0 if obj_maximize else 1.0
        self.fbest = objective_dir * fbest
        self.prev_param = param
        self.xbest = xbest
        return next_experiments

    def reset(self):
        """Reset internal parameters"""
        self.prev_param = None

    def to_dict(self):
        """Convert hyperparameters and internal state to a dictionary"""
        if self.prev_param is not None:
            params = deepcopy(self.prev_param)
            params[0] = (params[0][0].tolist(), params[0][1], params[0][2].tolist())
            params[1] = [p.to_dict() for p in params[1]]
        else:
            params = None
        strategy_params = dict(
            probability_p=self._p, dx_dim=self._dx_dim, prev_param=params
        )
        return super().to_dict(**strategy_params)

    @classmethod
    def from_dict(cls, d):
        snobfit = super().from_dict(d)
        params = d["strategy_params"]["prev_param"]
        if params is not None:
            params[0] = (
                numpy.array(params[0][0]),
                params[0][1],
                numpy.array(params[0][2]),
            )
            params[1] = [DataSet.from_dict(p) for p in params[1]]
        snobfit.prev_param = params
        return snobfit

    def _inner_suggest_experiments(
        self, num_experiments, prev_res: DataSet = None, prev_param=None
    ):
        """Inner loop for generation of suggested experiments using the SNOBFIT method
        Parameters
        ----------
        num_experiments: int
            The number of experiments (i.e., samples) to generate
        prev_res: summit.utils.data.DataSet, optional
            Dataset with data from previous experiments.
            If no data is passed, the SNOBFIT optimization algorithm
            will be initialized will suggest initial experiments.
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

        # intern
        stay_inner = False

        # Get bounds of input variables
        bounds = []
        input_var_names = []
        output_var_names = []
        for v in self.domain.variables:
            if not v.is_objective:
                if isinstance(v, ContinuousVariable):
                    bounds.append(v.bounds)
                    input_var_names.append(v.name)
                elif isinstance(v, CategoricalVariable):
                    if v.ds is not None:
                        descriptor_names = v.ds.data_columns
                        descriptors = numpy.asarray(
                            [
                                v.ds.loc[:, [l]].values.tolist()
                                for l in v.ds.data_columns
                            ]
                        )
                    else:
                        raise ValueError("No descriptors given for {}".format(v.name))
                    for d in descriptors:
                        bounds.append(
                            [numpy.min(numpy.asarray(d)), numpy.max(numpy.asarray(d))]
                        )
                    input_var_names.extend(descriptor_names)
                else:
                    raise TypeError(
                        "SNOBFIT can not handle variable type: {}".format(v.type)
                    )
            else:
                output_var_names.extend(v.name)
        bounds = numpy.asarray(bounds, dtype=float)

        # Initialization
        x0 = []
        y0 = []

        # Get previous results
        if prev_res is not None:
            # get always the same order according to the ordering in the domain -> this is actually done within transform
            # ordered_var_names = input_var_names + output_var_names
            # prev_res = prev_res[ordered_var_names]
            # transform
            inputs, outputs = self.transform.transform_inputs_outputs(
                prev_res, categorical_method="descriptors"
            )

            # Set up maximization and minimization
            for v in self.domain.variables:
                if v.is_objective and v.maximize:
                    outputs[v.name] = -1 * outputs[v.name]

            x0 = inputs.data_to_numpy().astype(float)
            y0 = outputs.data_to_numpy().astype(float)

            # Add uncertainties to measurements TODO: include uncertainties in input
            y = []
            for i in range(y0.shape[0]):
                y.append([y0[i].tolist()[0], math.sqrt(numpy.spacing(1))])
            y0 = numpy.asarray(y, dtype=float)
        # If no prev_res are given but prev_param -> raise error
        elif prev_param is not None:
            raise ValueError(
                "Parameter from previous optimization iteration are given but previous results are "
                "missing!"
            )

        # if no previous results are given initialize with empty lists
        if not len(x0):
            x0 = numpy.array(x0).reshape(0, len(bounds)).astype(float)
            y0 = numpy.array(y0).reshape(0, 2).astype(float)

        """ Determine SNOBFIT parameters
          config       structure variable defining the box [u,v] in which the
                       points are to be generated, the number nreq of
                       points to be generated and the probability p that a
                       point of type 4 is generated
                       config = struct('bounds',{u,v},'nreq',nreq,'p',p)
          dx           only used for the definition of a new problem (when
                       the program should continue from the values stored in
                       file.mat, the call should have only 4 input parameters!)
                       n-vector (n = dimension of the problem) of minimal
                       stnumpy.spacing(1), i.e., two points are considered to be different
                       if they differ by at least dx(i) in at least one
                       coordinate i
        """
        config = {"bounds": bounds, "p": self._p, "nreq": num_experiments}
        dx = (bounds[:, 1] - bounds[:, 0]) * self._dx_dim

        # Run SNOBFIT for one iteration
        request, xbest, fbest, param = self.snobfit(x0, y0, config, dx, prev_param)

        # Generate DataSet object with variable values of next experiments
        next_experiments = {}
        for i, v in enumerate(input_var_names):
            next_experiments[v] = request[:, i]
        next_experiments = DataSet.from_df(pd.DataFrame(data=next_experiments))

        # Violate constraint
        mask_valid_next_experiments = self.check_constraints(next_experiments)
        if not any(mask_valid_next_experiments):
            stay_inner = True

        if stay_inner:
            # add infinity as
            next_experiments[("constraint", "DATA")] = False
        else:
            # add optimization strategy
            next_experiments[("constraint", "DATA")] = mask_valid_next_experiments
            next_experiments[("strategy", "METADATA")] = ["SNOBFIT"] * len(request)

        # Do any necessary transformation back
        next_experiments = self.transform.un_transform(
            next_experiments, categorical_method="descriptors"
        )

        return next_experiments, xbest, fbest, param

    def snobfit(self, x, f, config, dx=None, prev_param=None):
        """
        The following snobfit code was copied and modified from the SQSnobFit package and was originally published by
        Wim Lavrijsen. The SQSnobFit package includes a python version of SNOBFIT which was originally published by
        A. Neumaier.

        Copyright of SNOBFIT (v2.1):
            Neumaier, University of Vienna

            Website: https://www.mat.univie.ac.at/~neum/software/snobfit/

        Copyright of SQSnobfit (v0.4.2)
            UC Regents, Berkeley

            Website: https://pypi.org/project/SQSnobFit/
        """

        """
        request, xbest, fbest = snobfit(x, f, config, dx=None)
        minimization of a function over a box in R^n
        Input:
        file         name of file for input and output
                    if nargin < 5, the program continues a previous run and
                    reads from file.mat the output is (again) stored in file.mat
        ^^do not use file - store variables globally,
        or make them available to be passed in?
        x            the rows are a set of new points entering the
                    optimization algorithm together with their function
                    values
        f            matrix containing the corresponding function values
                    and their uncertainties, i.e., f(j,1) = f(x(j))
                    and f(j,2) = df(x(j))
                    a value f(j,2)<=0 indicates that the corresponding
                    uncertainty is not known, and the program resets it to
                    sqrt(numpy.spacing(1))
        config       structure variable defining the box [u,v] in which the
                    points are to be generated, the number nreq of
                    points to be generated and the probability p that a
                    point of type 4 is generated
                    config = struct('bounds',{u,v},'nreq',nreq,'p',p)
        dx           only used for the definition of a new problem (when
                    the program should continue from the values stored in
                    file.mat, the call should have only 4 input parameters!)
                    n-vector (n = dimension of the problem) of minimal
                    stnumpy.spacing(1), i.e., two points are considered to be different
                    if they differ by at least dx(i) in at least one
                    coordinate i
        prev_res     results of previous iterations
        Output:
        request      nreq x (n+3)-matrix
                    request[j,1:n] is the jth newly generated point,
                    request[j,n+1] is its estimated function value and
                    request[j,n+3] indicates for which reason the point
                    request[j,1:n] has been generated
                    request[j,n+3] = 1 best prediction
                                    = 2 putative local minimizer
                                    = 3 alternative good point
                                    = 4 explore empty region
                                    = 5 to fill up the required number of
                                    function values if too little points of
                                    the other classes are found
        xbest        current best point
        fbest        current best function value (i.e. function value at xbest)
        res          current results (this iteration) including results from previous iterations
        """
        from SQSnobFit._gen_utils import diag, max_, min_, find, extend, rand, sort
        from SQSnobFit._snobinput import snobinput
        from SQSnobFit._snoblocf import snoblocf, snobround
        from SQSnobFit._snoblp import snoblp
        from SQSnobFit._snobnan import snobnan
        from SQSnobFit._snobnn import snobnn
        from SQSnobFit._snobpoint import snobpoint
        from SQSnobFit._snobqfit import snobqfit
        from SQSnobFit._snobsplit import snobsplit
        from SQSnobFit._snobupdt import snobupdt
        from SQSnobFit._snob5 import snob5

        ind = find(f[:, 1] <= 0)
        if not (ind.size <= 0 or numpy.all(ind == 0)):
            f[ind, 1] = math.sqrt(numpy.spacing(1))  # may be wrong

        rho = 0.5 * (math.sqrt(5) - 1)  # golden section number
        bounds = config["bounds"]
        u1 = bounds[:, 0].reshape(1, len(bounds))  # lower
        v1 = bounds[:, 1].reshape(1, len(bounds))  # upper

        nreq = config["nreq"]
        p = config["p"]
        n = u1.shape[1]  # dimension of the problem
        nneigh = n + 5  # number of nearest neighbors

        dy = 0.1 * (v1 - u1)  # defines the vector of minimal distances between two
        # points suggested in a single call to Snobfit
        if prev_param is None:  # a new job is started
            if numpy.any(dx <= 0):
                raise ValueError("dx should contain only positive entries")

            if dx.shape[0] > 1:
                dx = dx.T

            if x.size > 0:
                u = numpy.minimum(x.min(axis=0), u1)
                v = numpy.maximum(x.max(axis=0), v1)
            else:
                u = u1.copy()
                v = v1.copy()

            x, f, np, t = snobinput(x, f)  # throw out duplicates among the points
            # and compute mean function value and
            # deviation
            if x.size > 0:
                xl, xu, x, f, nsplit, small = snobsplit(x, f, u, v, None, u, v)
                d = numpy.inf * numpy.ones((1, len(x)))
            else:
                xl = numpy.array([])
                xu = numpy.array([])
                nsplit = numpy.array([])
                small = numpy.array([])

            notnan = find(numpy.isfinite(f[:, 0]))
            if notnan.size > 0:
                fmn = min_(f[notnan, 1])
                fmx = max_(f[notnan, 1])
            else:
                fmn = 1
                fmx = 0

            if len(x) >= nneigh + 1 and fmn < fmx:
                inew = range(len(x))
                near = numpy.zeros((len(x), nneigh))
                d = numpy.zeros(len(x))
                for j in inew:
                    near[j], d[j] = snobnn(x[j], x, nneigh, dx)

                fnan = find(numpy.isnan(f[:, 0]))
                if fnan.size > 0:
                    f = snobnan(fnan, f, near, inew)

                jsize = inew[-1]
                y = numpy.zeros((jsize, 2))
                g = numpy.zeros((jsize, 2))
                sigma = numpy.zeros(jsize)
                f = extend(f, 1)
                for j in inew:
                    y[j], f[j, 2], c, sigma[j] = snoblocf(
                        j, x, f[:, 0:2], near, dx, u, v
                    )
                    g[j] = c.reshape(1, len(c))

                fbest, jbest = min_(f[:, 0])
                xbest = x[jbest]
            else:
                fnan = numpy.array([], dtype=int)
                near = numpy.array([], dtype=int)
                d = numpy.inf * numpy.ones((1, len(x)))

                x5 = snob5(x, u1, v1, dx, nreq)
                request = numpy.concatenate(
                    (x5, numpy.nan * numpy.ones((nreq, 1)), 5 * numpy.ones((nreq, 1))),
                    1,
                )
                if x.size > 0 and f.size > 0:
                    fbest, jbest = min_(f[:, 0])
                    xbest = x[jbest]
                else:
                    xbest = numpy.nan * numpy.ones((1, n))
                    fbest = numpy.inf

                if len(request) < nreq:
                    snobwarn()

                y = None
                im_storage = (
                    xbest,
                    fbest,
                    x,
                    f,
                    xl,
                    xu,
                    y,
                    nsplit,
                    small,
                    near,
                    d,
                    np,
                    t,
                    fnan,
                    u,
                    v,
                    dx,
                )
                return request, xbest, fbest, im_storage
        else:
            xnew = x.copy()
            fnew = f.copy()
            (
                xbest,
                fbest,
                x,
                f,
                xl,
                xu,
                y,
                nsplit,
                small,
                near,
                d,
                np,
                t,
                fnan,
                u,
                v,
                dx,
            ) = prev_param

            nx = len(xnew)
            oldxbest = xbest

            xl, xu, x, f, nsplit, small, near, d, np, t, inew, fnan, u, v = snobupdt(
                xl,
                xu,
                x,
                f,
                nsplit,
                small,
                near,
                d,
                np,
                t,
                xnew,
                fnew,
                fnan,
                u,
                v,
                u1,
                v1,
                dx,
            )

            if near.size > 0:
                ind = find(numpy.isnan(f[:, 0]))
                if ind.size > 0:
                    fnan = numpy.concatenate((fnan, ind.flatten()))
                if fnan.size > 0:
                    f = snobnan(fnan, f, near, inew)

                fbest, jbest = min_(f[:, 0])
                xbest = x[jbest]
                jsize = int(inew[-1] + 1)
                if y is None:
                    y = numpy.zeros((jsize, x.shape[1]))
                else:
                    y = numpy.append(
                        y, numpy.zeros((jsize - len(y), x.shape[1])), axis=0
                    )
                g = numpy.zeros((jsize, x.shape[1]))
                sigma = numpy.zeros(jsize)
                f = extend(f, x.shape[1] - 1)
                for j in inew:
                    y[j], f[j, 2], c, sigma[j] = snoblocf(
                        j, x, f[:, 0:2], near, dx, u, v
                    )
                    g[j] = c

            else:
                x5 = snob5(x, u1, v1, dx, nreq)
                request = numpy.concatenate(
                    (x5, numpy.NaN * numpy.ones((nreq, 1)), 5 * numpy.ones((nreq, 1))),
                    1,
                )
                if x.size > 0:
                    (fbest, ibest) = min_(f[:, 0])
                    xbest = x[ibest]
                else:
                    xbest = numpy.array([])
                    fbest = numpy.inf
                if request.shape[0] < nreq:
                    snobwarn()

                im_storage = (
                    xbest,
                    fbest,
                    x,
                    f,
                    xl,
                    xu,
                    y,
                    nsplit,
                    small,
                    near,
                    d,
                    np,
                    t,
                    fnan,
                    u,
                    v,
                    dx,
                )
                return request, xbest, fbest, im_storage

        sx = len(x)
        request = numpy.array([]).reshape(0, x.shape[1] + 2)
        ind = find(
            numpy.sum(
                numpy.logical_and(
                    xl <= numpy.outer(numpy.ones(sx), v1),
                    xu >= numpy.outer(numpy.ones(sx), u1),
                ),
                1,
            )
            == n
        )
        minsmall, k = min_(small[ind])
        maxsmall = small[ind].max(0)
        m1 = numpy.floor((maxsmall - minsmall) / 3)
        k = find(small[ind] == minsmall)
        k = ind[k].flatten()
        fsort, j = sort(f[k, 0])
        k = k[j]
        isplit = k[0]

        if numpy.sum(numpy.logical_and(u1 <= xbest, xbest <= v1)) == n:
            z, f1 = snobqfit(jbest, x, f[:, 0], near, dx, u1, v1)
        else:
            fbes, jbes = min_(f[ind, 0])
            jbes = ind[jbes]
            xbes = x[jbes]
            z, f1 = snobqfit(jbes, x, f[:, 0], near, dx, u1, v1)

        z = snobround(z, u1, v1, dx)
        zz = numpy.outer(numpy.ones(sx), z)
        j = find(numpy.sum(numpy.logical_and(xl <= zz, zz <= xu), 1) == n)
        if len(j) > 1:
            msmall, j1 = min_(small[j])
            j = j[j1]

        if numpy.min(
            numpy.max(
                numpy.abs(x - numpy.outer(numpy.ones(sx), z))
                - numpy.outer(numpy.ones(sx), dx)
            )
        ) >= -numpy.spacing(1):
            dmax = numpy.max((xu[j] - xl[j]) / (v - u))
            dmin = numpy.min((xu[j] - xl[j]) / (v - u))
            if dmin <= 0.05 * dmax:
                isplit = numpy.append(isplit, j)
            else:
                request = numpy.vstack(
                    (
                        request,
                        numpy.concatenate((z, numpy.array((f1, 1), ndmin=2)), axis=1),
                    )
                )

        if len(request) < nreq:
            globloc = nreq - len(request)
            glob1 = globloc * p
            glob2 = math.floor(glob1)
            if rand(1) < glob1 - glob2:
                glob = glob2 + 1
            else:
                glob = glob2

            loc = globloc - glob
            if loc:
                local, nlocal = snoblp(f[:, 0], near, ind)
                fsort, k = sort(f[local, 2])  # uhhhhhh
                j = 0
                sreq = len(request)
                while sreq < (nreq - glob) and j < len(local):
                    l0 = local[k[j]]
                    y1 = snobround(y[l0], u1, v1, dx)
                    yy = numpy.outer(numpy.ones((len(x), 1)), y1)
                    l = find(numpy.sum(numpy.logical_and(xl <= yy, yy <= xu), 1) == n)
                    if len(l) > 1:
                        msmall, j1 = min_(small[l])
                        l = l[j1]

                    dmax = numpy.max((xu[l] - xl[l]) / (v - u))
                    dmin = numpy.min((xu[l] - xl[l]) / (v - u))
                    if dmin <= 0.05 * dmax:
                        isplit = numpy.append(isplit, l)
                        j += 1
                        continue

                    if numpy.max(abs(y1 - x[l]) - dx) >= -numpy.spacing(1) and (
                        not sreq
                        or numpy.min(
                            numpy.max(
                                numpy.abs(
                                    request[:, 0:n] - numpy.outer(numpy.ones(sreq), y1)
                                )
                                - numpy.outer(numpy.ones(sreq), numpy.maximum(dy, dx)),
                                axis=1,
                            )
                        )
                        >= -numpy.spacing(1)
                    ):
                        if numpy.sum(y1 == y[l0]) < n:
                            D = f[l0, 1] / dx ** 2
                            # Possibly problem area!
                            f1 = (
                                f[l0, 0]
                                + g[l0].dot((y1 - x[l0]).T)
                                + sigma[l0]
                                * (
                                    (y1 - x[l0]).dot(
                                        diag(D).dot((y1 - x[l0]).T) + f[l0, 1]
                                    )
                                )
                            )
                        else:
                            f1 = f[l0, 2]
                        request = numpy.vstack(
                            (
                                request,
                                numpy.concatenate(
                                    (y1, numpy.array((f1, 2), ndmin=2)), axis=1
                                ),
                            )
                        )

                    sreq = len(request)
                    j += 1

                if sreq < nreq - glob:
                    fsort, k = sort(f[nlocal, 2])

                j = 0
                while sreq < (nreq - glob) and j < len(nlocal):
                    l0 = nlocal[k[j]]
                    y1 = snobround(y[l0], u1, v1, dx)
                    yy = numpy.outer(numpy.ones(len(x)), y1)
                    l = find(numpy.sum(numpy.logical_and(xl <= yy, yy <= xu), 1) == n)
                    if len(l) > 1:
                        msmall, j1 = min_(small[l])
                        l = l[j1]

                    dmax = numpy.max((xu[l] - xl[l]) / (v - u))
                    dmin = numpy.min((xu[l] - xl[l]) / (v - u))
                    if dmin <= 0.05 * dmax:
                        isplit = numpy.append(isplit, l)
                        j += 1
                        continue

                    if numpy.max(numpy.abs(y1 - x[l]) - dx) >= -numpy.spacing(1) and (
                        not sreq
                        or numpy.min(
                            numpy.max(
                                numpy.abs(
                                    request[:, :n] - numpy.outer(numpy.ones(sreq), y1)
                                )
                                - numpy.outer(numpy.ones(sreq), numpy.maximum(dy, dx)),
                                axis=1,
                            )
                        )
                        >= -numpy.spacing(1)
                    ):
                        if numpy.sum(y1 == y[l0]) < n:
                            D = f[l0, 1] / (dx ** 2)
                            f1 = (
                                f[l0, 0]
                                + g[l0].dot((y1 - x[l0]).T)
                                + sigma[l0]
                                * (
                                    ((y1 - x[l0]).dot(diag(D).dot((y1 - x[l0]).T)))
                                    + f[l0, 1]
                                )
                            )
                        else:
                            f1 = f[l0, 2]

                        request = numpy.vstack(
                            (
                                request,
                                numpy.concatenate(
                                    (y1, numpy.array((f1, 3), ndmin=2)), axis=1
                                ),
                            )
                        )

                    sreq = len(request)
                    j += 1

        sreq = len(request)
        for l in isplit.flatten():
            jj = find(ind == l)
            ind = numpy.delete(ind, jj)  # ind(jj) = []
            y1, f1 = snobpoint(
                x[l], xl[l], xu[l], f[l, 0:2], g[l], sigma[l], u1, v1, dx
            )

            if numpy.max(numpy.abs(y1 - x[l]) - dx) >= -numpy.spacing(1) and (
                not sreq
                or numpy.min(
                    numpy.max(
                        numpy.abs(request[:, :n] - numpy.outer(numpy.ones(sreq), y1))
                        - numpy.outer(numpy.ones(sreq), dx),
                        axis=1,
                    )
                )
                >= -numpy.spacing(1)
            ):
                request = numpy.vstack(
                    (
                        request,
                        numpy.concatenate((y1, numpy.array((f1, 4), ndmin=2)), axis=1),
                    )
                )

            sreq = len(request)
            if sreq == nreq:
                break

        first = True
        while (
            sreq < nreq
        ) and ind.size > 0:  # and find(small[ind] <= (minsmall + m1)).any():
            for m in range(int(m1 + 1)):
                if first:
                    first = False
                    continue

                m = 0
                k = find(small[ind] == minsmall + m)
                while k.size <= 0:
                    m += 1
                    k = find(small[ind] == minsmall + m)

                if k.size > 0:
                    k = ind[k].flatten()
                    fsort, j = sort(f[k, 0])
                    k = k[j]
                    l = int(k[0])
                    jj = find(ind == l)
                    ind = numpy.delete(ind, jj)
                    y1, f1 = snobpoint(
                        x[l], xl[l], xu[l], f[l, 0:2], g[l], sigma[l], u1, v1, dx
                    )
                    if numpy.max(numpy.abs(y1 - x[l]) - dx) >= -numpy.spacing(1) and (
                        not sreq
                        or numpy.min(
                            numpy.max(
                                numpy.abs(
                                    request[:, :n] - numpy.outer(numpy.ones(sreq), y1)
                                )
                                - numpy.outer(numpy.ones(sreq), numpy.maximum(dy, dx)),
                                axis=1,
                            )
                        )
                        >= -numpy.spacing(1)
                    ):
                        request = numpy.vstack(
                            (
                                request,
                                numpy.concatenate(
                                    (y1, numpy.array((f1, 4), ndmin=2)), axis=1
                                ),
                            )
                        )

                    sreq = len(request)
                    if sreq == nreq:
                        break
                m = 0

        if len(request) < nreq:
            x5 = snob5(
                numpy.concatenate((x, request[:, :n])), u1, v1, dx, nreq - len(request)
            )
            nx = len(x)
            for j in range(len(x5)):
                x5j = x5[j, :]
                i = find(
                    (
                        numpy.sum(xl <= numpy.outer(numpy.ones(nx), x5j))
                        and (numpy.outer(numpy.ones((nx, 1)), x5j) <= xu),
                        1,
                    )
                    == n
                )
                if len(i) > 1:
                    minv, i1 = min_(small[i])
                    i = i[i1]

                D = f[i, 1] / (dx ** 2)
                f1 = (
                    f[i, 0]
                    + (x5j - x[i]).dot(g[i].T)
                    + sigma[i] * ((x5j - x[i]).dot(diag(D).dot((x5j - x[i]).T)))
                    + f[i, 1]
                )
                request = numpy.vstack(
                    (
                        request,
                        numpy.concatenate((x5j, numpy.array((f1, 5), ndmin=2)), axis=1),
                    )
                )

        if len(request) < nreq:
            snobwarn()

        im_storage = (
            xbest,
            fbest,
            x,
            f,
            xl,
            xu,
            y,
            nsplit,
            small,
            near,
            d,
            np,
            t,
            fnan,
            u,
            v,
            dx,
        )
        return request, xbest, fbest, im_storage

    # Function to check whether a point meets the constraints of the domain
    def check_constraints(self, tmp_next_experiments):
        constr_mask = numpy.asarray([True] * len(tmp_next_experiments)).T
        if len(self.domain.constraints) > 0:
            constr = [c.constraint_type + "0" for c in self.domain.constraints]
            constr_mask = [
                pd.eval(c.lhs + constr[i], resolvers=[tmp_next_experiments])
                for i, c in enumerate(self.domain.constraints)
            ]
            constr_mask = numpy.asarray([c.tolist() for c in constr_mask]).T
            constr_mask = constr_mask.all(1)
        return constr_mask
