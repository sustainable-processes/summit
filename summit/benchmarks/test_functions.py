from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np

from math import sin, cos, pi
from operator import mul
from functools import reduce
import numpy as np


class Himmelblau(Experiment):
    """Himmelblau function (2D) for testing optimization algorithms

    Virtual experiment corresponds to a function evaluation.

    Examples
    --------
    >>> b = Himmelblau()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    Notes
    -----
    This function is taken from http://benchmarkfcns.xyz/benchmarkfcns/himmelblaufcn.html.

    """

    def __init__(self, constraints=False, maximize=False, **kwargs):
        self.constraints = constraints
        self.maximize = maximize

        if self.maximize:
            self.equation = "-((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)"
        else:
            self.equation = "((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)"

        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Input 1"
        domain += ContinuousVariable(name="x_1", description=des_1, bounds=[-4, 4])

        des_2 = "Input 2"
        domain += ContinuousVariable(name="x_2", description=des_2, bounds=[-6, 6])

        # Objectives
        des_3 = "Function value"
        domain += ContinuousVariable(
            name="y",
            description=des_3,
            bounds=[-1000, 0],
            is_objective=True,
            maximize=self.maximize,
        )

        if self.constraints:
            domain += Constraint(lhs="x_1+x_2+7", constraint_type=">=")
            domain += Constraint(lhs="x_1*x_2+10", constraint_type=">=")

        return domain

    def _run(self, conditions, **kwargs):
        x_1 = float(conditions["x_1"])
        x_2 = float(conditions["x_2"])
        y = eval(self.equation)
        conditions[("y", "DATA")] = y

        # save evaluated points for plotting
        self.evaluated_points.append([x_1, x_2])

        return conditions, None

    def plot(self, ax=None, **kwargs):
        """Make a plot of the experiments evaluated thus far

        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to

        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis

        if ax is a matplotlib axis, returns only the axis

        Raises
        ------
        ValueError
            If there are no points to plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        # evaluated points in <run_experiments> and optional points added by the user
        extra_points = kwargs.get("extra_points", None)

        # polygons objects to be plotted
        polygons = kwargs.get("polygons", None)

        points = self.evaluated_points
        if extra_points is not None:
            points.append(extra_points)
        if extra_points is None:
            if points is None:
                raise ValueError("No points to plot.")
        else:
            if points is None:
                points = extra_points
            else:
                points.append(extra_points)

        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False

        # get domain bounds and plot frame/axes
        bounds_x_1 = self.domain.__getitem__("x_1").bounds
        bounds_x_2 = self.domain.__getitem__("x_2").bounds
        expand_bounds = 1
        plt.axis(
            [
                bounds_x_1[0] - expand_bounds,
                bounds_x_1[1] + expand_bounds,
                bounds_x_2[0] - expand_bounds,
                bounds_x_2[1] + expand_bounds,
            ]
        )

        ax.axvline(x=bounds_x_1[0], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[0], color="k", linestyle="--")
        ax.axvline(x=bounds_x_1[1], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[1], color="k", linestyle="--")

        # plot contour
        xlist = np.linspace(
            bounds_x_1[0] - expand_bounds, bounds_x_1[1] + expand_bounds, 1000
        )
        ylist = np.linspace(
            bounds_x_2[0] - expand_bounds, bounds_x_2[1] + expand_bounds, 1000
        )
        x_1, x_2 = np.meshgrid(xlist, ylist)
        if self.maximize:
            z = eval("-" + self.equation)
        else:
            z = eval(self.equation)
        ax.contour(x_1, x_2, z, levels=np.logspace(-2, 3, 30, base=10), alpha=0.3)

        # plot evaluated and extra points with enumeration
        for c in range(len(points)):
            tmp_x_1, tmp_x_2 = points[c][0], points[c][1]
            ax.scatter(tmp_x_1, tmp_x_2)
            ax.text(tmp_x_1 + 0.01, tmp_x_2 + 0.01, c + 1, fontsize=7)

        # plot constraints
        if len(self.domain.constraints) > 0:
            x = np.linspace(bounds_x_1[0], bounds_x_1[1], 400)
            y = np.linspace(bounds_x_2[0], bounds_x_2[1], 400)
            x_1, x_2 = np.meshgrid(x, y)
            for c in self.domain.constraints:
                z = eval(c.lhs)
                ax.contour(x_1, x_2, z, [0], colors="grey", linestyles="dashed")

        # plot polygons
        if polygons:
            patches = []
            for i in range(len(polygons)):
                polygon_obj = Polygon(polygons[i], True, hatch="x")
                patches.append(polygon_obj)

            p = PatchCollection(patches, facecolors="None", edgecolors="grey", alpha=1)
            ax.add_collection(p)

        if return_fig:
            return fig, ax
        else:
            return ax

    def reset(self):
        super().reset()
        self.evaluated_points = []


class Hartmann3D(Experiment):
    """Hartmann test function (3D) for testing optimization algorithms

    Virtual experiment corresponds to a function evaluation.

    Examples
    --------
    >>> b = Hartmann3D()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    Notes
    -----
    This function is taken from https://www.sfu.ca/~ssurjano/hart3.html.

    """

    def __init__(self, constraints=False, maximize=False, **kwargs):
        self.constraints = constraints
        self.evaluated_points = []

        if maximize:
            self.maximize = True
        else:
            self.maximize = False

        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Input 1"
        domain += ContinuousVariable(name="x_1", description=des_1, bounds=[0, 1])

        des_2 = "Input 2"
        domain += ContinuousVariable(name="x_2", description=des_2, bounds=[0, 1])

        des_3 = "Input 3"
        domain += ContinuousVariable(name="x_3", description=des_3, bounds=[0, 1])

        # Objectives
        des_4 = "Function value"
        domain += ContinuousVariable(
            name="y",
            description=des_4,
            bounds=[-3.9, 0],
            is_objective=True,
            maximize=self.maximize,
        )
        if self.constraints:
            domain += Constraint(lhs="x_1+x_2+x_3-1.625", constraint_type="<=")

        return domain

    def _run(self, conditions, **kwargs):
        def function_evaluation(x_1, x_2, x_3):
            x_exp = np.asarray([x_1, x_2, x_3])
            A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
            P = (
                np.array(
                    [
                        [3689, 1170, 2673],
                        [4699, 4387, 7470],
                        [1091, 8732, 5547],
                        [381, 5743, 8828],
                    ]
                )
                * 10 ** (-4)
            )
            alpha = np.array([1, 1.2, 3.0, 3.2])
            d = np.zeros((4, 1))
            for k in range(4):
                d[k] = np.sum(np.dot(A[k, :], (x_exp - P[k, :]) ** 2))
            y = np.sum(np.dot(alpha, np.exp(-d))).T
            if not self.maximize:
                y = -y
            return y

        x_1 = float(conditions["x_1"])
        x_2 = float(conditions["x_2"])
        x_3 = float(conditions["x_3"])

        y = function_evaluation(x_1, x_2, x_3)
        conditions[("y", "DATA")] = y

        # save evaluated points for plotting
        self.evaluated_points.append([x_1, x_2, x_3, y])

        return conditions, None

    def plot(self, ax=None, **kwargs):
        """Make a plot of the experiments evaluated thus far

        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to

        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis

        if ax is a matplotlib axis, returns only the axis

        Raises
        ------
        ValueError
            If there are no points to plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # evaluated points in <run_experiments> and optional points added by the user
        extra_points = kwargs.get("extra_points", None)

        # polygons objects to be plotted
        polygons = kwargs.get("polygons", None)

        points = self.evaluated_points
        if extra_points is None:
            if points is None:
                raise ValueError("No points to plot.")
        else:
            if points is None:
                points = extra_points
            else:
                points.append(extra_points)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            return_fig = True
        else:
            return_fig = False

        # get domain bounds and plot frame/axes
        bounds_x_1 = self.domain.__getitem__("x_1").bounds
        bounds_x_2 = self.domain.__getitem__("x_2").bounds
        bounds_x_3 = self.domain.__getitem__("x_3").bounds

        expand_bounds = 0
        ax.axes.set_xlim3d(
            left=bounds_x_1[0] - expand_bounds, right=bounds_x_1[1] + expand_bounds
        )
        ax.axes.set_ylim3d(
            bottom=bounds_x_2[0] - expand_bounds, top=bounds_x_2[1] + expand_bounds
        )
        ax.axes.set_zlim3d(
            bottom=bounds_x_3[0] - expand_bounds, top=bounds_x_3[1] + expand_bounds
        )

        # plot evaluated and extra points with enumeration
        for c in range(len(points)):
            tmp_x_1, tmp_x_2, tmp_x_3, col = (
                points[c][0],
                points[c][1],
                points[c][2],
                points[c][3],
            )
            if self.maximize:
                col = -col
            p = ax.scatter(
                tmp_x_1, tmp_x_2, tmp_x_3, c=col, cmap="seismic", vmin=-4, vmax=0
            )
            ax.text(tmp_x_1 + 0.05, tmp_x_2 + 0.05, tmp_x_3 + 0.05, c + 1, fontsize=7)
        fig.colorbar(p, shrink=0.5)

        # plot constraints
        if len(self.domain.constraints) > 0:
            x = np.linspace(bounds_x_1[0], bounds_x_1[1], 100)
            y = np.linspace(bounds_x_2[0], bounds_x_2[1], 100)
            x_1, x_2 = np.meshgrid(x, y)
            # workaround to plot constraint plane
            x_3 = 0
            for c in self.domain.constraints:
                z = -eval(c.lhs)
                z[z < bounds_x_3[0]] = np.nan
                z[z > bounds_x_3[1]] = np.nan
                ax.plot_surface(
                    x_1,
                    x_2,
                    z,
                    vmin=bounds_x_3[0],
                    vmax=bounds_x_3[1],
                    rstride=4,
                    cstride=4,
                    alpha=0.8,
                )

        # plot polygons
        if polygons:
            for i in range(len(polygons)):
                polygon = Poly3DCollection(polygons[i], alpha=0.1)
                polygon.set_edgecolor("b")
                ax.add_collection3d(polygon)

        if return_fig:
            return fig, ax
        else:
            return ax


class ThreeHumpCamel(Experiment):
    """Three-Hump Camel function (2D) for testing optimization algorithms

    Virtual experiment corresponds to a function evaluation.

    Examples
    --------
    >>> b = ThreeHumpCamel()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    Notes
    -----
    This function is taken from https://www.sfu.ca/~ssurjano/camel3.html.

    """

    def __init__(self, constraints=False, maximize=False, **kwargs):
        self.constraints = constraints
        self.evaluated_points = []
        self.maximize = maximize

        if self.maximize:
            self.equation = "-(2*x_1**2 - 1.05*x_1**4 + (x_1**6)/6 + x_1*x_2 + x_2**2)"
        else:
            self.equation = "2*x_1**2 - 1.05*x_1**4 + (x_1**6)/6 + x_1*x_2 + x_2**2"

        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Input 1"
        domain += ContinuousVariable(name="x_1", description=des_1, bounds=[-2, 2])

        des_2 = "Input 2"
        domain += ContinuousVariable(name="x_2", description=des_2, bounds=[-2, 2])

        # Objectives
        des_3 = "Function value"
        domain += ContinuousVariable(
            name="y",
            description=des_3,
            bounds=[-1000, 1000],
            is_objective=True,
            maximize=self.maximize,
        )

        if self.constraints:
            domain += Constraint(lhs="x_1+x_2+7", constraint_type=">=")
            domain += Constraint(lhs="x_1*x_2+10", constraint_type=">=")

        return domain

    def _run(self, conditions, **kwargs):
        x_1 = float(conditions["x_1"])
        x_2 = float(conditions["x_2"])
        y = eval(self.equation)
        conditions[("y", "DATA")] = y

        # save evaluated points for plotting
        self.evaluated_points.append([x_1, x_2])

        return conditions, None

    def plot(self, ax=None, **kwargs):
        """Make a plot of the experiments evaluated thus far

        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to

        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis

        if ax is a matplotlib axis, returns only the axis

        Raises
        ------
        ValueError
            If there are no points to plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as Colors
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        # evaluated points in <run_experiments> and optional points added by the user
        extra_points = kwargs.get("extra_points", None)

        # polygons objects to be plotted
        polygons = kwargs.get("polygons", None)

        points = self.evaluated_points
        if extra_points is not None:
            points.append(extra_points)
        if extra_points is None:
            if points is None:
                raise ValueError("No points to plot.")
        else:
            if points is None:
                points = extra_points
            else:
                points.append(extra_points)

        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False

        # get domain bounds and plot frame/axes
        bounds_x_1 = self.domain.__getitem__("x_1").bounds
        bounds_x_2 = self.domain.__getitem__("x_2").bounds
        expand_bounds = 1
        plt.axis(
            [
                bounds_x_1[0] - expand_bounds,
                bounds_x_1[1] + expand_bounds,
                bounds_x_2[0] - expand_bounds,
                bounds_x_2[1] + expand_bounds,
            ]
        )

        ax.axvline(x=bounds_x_1[0], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[0], color="k", linestyle="--")
        ax.axvline(x=bounds_x_1[1], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[1], color="k", linestyle="--")

        # plot contour
        xlist = np.linspace(
            bounds_x_1[0] - expand_bounds, bounds_x_1[1] + expand_bounds, 1000
        )
        ylist = np.linspace(
            bounds_x_2[0] - expand_bounds, bounds_x_2[1] + expand_bounds, 1000
        )
        x_1, x_2 = np.meshgrid(xlist, ylist)
        if self.maximize:
            z = eval("-" + self.equation)
        else:
            z = eval(self.equation)
        ax.contour(
            x_1,
            x_2,
            z,
            levels=np.logspace(-2, 1, 20, base=10),
            cmap="Spectral",
            norm=Colors.LogNorm(),
            alpha=0.6,
        )

        # plot evaluated and extra points with enumeration
        for c in range(len(points)):
            tmp_x_1, tmp_x_2 = points[c][0], points[c][1]
            ax.scatter(tmp_x_1, tmp_x_2)
            ax.text(tmp_x_1 + 0.01, tmp_x_2 + 0.01, c + 1, fontsize=7)

        # plot constraints
        if len(self.domain.constraints) > 0:
            x = np.linspace(bounds_x_1[0], bounds_x_1[1], 400)
            y = np.linspace(bounds_x_2[0], bounds_x_2[1], 400)
            x_1, x_2 = np.meshgrid(x, y)
            for c in self.domain.constraints:
                z = eval(c.lhs)
                ax.contour(x_1, x_2, z, [0], colors="grey", linestyles="dashed")

        # plot polygons
        if polygons:
            patches = []
            for i in range(len(polygons)):
                polygon_obj = Polygon(polygons[i], True, hatch="x")
                patches.append(polygon_obj)

            p = PatchCollection(patches, facecolors="None", edgecolors="grey", alpha=1)
            ax.add_collection(p)

        if return_fig:
            return fig, ax
        else:
            return ax


class DTLZ2(Experiment):
    def __init__(self, num_inputs=6, num_objectives=2, **kwargs):
        if num_objectives >= num_inputs:
            raise ValueError(
                "Number of inputs must be greater than number of objectives."
            )
        self.nobjs = num_objectives
        self.nvars = num_inputs
        domain = self._setup_domain(self.nobjs, self.nvars)
        super().__init__(domain)

    def _setup_domain(self, nobjs, nvars):
        variables = [
            ContinuousVariable(f"x_{i}", f"Decision variable {i}", bounds=[0, 1.0])
            for i in range(nvars)
        ]
        variables += [
            ContinuousVariable(
                f"y_{i}",
                f"Objective {i}",
                bounds=[0, 1.0],
                is_objective=True,
                maximize=False,
            )
            for i in range(nobjs)
        ]
        return Domain(variables)

    def _run(self, conditions, **kwargs):
        # Convert from dataframe
        x = conditions[[f"x_{i}" for i in range(self.nvars)]]
        x = x.to_numpy().astype(np.float64)
        x = np.atleast_2d(x)

        nobjs = self.nobjs
        """Copied from DEAP"""
        """https://github.com/DEAP/deap/blob/master/deap/benchmarks/__init__.py"""
        for individual in x:
            xc = individual[: nobjs - 1]
            xm = individual[nobjs - 1 :]
            g = sum((xi - 0.5) ** 2 for xi in xm)
            f = [(1.0 + g) * reduce(mul, (cos(0.5 * xi * pi) for xi in xc), 1.0)]
            f.extend(
                (1.0 + g)
                * reduce(mul, (cos(0.5 * xi * pi) for xi in xc[:m]), 1)
                * sin(0.5 * xc[m] * pi)
                for m in range(nobjs - 2, -1, -1)
            )

        # Convert to dataset
        for i in range(nobjs):
            conditions[(f"y_{i}", "DATA")] = f[i]
        return conditions, {}


class VLMOP2(Experiment):
    def __init__(self, maximize: bool = False, **kwargs):
        domain = self._setup_domain(2, 2, maximize)
        self.nvars = 2
        self.nobjs = 2
        self.maximize= maximize
        super().__init__(domain)

    def _setup_domain(self, nobjs, nvars, maximize):
        variables = [
            ContinuousVariable(f"x_{i}", f"Decision variable {i}", bounds=[-2, 2])
            for i in range(nvars)
        ]
        variables += [
            ContinuousVariable(
                f"y_{i}",
                f"Objective {i}",
                bounds=[-2, 2],
                is_objective=True,
                maximize=maximize,
            )
            for i in range(nobjs)
        ]
        return Domain(variables)

    def _run(self, conditions, **kwargs):
        # Convert from dataframe
        x = conditions[[f"x_{i}" for i in range(self.nvars)]]
        x = x.to_numpy().astype(np.float64)
        x = np.atleast_2d(x)

        transl = 1 / np.sqrt(2)
        part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
        part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
        y1 = 1 - np.exp(-1 * part1)
        y2 = 1 - np.exp(-1 * part2)
        f = np.hstack((y1, y2))

        if self.maximize:
            f *= -1.0

        # Convert to dataset
        for i in range(self.nobjs):
            conditions[(f"y_{i}", "DATA")] = f[:, i][0]
        return conditions, {}
