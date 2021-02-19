from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
from scipy.integrate import solve_ivp


class MIT_case2(Experiment):
    """Benchmark representing a simulated kinetic reaction network and accompanying kinetic constants (see reference).

    The reactions occur in a batch reactor.
    The objective is to maximize yield (y), defined as the concentration of product dividen by the initial concentration of
    the limiting reagent (We can do this because the stoichiometry is 1:1).

    We optimize the reactions by changing the catalyst concentration, reaction time, choice of catalyst, and temperature.

    Parameters
    ----------

    noise_level: float, optional
        The mean of the random noise added to the concentration measurements in terms of
        percent of the signal. Default is 0.


    Examples
    --------

    Notes
    -----

    This benchmark relies on the kinetics simulated by Jensen et al. The mechanistic
    model is integrated using scipy to find outlet concentrations of all species.


    References
    ----------

    K. Jensen et al., React. Chem. Eng., 2018, 3,301
    DOI: 10.1039/c8re00032h
    """

    def __init__(self, noise_level=0, **kwargs):
        domain = self._setup_domain()
        super().__init__(domain)
        self.rng = np.random.default_rng()
        self.noise_level = noise_level

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "catalyst concentration"
        domain += ContinuousVariable(
            name="conc_cat",
            description=des_1,
            bounds=[0.835 * 10 ** (-3), 4.175 * 10 ** (-3)],
        )

        des_2 = "reaction time"
        domain += ContinuousVariable(name="t", description=des_2, bounds=[60, 600])

        des_3 = "Choice of catalyst"
        domain += CategoricalVariable(
            name="cat_index", description=des_3, levels=[0, 1, 2, 3, 4, 5, 6, 7]
        )

        des_4 = "Reactor temperature in degress celsius"
        domain += ContinuousVariable(
            name="temperature", description=des_4, bounds=[30, 110]
        )

        # Objectives
        des_5 = "yield (%)"
        domain += ContinuousVariable(
            name="y",
            description=des_5,
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )

        return domain

    def _run(self, conditions, **kwargs):
        conc_cat = float(conditions["conc_cat"])
        t = float(conditions["t"])
        cat_index = int(conditions["cat_index"])
        T = float(conditions["temperature"])
        y, res = self._integrate_equations(conc_cat, t, cat_index, T)
        conditions[("y", "DATA")] = y
        return conditions, {}

    def _integrate_equations(self, conc_cat, t, cat_index, T):
        # Initial Concentrations in mM
        self.C_i = np.zeros(6)
        self.C_i[0] = 0.167  # Initial conc of A
        self.C_i[1] = 0.250  # Initial conc of B
        self.C_i[2] = conc_cat  # Initial conc of cat

        # Integrate
        res = solve_ivp(self._integrand, [0, t], self.C_i, args=(cat_index, T))
        C_final = res.y[:, -1]

        # Add measurment noise
        C_final += (
            C_final * self.rng.normal(scale=self.noise_level, size=len(C_final)) / 100
        )
        C_final[
            C_final < 0
        ] = 0  # prevent negative values of concentration introduced by noise

        # calculate yield
        # M = [159.09, 71.12, 210.21, 210.21, 261.33]  # molecular weights (g/mol)
        y = C_final[3] / self.C_i[0]
        return y, res

    def _integrand(self, t, C, cat_index, T):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 90 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        conc_cat = C[2]
        A_R = 3.1 * 10 ** 7
        A_S1 = 1 * 10 ** 12
        A_S2 = 3.1 * 10 ** 5

        E_Ai = [0, 0, 0.3, 0.7, 0.7, 2.2, 3.8, 7.3]
        # cat_index = 1
        E_AR = 55

        k = (
            lambda conc_cat, A, E_A, E_Ai, temp: np.sqrt(conc_cat)
            * A
            * np.exp(-(E_A + E_Ai) / (R * temp))
        )
        k_R = k(conc_cat, A_R, E_AR, E_Ai[cat_index], T)
        k_S1 = 0  # k(conc_cat, A_S1, 100, 0 , T)
        k_S2 = 0  # k(conc_cat, A_S2, 50, 0,  T)

        # Reaction Rates
        r = np.zeros(6)
        # for i in [0, 1]:  # Set to reactants when close
        #    C[i] = 0 if C[i] < 1e-6 * self.C_i[i] else C[i]
        r[0] = -k_R * C[0] * C[1]
        r[1] = -k_R * C[0] * C[1] - k_S1 * C[1] - k_S2 * C[1] * C[3]
        r[2] = 0
        r[3] = k_R * C[0] * C[1] - k_S2 * C[1] * C[3]
        r[4] = k_S1 * C[1]
        r[5] = k_S2 * C[1] * C[3]
        # C[0]: A
        # C[1]: B
        # C[2]: Cat
        # C[3]: R
        # C[4]: S1
        # C[5]: S2

        # Reactions
        # A+B -> R
        # B -> S1
        # B+R -> S2

        # Deltas
        dcdt = r
        return dcdt

    def to_dict(self, **kwargs):
        experiment_params = dict(noise_level=self.noise_level)
        return super().to_dict(**experiment_params)
