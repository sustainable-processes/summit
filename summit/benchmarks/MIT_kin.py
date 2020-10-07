from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
from scipy.integrate import solve_ivp


class MIT_case1(Experiment):
    """Benchmark representing a nucleophilic aromatic substitution (SnAr) reaction

    The SnAr reactions occurs in a plug flow reactor where residence time, stoichiometry and temperature
    can be adjusted. Maximizing Space time yield (STY) and minimising E-factor are the objectives.

    Parameters
    ----------

    noise_level: float, optional
        The mean of the random noise added to the concentration measurements in terms of
        percent of the signal. Default is 0.


    Examples
    --------

    >>> b = SnarBenchmark()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    Notes
    -----

    This benchmark relies on the kinetics observerd by [Hone]_ et al. The mechanistic
    model is integrated using scipy to find outlet concentrations of all species. These
    concentrations are then used to calculate STY and E-factor.

    References
    ----------

    .. [Hone] C. A. Hone et al., React. Chem. Eng., 2017, 2, 103–108. DOI:
       `10.1039/C6RE00109B <https://doi.org/10.1039/C6RE00109B>`_

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
        domain += ContinuousVariable(name="conc_cat", description=des_1, bounds=[?])

        des_2 = "reaction time"
        domain += ContinuousVariable(
            name="t", description=des_2, bounds=[?]
        )

        des_3 = "Choice of catalyst"
        domain += ContinuousVariable(
            name="cat", description=des_3, bounds=[?]
        )

        des_4 = "Reactor temperature in degress celsius"
        domain += ContinuousVariable(
            name="temperature", description=des_4, bounds=[?]
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
        cat = float(conditions["cat"])
        T = float(conditions["temperature"])
        y = self._integrate_equations(conc_cat, t, cat, T)
        conditions[("y", "DATA")] = y
        return conditions, {}

    def _integrate_equations(self, conc_cat, t, cat, T):
        # Initial Concentrations in mMß
        self.C_i = np.zeros(5)
        self.C_i[0] = conc_cat
        self.C_i[1] = equiv_pldn * conc_dfnb

        # Flowrate and residence time
        V = 3  # mL
        q_tot = V / tau
        C1_0 = 2.0  # reservoir concentration of 1 is 1 M = 1 mM
        C2_0 = 4.2  # reservoi concentration of  2 is 2 M = 2 mM
        q_1 = self.C_i[0] / C1_0 * q_tot  # flowrate of 1 (dfnb)
        q_2 = self.C_i[1] / C2_0 * q_tot  # flowrate of 2 (pldn)
        q_eth = q_tot - q_1 - q_2  # flowrate of ethanol

        # Integrate
        res = solve_ivp(self._integrand, [0, tau], self.C_i, args=(temperature,))
        C_final = res.y[:, -1]

        # Add measurment noise
        C_final += (
            C_final * self.rng.normal(scale=self.noise_level, size=len(C_final)) / 100
        )
        C_final[
            C_final < 0
        ] = 0  # prevent negative values of concentration introduced by noise

        # Calculate STY and E-factor
        M = [159.09, 71.12, 210.21, 210.21, 261.33]  # molecular weights (g/mol)
        sty = 6e4 / 1000 * M[2] * C_final[2] * q_tot / V  # convert to kg m^-3 h^-1
        if sty < 1e-6:
            sty = 1e-6
        rho_eth = 0.789  # g/mL (should adjust to temp, but just using @ 25C)
        term_2 = 1e-3 * sum([M[i] * C_final[i] * q_tot for i in range(5) if i != 2])
        if np.isclose(C_final[2], 0.0):
            # Set to a large value if no product formed
            e_factor = 1e3
        else:
            e_factor = (q_tot * rho_eth + term_2) / (1e-3 * M[2] * C_final[2] * q_tot)
        if e_factor > 1e3:
            e_factor = 1e3

        return sty, e_factor, {}

    def _integrand(self, t, C, T):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 90 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
        k = (
            lambda k_ref, E_a, temp: 0.6
            * k_ref
            * np.exp(-E_a / R * (1 / temp - 1 / T_ref))
        )
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)

        # Reaction Rates
        r = np.zeros(5)
        for i in [0, 1]:  # Set to reactants when close
            C[i] = 0 if C[i] < 1e-6 * self.C_i[i] else C[i]
        r[0] = -(k_a + k_b) * C[0] * C[1]
        r[1] = -(k_a + k_b) * C[0] * C[1] - k_c * C[1] * C[2] - k_d * C[1] * C[3]
        r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
        r[3] = k_a * C[0] * C[1] - k_d * C[1] * C[3]
        r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]

        # Deltas
        dcdtau = r
        return dcdtau

    def to_dict(self, **kwargs):
        experiment_params = dict(noise_level=self.noise_level)
        return super().to_dict(**experiment_params)