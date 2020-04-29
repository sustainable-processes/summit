from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
import numpy as np
from scipy.integrate import solve_ivp

class SnarBenchmark(Experiment):
    ''' SnAr Benchmark

    Virtual experiments representing a nucleophilic aromatic
    substitution happening in a plug flow reactor. 
    
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
    This benchmark is based on Hone et al. Reac Engr. & Chem. 2016.
    
    ''' 
    def __init__(self):
        domain = self._setup_domain()
        super().__init__(domain)

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "residence time in minutes"
        domain += ContinuousVariable(name='tau',
                                     description=des_1,
                                     bounds=[0.5, 2])
        
        des_2 = "equivalents of pyrrolidine"
        domain += ContinuousVariable(name='equiv_pldn',
                                     description=des_2,
                                     bounds=[1.5, 3.5])   

        des_3 = "concentration of 2,4 dinitrofluorobenenze at reactor inlet (after mixing) in M"
        domain += ContinuousVariable(name='conc_dfnb',
                                     description=des_3,
                                     bounds=[0.1, 0.28])

        des_4 = "Reactor temperature in degress celsius"
        domain += ContinuousVariable(name='temperature',
                                     description=des_4,
                                     bounds=[30, 120])

        # Objectives
        des_5 = 'space time yield (kg/m^3/h)'
        domain += ContinuousVariable(name='sty',
                                     description=des_5,
                                     bounds=[0, 100],
                                     is_objective=True,
                                     maximize=True)

        des_6 = "E-factor"
        domain += ContinuousVariable(name='e_factor',
                                     description=des_6,
                                     bounds=[0, 1e6],
                                     is_objective=True,
                                     maximize=False)

        return domain  

    def _run(self, conditions, **kwargs):
        tau = float(conditions['tau'])
        equiv_pldn = float(conditions['equiv_pldn'])
        conc_dfnb = float(conditions['conc_dfnb'])
        T = float(conditions['temperature'])
        y, e_factor, res = self._integrate_equations(tau, equiv_pldn, conc_dfnb,T)   
        conditions[('sty', 'DATA')] = y
        conditions[('e_factor', 'DATA')] = e_factor
        return conditions, {'integration_result': res}

    def _integrate_equations(self, 
                             tau,
                             equiv_pldn,
                             conc_dfnb,
                             temperature):        
        # Initial Concentrations in mM
        self.C_i = np.zeros(5)
        self.C_i[0] = conc_dfnb
        self.C_i[1] = equiv_pldn*conc_dfnb

        # Flowrate and residence time
        V = 3 #mL
        q_tot = V/tau
        C1_0 = 2.0  # reservoir concentration of 1 is 1 M = 1 mM
        C2_0 = 4.2  # reservoi concentration of  2 is 2 M = 2 mM
        q_1 = self.C_i[0]/C1_0*q_tot  # flowrate of 1 (dfnb)
        q_2 = self.C_i[1]/C2_0*q_tot  # flowrate of 2 (pldn)
        q_eth = q_tot-q_1-q_2    # flowrate of ethanol

        # Integrate
        res = solve_ivp(self._integrand,[0, tau], self.C_i,
                        args=(temperature,))
        C_final = res.y[:, -1]

        # Calculate STY and E-factor
        M = [159.09, 71.12, 210.21, 210.21, 261.33] # molecular weights (g/mol)
        sty = 6e4/1000*M[2]*C_final[2]*q_tot/V   # convert to kg m^-3 h^-1
        rho_eth = 0.789 # g/mL (should adjust to temp, but just using @ 25C)
        term_2 = 1e-3*sum([M[i]*C_final[i]*q_tot for i in range(5)
                           if i!=2])
        if np.isclose(C_final[2], 0.0):
            #Set to a large value if no product formed
            e_factor=1e9
        else:
            e_factor = term_2/(1e-3*M[2]*C_final[2]*q_tot)
        if e_factor > 1e9:
            e_factor = 1e9
        return sty, e_factor, res
        
    def _integrand(self,t, C, T):
        # Kinetic Constants
        R = 8.314/1000  #kJ/K/mol
        T_ref = 90 + 273.71  #Convert to deg K
        T = T + 273.71       #Convert to deg K
        #Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
        k = lambda k_ref, E_a, temp: 6e3*k_ref*np.exp(-E_a/R*(1/temp-1/T_ref))
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)
        
        # Reaction Rates
        r = np.zeros(5)
        for i in [0, 1]: #Set to reactants when close
            C[i] = 0 if C[i]< 1e-6*self.C_i[i] else C[i]
        r[0] = -(k_a+k_b)*C[0]*C[1]
        r[1] = -(k_a+k_b)*C[0]*C[1]-k_c*C[1]*C[2]-k_d*C[1]*C[3]
        r[2] = k_a*C[0]*C[1]-k_c*C[1]*C[2]
        r[3] = k_a*C[0]*C[1]-k_d*C[1]*C[3]
        r[4] = k_c*C[1]*C[2]+k_d*C[1]*C[3]

        # Deltas
        dcdtau = r
        return dcdtau





 