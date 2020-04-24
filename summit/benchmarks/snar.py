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
    >>> values = [v.bounds[0]+ 0.1*(v.bounds[1]-v.bounds[0])
              for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiment(conditions)
    
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
        des_1 = "Flowrate of 2,4-difluoronitrobenzene in ethanol (ml/min)"
        domain += ContinuousVariable(name='q_dfnb',
                                     description=des_1,
                                     bounds=[0, 10])
        
        des_2 = "Flowrate of pyrrolidine in ethanol (mL/min)"
        domain += ContinuousVariable(name='q_pldn',
                                     description=des_2,
                                     bounds=[0, 10])   

        des_3 = "Flowrate of ethanol (mL/min)"
        domain += ContinuousVariable(name='q_eth',
                                     description=des_3,
                                     bounds=[0, 10])

        des_4 = "Reactor temperature (deg C)"
        domain += ContinuousVariable(name='temperature',
                                     description=des_4,
                                     bounds=[30, 120])

        # Constraints
        lhs = "q_dfnb+q_pldn+q_eth-10"
        domain += Constraint(lhs, "<=")

        # Objectives
        des_5 = 'space time yield'
        domain += ContinuousVariable(name='sty',
                                     description=des_5,
                                     bounds=[0, 100],
                                     is_objective=True,
                                     maximize=True)

        des_6 = "E-factor"
        domain += ContinuousVariable(name='e_factor',
                                     description=des_6,
                                     bounds=[0, 1000],
                                     is_objective=True,
                                     maximize=False)

        return domain  

    def _run(self, conditions, **kwargs):
        T = float(conditions['temperature'])
        q_dfnb = float(conditions['q_dfnb'])
        q_pldn = float(conditions['q_pldn'])
        q_eth = float(conditions['q_eth'])
        q_tot = q_dfnb+q_pldn+q_eth
        if q_tot > 10.0:
            raise ValueError(f"Total flowrate must be less than 10.0 mL/min, currently is {q_tot} mL/min.")
        y, e_factor, res = self._integrate_equations(q_dfnb, q_pldn, q_eth,T)   
        conditions['sty'] = y
        conditions['e_factor'] = e_factor
        return conditions, {'integration_result': res}

    def _integrate_equations(self, 
                             q_dfnb,
                             q_pldn,
                             q_eth,
                             temperature):
        # Flowrate and residence time
        V = 5 #mL
        q_tot = q_dfnb+q_pldn+q_eth
        tau = V/q_tot

        # Initial Concentrations
        C_i = np.zeros(5)
        C_10 = 1
        C_20 = 2
        C_i[0] = C_10*q_dfnb/q_tot
        C_i[1] = C_10*q_pldn/q_tot

        # Integrate
        res = solve_ivp(self._integrand,[0, tau], C_i,
                        args=(temperature,))
        C_final = res.y[:, -1]

        # Calculate STY and E-factor
        sty = C_final[2]/tau
        rho_eth = 0.789 # g/mL (should adjust to temp, but just using @ 25C)
        M = [159.09, 71.12, 210.21, 210.21, 261.33] # molecular weights (g/mol)
        term_2 = sum([M[i]*C_final[i]*q_tot for i in range(5)
                     if i!=2])
        e_factor = (q_eth*rho_eth+term_2)/(M[2]*C_final[2]*q_tot)

        return sty, e_factor, res
        
    def _integrand(self,t, C, T):
        # Kinetic Constants
        R = 8.71
        T_ref = 90 + 273.71
        k = lambda k_ref, E_a, T: k_ref*np.exp(-E_a/R*(1/T-1/T_ref))
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.864, 38.9, T)
        k_d = k(1.63, 44.8, T)
        
        # Reaction Rates
        r = np.zeros(5)
        r[0] = -(k_a+k_b)*C[0]*C[1]
        r[1] = -(k_a+k_b)*C[0]*C[1]-k_c*C[1]*C[2]-k_d*C[1]*C[3]
        r[2] = k_a*C[0]*C[1]-k_c*C[1]*C[2]
        r[3] = k_a*C[0]*C[1]-k_d*C[1]*C[3]
        r[4] = k_c*C[1]*C[2]-k_d*C[1]*C[3]

        # Deltas
        dcdtau = r
        return dcdtau



 