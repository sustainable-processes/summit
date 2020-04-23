from summit.experiment import Experiment
from summit.domain import *

class SnarBenchmark(Experiment):
    def __init__(self):
        domain = self._setup_domain()
        super().__init__(domain)

    def _run(self, conditions, **kwargs):
        pass

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Flowrate of 2,4-difluoronitrobenzene in ethanol (ml/min)"
        domain += ContinuousVariable(name='flowrate_dfnb',
                                     description=des_1,
                                     bounds=[0, 10])
        
        des_2 = "Flowrate of pyrrolidine in ethanol (mL/min)"
        domain += ContinuousVariable(name='flowrate_pldn',
                                     description=des_2,
                                     bounds=[0, 10])   

        des_3 = "Flowrate of ethanol (mL/min)"
        domain += ContinuousVariable(name='flowrate_eth',
                                     description=des_3,
                                     bounds=[0, 10])

        des_4 = "Reactor temperature (deg C)"
        domain += ContinuousVariable(name='temperature',
                                     description=des_4,
                                     bounds=[30, 120])

        # Constraints
        lhs = "flowrate_dfnb+flowrate_pldn+flowrate_eth-10"
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