
import pytest
from summit.domain import Domain, ContinuousVariable
from summit.strategies import Random, LHS
import numpy as np

def test_random():
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    strategy = Random(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)
    arr = np.array(([[77.53989513,  0.45851724,  0.11195048],
                    [85.40739113,  0.15023412,  0.28273329],
                    [64.54523695,  0.18289715,  0.35965762],
                    [75.54138026,  0.12058688,  0.21139491],
                    [94.64734772,  0.27632394,  0.37050196]]))
    assert np.isclose(results.data_to_numpy().all(), arr.all())
    return results

def test_lhs():
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    strategy = LHS(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)
    arr = np.array([[95.  ,  0.46,  0.38],
                    [65.  ,  0.14,  0.14],
                    [55.  ,  0.22,  0.3 ],
                    [85.  ,  0.3 ,  0.46],
                    [75.  ,  0.38,  0.22]])
    assert np.isclose(results.data_to_numpy().all(), arr.all())
    return results

def test_tsemo():
    pass

def test_snobfit():
    from summit.domain import Domain, ContinuousVariable
    from summit.strategies import SNOBFIT
    from summit.utils.dataset import DataSet
    import pandas as pd

    # Single-objective optimization problem with 3 dimensional input domain (only continuous inputs)
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 50])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[-1000,1000], is_objective=True, maximize=True)

    # Simulating experiments with hypothetical relationship of inputs and outputs
    def sim_fun(x):
        y = (-1/30 * x[0] - 3 * x[1] ** 2 + 4 * x[0] * x[2])
        return y
    def test_fun(x):
        y = np.array([sim_fun(x[i]) for i in range(0, x.shape[0])])
        return y

    # Initialize with "experimental" data
    initial_exp = pd.DataFrame(data={'temperature': [10,4,5,3], 'flowrate_a': [0.6,0.3,0.2,0.1],
                                     'flowrate_b': [0.1,0.3,0.2,0.1]})
    initial_exp.insert(3,'yield', test_fun(initial_exp.to_numpy()))
    initial_exp = DataSet.from_df(initial_exp)

    strategy = SNOBFIT(domain)

    # run snobfit loop for fixed number of iteration whereas with num_experiments each
    num_experiments = 5
    num_iter = 10
    for i in range(num_iter):
        if i == 0:
            next_experiments, xbest, fbest, res = strategy.suggest_experiments(num_experiments, prev_res=initial_exp)
        else:
            yields = test_fun(next_experiments.data_to_numpy())
            next_experiments['yield', 'DATA'] = yields
            next_experiments, xbest, fbest, res = strategy.suggest_experiments(num_experiments,
                                                                               prev_res=next_experiments,prev_param=res)
            print(next_experiments)
            print("\n")

    # Extrema of test function: glob_max = 595/3 at (50,0,1), glob_min = -14/3 at (50,1,0), loc_min = -3 at (0,1,1)
    assert xbest[0] == 50 and xbest[1] == 1 and xbest[2] == 0   and fbest == -14/3
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
