
import pytest

from summit.domain import Domain, ContinuousVariable, Constraint
from summit.strategies import *
from summit.utils.dataset import DataSet
from summit.benchmarks import test_functions

import numpy as np
import pandas as pd

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
    results_arr = results.data_to_numpy().astype(np.float32)
    assert np.isclose(results_arr.all(), arr.all())
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
    results_arr = results.data_to_numpy().astype(np.float32)
    assert np.isclose(results_arr.all(), arr.all())
    return results

def test_multitosingleobjective_transform():
    class MockStrategy(Strategy):
        def suggest_experiments(self, num_experiments, previous_results):
            inputs, outputs = self.transform.transform_inputs_outputs(previous_results)
            objectives = [v for v in self.domain.variables if v.is_objective]
            assert len(objectives) == 1
            assert objectives[0].name == 'scalar_objective'
            assert outputs['scalar_objective'].iloc[0] == 70.0
            return self.transform.un_transform(inputs)

    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='yield_', description='', bounds=[0,100], is_objective=True, maximize=True)
    domain += ContinuousVariable(name='de', description='diastereomeric excess', bounds=[0,100], is_objective=True, maximize=True)
    columns = [v.name for v in domain.variables]
    values  =   {('temperature', 'DATA'): 60, 
                ('flowrate_a', 'DATA'): 0.5,  
                ('flowrate_b', 'DATA'): 0.5,
                ('yield_', 'DATA'): 50, 
                ('de', 'DATA'): 90,
                }
    previous_results = DataSet([values], columns=columns)
    transform = MultitoSingleObjective(domain, expression='(yield_+de)/2', maximize=True)
    strategy = MockStrategy(domain, transform=transform)
    strategy.suggest_experiments(5, previous_results)

def test_logspaceobjectives_transform():
    class MockStrategy(Strategy):
        def suggest_experiments(self, num_experiments, previous_results):
            inputs, outputs = self.transform.transform_inputs_outputs(previous_results)
            objectives = [v for v in self.domain.variables if v.is_objective]
            assert len(objectives) == 2
            assert np.isclose(outputs['log_yield_'].iloc[0], np.log(50))
            assert np.isclose(outputs['log_de'].iloc[0], np.log(90))
            return self.transform.un_transform(inputs)

    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[50, 100])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0.1, 0.5])
    domain += ContinuousVariable(name='yield_', description='', bounds=[0,100], is_objective=True, maximize=True)
    domain += ContinuousVariable(name='de', description='diastereomeric excess', bounds=[0,100], is_objective=True, maximize=True)
    columns = [v.name for v in domain.variables]
    values  =   {('temperature', 'DATA'): [60, 100],
                ('flowrate_a', 'DATA'): [0.5, 0.4],  
                ('flowrate_b', 'DATA'): [0.5, 0.4],
                ('yield_', 'DATA'): [50, 60], 
                ('de', 'DATA'): [90, 80],
                }
    previous_results = DataSet(values, columns=columns)
    transform = LogSpaceObjectives(domain)
    strategy = MockStrategy(domain, transform=transform)
    strategy.suggest_experiments(5, previous_results)
    

def test_tsemo():
    pass

'''
@pytest.mark.parametrize('num_experiments', [1, 2, 4])
@pytest.mark.parametrize('maximize', [True, False])
def test_snobfit(num_experiments, maximize):
    # Single-objective optimization problem with 3 dimensional input domain (only continuous inputs)
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='yield', description='relative conversion to xyz',
                                 bounds=[-1000,1000], is_objective=True, maximize=maximize)
    domain += Constraint(lhs="temperature+flowrate_a+flowrate_b-1", constraint_type="<=") #TODO: implement decoding of constraints
    constraint = False
    strategy = SNOBFIT(domain, probability_p=0.5, dx_dim=1E-5)

    # Simulating experiments with hypothetical relationship of inputs and outputs,
    # here Hartmann 3D function: https://www.sfu.ca/~ssurjano/hart3.html
    # Note that SNOBFIT treats constraints implicitly, i.e., for variable sets that
    # violate one of the constraints return NaN as function value (so-called: hidden constraints)
    def sim_fun(x_exp):
        if constr(x_exp):
            x_exp = x_exp[:3]
            A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
            P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*10**(-4)
            alpha = np.array([1,1.2,3.0,3.2])
            d = np.zeros((4,1))
            for k in range(4):
                d[k] = np.sum(np.dot(A[k,:],(x_exp-P[k,:])**2))
            y_exp = np.sum(np.dot(alpha,np.exp(-d)))
            if not maximize:
                y_exp *= -1.0
        else:
            y_exp = np.nan
        return y_exp
    def test_fun(x):
        y = np.array([sim_fun(x[i]) for i in range(0, x.shape[0])])
        return y
    # Define hypothetical constraint (for real experiments, check constraint and return NaN)
    def constr(x):
        if constraint:
            return (x[0]+x[1]+x[2]<=1)
        else:
            return True

    # Initialize with "experimental" data
    initial_exp = pd.DataFrame(data={'temperature': [0.409,0.112,0.17,0.8], 'flowrate_a': [0.424,0.33,0.252,0.1],
                                     'flowrate_b': [0.13,0.3,0.255,0.01]})   # initial experimental points
    initial_exp.insert(3,'yield', test_fun(initial_exp.to_numpy()))   # initial results
    initial_exp = DataSet.from_df(initial_exp)

    # run SNOBFIT loop for fixed <num_iter> number of iteration with <num_experiments> number of experiments each
    # stop loop if <max_stop> consecutive iterations have not produced an improvement
    # num_experiments = 4
    num_iter = 400//num_experiments
    max_stop = 50//num_experiments
    nstop = 0
    fbestold = float("inf")
    for i in range(num_iter):
        # initial run without history
        if i == 0:
            next_experiments, xbest, fbest, res = strategy.suggest_experiments(num_experiments, prev_res=initial_exp)
        # runs with history
        else:
            next_experiments_mod = next_experiments.data_to_numpy()
            if not constraint:
                # This is the part where experiments take place
                exp_yield = test_fun(next_experiments.data_to_numpy())
                next_experiments['yield', 'DATA'] = exp_yield
                # Call of SNOBFIT
                next_experiments, xbest, fbest, res = \
                    strategy.suggest_experiments(num_experiments, prev_res=next_experiments, prev_param=res)
            else:
                constr_mask = []
                for exp in next_experiments_mod:
                    # check whether constraint is satisfied (True) or violated (False)
                    if constr(exp):
                        constr_mask.append(True)
                    else:
                        constr_mask.append(False)

                next_experiments['yield', 'DATA'] = np.nan   # set results for experiments with violated constraints to NaN
                next_experiments['constraint', 'DATA'] = constr_mask
                # This is the part where the experiments take place
                exp_yield = test_fun(next_experiments.iloc[constr_mask].data_to_numpy())
                next_experiments.loc[next_experiments['constraint'] != False, 'yield'] = exp_yield   # set results for valid experiments
                # Call of SNOBFIT
                next_experiments, xbest, fbest, res = \
                    strategy.suggest_experiments(num_experiments, prev_res= \
                    next_experiments.loc[:,~next_experiments.columns.isin([('constraint','DATA')])],prev_param=res)

        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break
        print(next_experiments)   # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    if not constraint:
        # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
        assert (xbest[0] >= 0.11 and xbest[0] <= 0.12) and (xbest[1] >= 0.55 and xbest[1] <= 0.56) and \
               (xbest[2] >= 0.85 and xbest[2] <= 0.86) and (fbest <= -3.85 and fbest >= -3.87)
    else:
        # Extrema of test function with constraint: tbd /TODO: determine optimum with constraint with other algorithms
        assert fbest <= -1
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
'''

@pytest.mark.parametrize('x_start', [[0,0],[4,6],[1,2],[-2,5]])
@pytest.mark.parametrize('maximize', [True, False])
@pytest.mark.parametrize('constraint', [True, False])
def test_nm2D(x_start,maximize,constraint, plot=False):

    himmelblau = test_functions.Himmelblau(maximize=maximize, constraints=constraint)
    strategy = NelderMead(himmelblau.domain, x_start=x_start, adaptive=False)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    #initial_exp = pd.DataFrame(data={'x_1': [4.0,4.0,2.0], 'x_2': [2.0,3.0,-6.0]})   # initial experimental points
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = himmelblau.run_experiments(initial_exp)  # initial results

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 100   # maximum number of iterations
    max_stop = 20   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    polygons_points = []

    #Initial experiments
    if initial_exp is not None:
        polygons_points.append(np.asarray(
            [(initial_exp.data_to_numpy()[i][:2].tolist(), initial_exp.data_to_numpy()[j][:2])
                for i in range(len(initial_exp.data_to_numpy())) for j in
                range(len(initial_exp.data_to_numpy()))]))
        next_experiments=initial_exp
    else:
        next_experiments = None

    param=None
    for i in range(num_iter):
        next_experiments, xbest, fbest, param = \
            strategy.suggest_experiments(prev_res=next_experiments, prev_param=param)\

        # This is the part where experiments take place
        next_experiments = himmelblau.run_experiments(next_experiments)

        # save polygon points for plotting
        polygons_points.append(np.asarray([param[0][0][i].tolist() for i in range(len(param[0][0]))]))

        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break

        print(next_experiments)   # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    assert fbest <= 0.1
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraints: four identical local minima f = 0 at x1 = (3.000, 2.000),
    # x2 = (-2.810, 3.131), x3 = (-3.779, -3.283), x4 = (3.584, -1.848)

    # plot
    if plot:
        himmelblau.plot(polygons=polygons_points)


@pytest.mark.parametrize('x_start, maximize, constraint',
                         [([0,0,0], True, True), ([0,0,0], True, False),
                          ([0,0,0], False, True), ([0,0,0], False, False),
                          ([1,1,0.2], True, False), ([1,1,0.2], False, False),
                          ([], True, True), ([], True, False), ([], False, True), ([], False, False),
                          ([0.4,0.2,0.6], True, True), ([0.4,0.2,0.6], True, False),
                          ([0.4,0.2,0.6], False, True), ([0.4,0.2,0.6], False, False)])
def test_nm3D(maximize,x_start,constraint, plot=False):

    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraint)
    strategy = NelderMead(hartmann3D.domain,x_start=x_start)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    #initial_exp = pd.DataFrame(data={'x_1': [0.1,0.1,0.4,0.3], 'x_2': [0.6,0.2,0.4,0.5], 'x_3': [1,1,1,0.3]})   # initial experimental points
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = hartmann3D.run_experiments(initial_exp)

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 200   # maximum number of iterations
    max_stop = 20   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    polygons_points = []

    #Initial experiments
    if initial_exp is not None:
        polygons_points.append(np.asarray(
            [(initial_exp.data_to_numpy()[i][:3].tolist(), initial_exp.data_to_numpy()[j][:3])
                for i in range(len(initial_exp.data_to_numpy())) for j in
                range(len(initial_exp.data_to_numpy()))]))
        next_experiments=initial_exp
    else:
        next_experiments = None

    param=None
    for i in range(num_iter):
        next_experiments, xbest, fbest, param = \
                    strategy.suggest_experiments(prev_res=next_experiments, prev_param=param)

        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        polygons_points.append(np.asarray([(param[0][0][i].tolist(),param[0][0][j].tolist())
                        for i in range(len(param[0][0])) for j in range(len(param[0][0]))]))

        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break
        print(next_experiments)   # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)

    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
        #assert (xbest[0] >= 0.113 and xbest[0] <= 0.115) and (xbest[1] >= 0.555 and xbest[1] <= 0.557) and \
        #       (xbest[2] >= 0.851 and xbest[2] <= 0.853) and (fbest <= -3.85 and fbest >= -3.87)
    if plot:
        hartmann3D.plot(polygons=polygons_points)
