
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

@pytest.mark.parametrize('num_experiments', [1, 2, 4])
@pytest.mark.parametrize('maximize', [False, True])
@pytest.mark.parametrize('constraints', [False, True])
def test_snobfit(num_experiments, maximize, constraints):

    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraints)
    strategy = SNOBFIT(hartmann3D.domain, probability_p=0.5, dx_dim=1E-5)


    initial_exp = None
    # Comment out to start without initial data
    initial_exp = pd.DataFrame(data={'x_1': [0.409,0.112,0.17,0.8], 'x_2': [0.424,0.33,0.252,0.1],
                                     'x_3': [0.13,0.3,0.255,0.01]})   # initial experimental points
    initial_exp = DataSet.from_df(initial_exp)
    initial_exp = hartmann3D.run_experiments(initial_exp)   # initial results

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
            # This is the part where experiments take place
            next_experiments = hartmann3D.run_experiments(next_experiments)

            # Call of SNOBFIT
            next_experiments, xbest, fbest, res = \
                strategy.suggest_experiments(num_experiments, prev_res=next_experiments, prev_param=res)


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
    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
    assert (xbest[0] >= 0.11 and xbest[0] <= 0.12) and (xbest[1] >= 0.55 and xbest[1] <= 0.56) and \
               (xbest[2] >= 0.85 and xbest[2] <= 0.86) and (fbest <= -3.85 and fbest >= -3.87)

    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    hartmann3D.plot()

@pytest.mark.parametrize('x_start', [[0,0],[4,6],[-3,-4],[1,2],[-2,5]])
@pytest.mark.parametrize('maximize', [True, False])
@pytest.mark.parametrize('constraint', [True, False])
def test_nm2D(x_start,maximize,constraint):

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
    for i in range(num_iter):
        # initial run without history
        if i == 0:
            try:
                if initial_exp is not None:
                    x = np.asarray([initial_exp.data_to_numpy()[i][:2] for i in range(len(initial_exp))])
                    polygons_points.append(x)
                    next_experiments, xbest, fbest, param = strategy.suggest_experiments(prev_res=initial_exp)
                else:
                    next_experiments, xbest, fbest, param = strategy.suggest_experiments()

            # TODO: how to handle internal errors? Here implemented as ValueError - maybe introduce a InternalError class for strategies
            except ValueError as e:
                print(e)
                break

        # runs with history
        else:
            # This is the part where experiments take place
            next_experiments = himmelblau.run_experiments(next_experiments)

            # Call Nelder-Mead Simplex
            try:
                next_experiments, xbest, fbest, param = \
                    strategy.suggest_experiments(prev_res=next_experiments, prev_param=param)

            # TODO: how to handle internal stopping criteria? Here implemented as ValueError - maybe introduce a StoppingError class for strategies
            except (NotImplementedError, ValueError) as e:
                print(e)
                break

        # save polygon points for plotting
        x = np.asarray([param[0][0][i].tolist() for i in range(len(param[0][0]))])
        polygons_points.append(x)

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
    himmelblau.plot(polygons=polygons_points)

@pytest.mark.parametrize('x_start', [[0,0,0],[1,1,0.2],[],[0.4,0.2,0.6]])
@pytest.mark.parametrize('maximize', [True, False])
@pytest.mark.parametrize('constraint', [True, False])
def test_nm3D(maximize,x_start,constraint):

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
    for i in range(num_iter):
        # initial run without history
        if i == 0:
            try:
                if initial_exp is not None:
                    polygons_points.append(np.asarray(
                        [(initial_exp.data_to_numpy()[i][:3].tolist(), initial_exp.data_to_numpy()[j][:3])
                         for i in range(len(initial_exp.data_to_numpy())) for j in
                         range(len(initial_exp.data_to_numpy()))]))

                    next_experiments, xbest, fbest, param = strategy.suggest_experiments(prev_res=initial_exp)

                else:
                    next_experiments, xbest, fbest, param = strategy.suggest_experiments()

            # TODO: how to handle internal errors? Here implemented as ValueError - maybe introduce a InternalError class for strategies
            except ValueError as e:
                print(e)
                return


        # runs with history
        else:
            # This is the part where experiments take place
            next_experiments = hartmann3D.run_experiments(next_experiments)

            try:
                next_experiments, xbest, fbest, param = \
                    strategy.suggest_experiments(prev_res=next_experiments, prev_param=param)

            # TODO: how to handle internal stopping criteria? Here implemented as ValueError - maybe introduce a StoppingError class for strategies
            except (ValueError, NotImplementedError) as e:
                print(e)
                break

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

    hartmann3D.plot(polygons=polygons_points)
