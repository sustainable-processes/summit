import pytest

from summit.domain import *
from summit.strategies import *
from summit.utils.dataset import DataSet
from summit.benchmarks import test_functions

from fastprogress.fastprogress import progress_bar
import numpy as np
import pandas as pd
import os

def test_random():
    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[50, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flow of reactant a in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5]
    )

    strategy = Random(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)
    arr = np.array(
        (
            [
                [77.53989513, 0.45851724, 0.11195048],
                [85.40739113, 0.15023412, 0.28273329],
                [64.54523695, 0.18289715, 0.35965762],
                [75.54138026, 0.12058688, 0.21139491],
                [94.64734772, 0.27632394, 0.37050196],
            ]
        )
    )
    results_arr = results.data_to_numpy().astype(np.float32)
    assert np.isclose(results_arr.all(), arr.all())

    
    solvent_ds = DataSet(
        [[5, 81], [-93, 111]],
        index=["benzene", "toluene"],
        columns=["melting_point", "boiling_point"],
    )
    domain += CategoricalVariable("solvent", "solvent descriptors", descriptors=solvent_ds)
    strategy = Random(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)
    return results

def test_lhs():
    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[50, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flow of reactant a in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5]
    )
    strategy = LHS(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)
    arr = np.array(
        [
            [95.0, 0.46, 0.38],
            [65.0, 0.14, 0.14],
            [55.0, 0.22, 0.3],
            [85.0, 0.3, 0.46],
            [75.0, 0.38, 0.22],
        ]
    )
    results_arr = results.data_to_numpy().astype(np.float32)
    assert np.isclose(results_arr.all(), arr.all())

    solvent_ds = DataSet(
        [[5, 81], [-93, 111]],
        index=["benzene", "toluene"],
        columns=["melting_point", "boiling_point"],
    )
    domain += CategoricalVariable("solvent", "solvent descriptors", descriptors=solvent_ds)
    strategy = LHS(domain, random_state=np.random.RandomState(3))
    results = strategy.suggest_experiments(5)

    return results

def test_doe():
    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[50, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flow of reactant a in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="yield_", description="", bounds=[0, 100], is_objective=True, maximize=True
    )
    domain += ContinuousVariable(
        name="de",
        description="diastereomeric excess",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )

    strategy = FullFactorial(domain)
    levels = dict(temperature=[50,100], flowrate_a=[0.1,0.5],
                  flowrate_b=[0.1,0.5])
    experiments = strategy.suggest_experiments(levels)
    return experiments


def test_multitosingleobjective_transform():
    class MockStrategy(Strategy):
        def suggest_experiments(self, num_experiments, previous_results):
            inputs, outputs = self.transform.transform_inputs_outputs(previous_results)
            objectives = [v for v in self.domain.variables if v.is_objective]
            assert len(objectives) == 1
            assert objectives[0].name == "scalar_objective"
            assert outputs["scalar_objective"].iloc[0] == 70.0
            return self.transform.un_transform(inputs)

    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[50, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flow of reactant a in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="yield_", description="", bounds=[0, 100], is_objective=True, maximize=True
    )
    domain += ContinuousVariable(
        name="de",
        description="diastereomeric excess",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    columns = [v.name for v in domain.variables]
    values = {
        ("temperature", "DATA"): 60,
        ("flowrate_a", "DATA"): 0.5,
        ("flowrate_b", "DATA"): 0.5,
        ("yield_", "DATA"): 50,
        ("de", "DATA"): 90,
    }
    previous_results = DataSet([values], columns=columns)
    transform = MultitoSingleObjective(
        domain, expression="(yield_+de)/2", maximize=True
    )
    strategy = MockStrategy(domain, transform=transform)
    strategy.suggest_experiments(5, previous_results)


def test_logspaceobjectives_transform():
    class MockStrategy(Strategy):
        def suggest_experiments(self, num_experiments, previous_results):
            inputs, outputs = self.transform.transform_inputs_outputs(previous_results)
            objectives = [v for v in self.domain.variables if v.is_objective]
            assert len(objectives) == 2
            assert np.isclose(outputs["log_yield_"].iloc[0], np.log(50))
            assert np.isclose(outputs["log_de"].iloc[0], np.log(90))
            return self.transform.un_transform(inputs)

    domain = Domain()
    domain += ContinuousVariable(
        name="temperature",
        description="reaction temperature in celsius",
        bounds=[50, 100],
    )
    domain += ContinuousVariable(
        name="flowrate_a", description="flow of reactant a in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="flowrate_b", description="flow of reactant b in mL/min", bounds=[0.1, 0.5]
    )
    domain += ContinuousVariable(
        name="yield_", description="", bounds=[0, 100], is_objective=True, maximize=True
    )
    domain += ContinuousVariable(
        name="de",
        description="diastereomeric excess",
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )
    columns = [v.name for v in domain.variables]
    values = {
        ("temperature", "DATA"): [60, 100],
        ("flowrate_a", "DATA"): [0.5, 0.4],
        ("flowrate_b", "DATA"): [0.5, 0.4],
        ("yield_", "DATA"): [50, 60],
        ("de", "DATA"): [90, 80],
    }
    previous_results = DataSet(values, columns=columns)
    transform = LogSpaceObjectives(domain)
    strategy = MockStrategy(domain, transform=transform)
    strategy.suggest_experiments(5, previous_results)


def test_tsemo():
    pass


@pytest.mark.parametrize("num_experiments", [1, 2, 4])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("constraints", [True, False])
def test_snobfit(num_experiments, maximize, constraints):

    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraints)
    strategy = SNOBFIT(hartmann3D.domain, probability_p=0.5, dx_dim=1e-5)

    initial_exp = None
    # Comment out to start without initial data
    # initial_exp = pd.DataFrame(data={'x_1': [0.409,0.112,0.17,0.8], 'x_2': [0.424,0.33,0.252,0.1],
    #                                 'x_3': [0.13,0.3,0.255,0.01]})   # initial experimental points
    # initial_exp = DataSet.from_df(initial_exp)
    # initial_exp = hartmann3D.run_experiments(initial_exp)   # initial results

    # run SNOBFIT loop for fixed <num_iter> number of iteration with <num_experiments> number of experiments each
    # stop loop if <max_stop> consecutive iterations have not produced an improvement
    num_iter = 400 // num_experiments
    max_stop = 50 // num_experiments
    nstop = 0
    fbestold = float("inf")

    # Initial experiments
    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        # Call of SNOBFIT
        next_experiments = strategy.suggest_experiments(
            num_experiments, prev_res=next_experiments
        )

        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)

    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
    assert fbest <= -3.85 and fbest >= -3.87

    # Test saving and loading
    strategy.save('snobfit_test.json')
    strategy_2 = SNOBFIT.load('snobfit_test.json')
    
    for a,b in zip(strategy.prev_param[0], strategy_2.prev_param[0]):
        if type(a) == list:
            assert all(a) == all(b)
        elif type(a) == np.ndarray:
            assert a.all() == b.all()
        elif np.isnan(a):
            assert np.isnan(b)
        else:
            assert a == b 
    assert all(strategy.prev_param[1][0]) == all(strategy_2.prev_param[1][0])
    os.remove('snobfit_test.json')

    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))


@pytest.mark.parametrize("x_start", [[0, 0], [4, 6], [1, 2], [-2, 5]])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("constraint", [True, False])
def test_nm2D(x_start, maximize, constraint, plot=False):

    himmelblau = test_functions.Himmelblau(maximize=maximize, constraints=constraint)
    strategy = NelderMead(himmelblau.domain, x_start=x_start, adaptive=False)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    # initial_exp = pd.DataFrame(data={'x_1': [4.0,4.0,2.0], 'x_2': [2.0,3.0,-6.0]})
    # initial_exp = DataSet.from_df(initial_exp)
    # initial_exp = himmelblau.run_experiments(initial_exp)  # initial results

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 100  # maximum number of iterations
    max_stop = 20  # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    polygons_points = []

    # Initial experiments
    if initial_exp is not None:
        polygons_points.append(
            np.asarray(
                [
                    (
                        initial_exp.data_to_numpy()[i][:2].tolist(),
                        initial_exp.data_to_numpy()[j][:2],
                    )
                    for i in range(len(initial_exp.data_to_numpy()))
                    for j in range(len(initial_exp.data_to_numpy()))
                ]
            )
        )
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        next_experiments = strategy.suggest_experiments(
            prev_res=next_experiments
        )
        # This is the part where experiments take place
        next_experiments = himmelblau.run_experiments(next_experiments)

        # save polygon points for plotting
        param = strategy.prev_param
        polygons_points.append(
            np.asarray([param[0]['sim'][i].tolist() for i in range(len(param[0]['sim']))])
        )

        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    assert fbest <= 0.1
    # print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraints: four identical local minima f = 0 at x1 = (3.000, 2.000),
    # x2 = (-2.810, 3.131), x3 = (-3.779, -3.283), x4 = (3.584, -1.848)

    # Test saving and loading
    strategy.save('nm_2d.json')
    strategy_2 = NelderMead.load('nm_2d.json')

    assert strategy._x_start == strategy_2._x_start
    assert strategy._dx == strategy_2._dx
    assert strategy._df == strategy_2._df
    assert strategy._adaptive == strategy_2._adaptive
    p = strategy.prev_param[0]
    p2 = strategy.prev_param[0]
    for k,v in p.items():
        if type(v) not in [list, np.ndarray]:
            assert v == p2[k]
        elif type(v) == list:
            for i,l in enumerate(v):
                if type(l) in [np.ndarray, DataSet]:
                    assert l.all() == p2[k][i].all()
                else:
                    assert l == p2[k][i]
    assert all(strategy.prev_param[1]) == all(strategy_2.prev_param[1])
    os.remove('nm_2d.json')

    # plot
    if plot:
        fig, ax = himmelblau.plot(polygons=polygons_points)


@pytest.mark.parametrize(
    "x_start, maximize, constraint",
    [
        ([0, 0, 0], True, True),
        ([0, 0, 0], True, False),
        ([0, 0, 0], False, True),
        ([0, 0, 0], False, False),
        ([1, 1, 0.2], True, False),
        ([1, 1, 0.2], False, False),
        ([], True, True),
        ([], True, False),
        ([], False, True),
        ([], False, False),
        ([0.4, 0.2, 0.6], True, True),
        ([0.4, 0.2, 0.6], True, False),
        ([0.4, 0.2, 0.6], False, True),
        ([0.4, 0.2, 0.6], False, False),
    ],
)
def test_nm3D(maximize, x_start, constraint, plot=False):

    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraint)
    strategy = NelderMead(hartmann3D.domain, x_start=x_start)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    # initial_exp = pd.DataFrame(data={'x_1': [0.1,0.1,0.4,0.3], 'x_2': [0.6,0.2,0.4,0.5], 'x_3': [1,1,1,0.3]})
    # initial_exp = DataSet.from_df(initial_exp)
    # initial_exp = hartmann3D.run_experiments(initial_exp)

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 200  # maximum number of iterations
    max_stop = 20  # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    polygons_points = []

    # Initial experiments
    if initial_exp is not None:
        polygons_points.append(
            np.asarray(
                [
                    (
                        initial_exp.data_to_numpy()[i][:3].tolist(),
                        initial_exp.data_to_numpy()[j][:3],
                    )
                    for i in range(len(initial_exp.data_to_numpy()))
                    for j in range(len(initial_exp.data_to_numpy()))
                ]
            )
        )
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        next_experiments = strategy.suggest_experiments(
            prev_res=next_experiments
        )
        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        # save polygon points for plotting
        param = strategy.prev_param
        polygons_points.append(
            np.asarray([param[0]['sim'][i].tolist() for i in range(len(param[0]['sim']))])
        )
        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
    # assert (xbest[0] >= 0.113 and xbest[0] <= 0.115) and (xbest[1] >= 0.555 and xbest[1] <= 0.557) and \
    #        (xbest[2] >= 0.851 and xbest[2] <= 0.853) and (fbest <= -3.85 and fbest >= -3.87)

    # Test saving and loading
    strategy.save('nm_3d.json')
    strategy_2 = NelderMead.load('nm_3d.json')

    assert strategy._x_start == strategy_2._x_start
    assert strategy._dx == strategy_2._dx
    assert strategy._df == strategy_2._df
    assert strategy._adaptive == strategy_2._adaptive
    p = strategy.prev_param[0]
    p2 = strategy.prev_param[0]
    for k,v in p.items():
        if type(v) not in [list, np.ndarray]:
            assert v == p2[k]
        elif type(v) == list:
            for i,l in enumerate(v):
                if type(l) in [np.ndarray, DataSet]:
                    assert l.all() == p2[k][i].all()
                else:
                    assert l == p2[k][i]
    assert all(strategy.prev_param[1]) == all(strategy_2.prev_param[1])
    os.remove('nm_3d.json')

    if plot:
        fig, ax = hartmann3D.plot(polygons=polygons_points)



@pytest.mark.parametrize(
    "batch_size, max_num_exp, maximize, constraint, check_convergence",
    [
        [1, 1, True, True, False],
        [1, 200, True, True, True],
        [2, 200, True, True, True],
        [4, 200, True, True, True],
        [1, 200, False, True, True],
        [2, 200, False, True, True],
        [4, 200, False, True, True],
        [1, 200, True, False, True],
        [2, 200, True, False, True],
        [4, 200, True, False, True],     
    ]
)
def test_sobo(batch_size, max_num_exp, maximize, constraint,check_convergence, plot=False):
    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraint)
    strategy = SOBO(domain=hartmann3D.domain)

    # Uncomment to start algorithm with pre-defined initial experiments
    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    #initial_exp = pd.DataFrame(data={'x_1': [0.1,0.1,0.4,0.3], 'x_2': [0.6,0.2,0.4,0.5], 'x_3': [1,1,1,0.3]})   # initial experimental points
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = hartmann3D.run_experiments(initial_exp)

    # run SOBO loop for fixed <num_iter> number of iteration
    num_iter = max_num_exp//batch_size   # maximum number of iterations
    max_stop = 80//batch_size   # allowed number of consecutive iterations w/o improvement
    min_stop = 20//batch_size   # minimum number of iterations before algorithm is stopped
    nstop = 0
    fbestold = float("inf")

    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    pb = progress_bar(range(num_iter))
    for i in pb:
        next_experiments = \
            strategy.suggest_experiments(num_experiments=batch_size, prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            if fbest < 0.99*fbestold or i < min_stop:
                nstop = 0
            else:
                nstop += 1
            fbestold = fbest
        else:
            nstop += 1
        if nstop >= max_stop:
            print("Stopping criterion reached. No improvement in last " + str(max_stop) + " iterations.")
            break
        if fbest < -3.85:
            print("Stopping criterion reached. Function value below -3.85.")
            break

        pb.comment = f"Best f value: {fbest}"
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
    if check_convergence:
        assert (fbest <= -3.85 and fbest >= -3.87)

    # Test saving and loading
    strategy.save('sobo_test.json')
    strategy_2 = SOBO.load('sobo_test.json')
    os.remove('sobo_test.json')

    if strategy.prev_param is not None: 
        assert strategy.prev_param[0].all() == strategy_2.prev_param[0].all()
        assert strategy.prev_param[1].all() == strategy_2.prev_param[1].all()

    if plot:
        fig, ax = hartmann3D.plot()

@pytest.mark.parametrize(
    "batch_size, max_num_exp, maximize, constraint, check_convergence, test_id",
    [
        [1, 1, True, False, False, 0],
        [1, 200, True, False, True, 1],
#        [2, 200, True, False, True, 2],
#        [4, 200, True, False, True, 3],
#        [1, 200, False, False, True, 4],
        [2, 200, False, False, True, 5],
#        [4, 200, False, False, True, 6]
    ]
)
def test_gryffin_himmelblau(batch_size, max_num_exp, maximize, constraint, check_convergence, test_id, plot=False):
    himmelblau = test_functions.Himmelblau(maximize=maximize, constraints=constraint)
    strategy = GRYFFIN(domain=himmelblau.domain, sampling_strategies=batch_size)

    # Uncomment to start algorithm with pre-defined initial experiments
    initial_exp = None

    # run Gryffin loop for fixed <num_iter> number of iteration
    num_iter = max_num_exp // batch_size  # maximum number of iterations
    max_stop = 60 // batch_size  # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")

    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    pb = progress_bar(range(num_iter))
    for i in pb:
        next_experiments = \
            strategy.suggest_experiments(prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = himmelblau.run_experiments(next_experiments)

        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break
        if fbest < 2:
            print("Stopping criterion reached. Function value below 0.5.")
            break

        pb.comment = f"Best f value: {fbest}"
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraints: four identical local minima f = 0 at x1 = (3.000, 2.000),
    # x2 = (-2.810, 3.131), x3 = (-3.779, -3.283), x4 = (3.584, -1.848)
    if check_convergence:
        assert (fbest <= 2)

    # Test saving and loading
    save_name = 'gryffin_test_himmelblau_' + str(test_id) + '.json'
    strategy.save(save_name)
    strategy_2 = GRYFFIN.load(save_name)
    #os.remove(save_name)

    if strategy.prev_param is not None:
        assert strategy.prev_param == strategy_2.prev_param

    if plot:
        fig, ax = himmelblau.plot()


@pytest.mark.parametrize(
    "batch_size, max_num_exp, maximize, constraint, check_convergence, test_id",
    [
        [1, 1, True, False, False, 0],
        [1, 200, True, False, True, 1],
#        [2, 200, True, False, True, 2],
#        [4, 200, True, False, True, 3],
#        [1, 200, False, False, True, 4],
        [2, 200, False, False, True, 5],
#        [4, 200, False, False, True, 6]
    ]
)
def test_gryffin_hartmann(batch_size, max_num_exp, maximize, constraint,check_convergence, test_id, plot=False, ):
    hartmann3D = test_functions.Hartmann3D(maximize=maximize, constraints=constraint)
    strategy = GRYFFIN(domain=hartmann3D.domain, sampling_strategies=batch_size)

    # Uncomment to start algorithm with pre-defined initial experiments
    initial_exp = None

    # run GRYFFIN loop for fixed <num_iter> number of iteration
    num_iter = max_num_exp//batch_size   # maximum number of iterations
    max_stop = 60//batch_size   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")

    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    pb = progress_bar(range(num_iter))
    for i in pb:
        next_experiments = \
            strategy.suggest_experiments(prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        fbest = strategy.fbest * -1.0 if maximize else strategy.fbest
        xbest = strategy.xbest
        if fbest < fbestold:
            fbestold = fbest
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break
        if fbest < -3.5:
            print("Stopping criterion reached. Function value below -3.85.")
            break
    
        pb.comment = f"Best f value: {fbest}"
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
    # Extrema of test function without constraint: glob_min = -3.86 at (0.114,0.556,0.853)
    if check_convergence:
        assert (fbest <= -3.5 and fbest >= -3.87)

    # Test saving and loading
    save_name = 'gryffin_test_hartmann3D_' + str(test_id) + '.json'
    strategy.save(save_name)
    strategy_2 = GRYFFIN.load(save_name)
    #os.remove(save_name)

    if strategy.prev_param is not None:
        assert strategy.prev_param == strategy_2.prev_param

    if plot:
        fig, ax = hartmann3D.plot()


maximize=False
constraint=False
def test_dro3D():
    hartmann3D = test_functions.Hartmann3D(maximize=False, constraints=constraint)
    strategy = DRO(hartmann3D.domain)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    # initial_exp = pd.DataFrame(data={'x_1': [0.1,0.1,0.4,0.3], 'x_2': [0.6,0.2,0.4,0.5], 'x_3': [1,1,1,0.3]})
    # initial_exp = DataSet.from_df(initial_exp)
    # initial_exp = hartmann3D.run_experiments(initial_exp)

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 200  # maximum number of iterations
    max_stop = 40  # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")

    # Initial experiments
    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        next_experiments, xbest, y, param = \
            strategy.suggest_experiments(prev_res=next_experiments, prev_param=param)

        # This is the part where experiments take place
        next_experiments = hartmann3D.run_experiments(next_experiments)

        print(y)
        if y < fbestold:
            fbestold = y
            nstop = 0
        else:
            nstop += 1
        if nstop >= max_stop:
            print("No improvement in last " + str(max_stop) + " iterations.")
            break
        print(next_experiments)  # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbestold = np.around(fbestold, decimals=3)

    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbestold))

test_dro3D()