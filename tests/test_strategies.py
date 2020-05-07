
import pytest
from summit.domain import Domain, ContinuousVariable, Constraint
from summit.strategies import Random, LHS, NelderMead
from summit.utils.dataset import DataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def test_tsemo():
    pass

def test_nm():
    from summit.domain import Domain, ContinuousVariable
    from summit.strategies import NelderMead
    from summit.utils.dataset import DataSet
    import pandas as pd
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='yield', description='relative conversion to xyz', bounds=[0,100], is_objective=True, maximize=True)
    d = {'temperature': [0.5], 'flowrate_a': [0.6], 'yield': 1}
    df = pd.DataFrame(data=d)
    previous = DataSet.from_df(df)
    strategy = NelderMead(domain)
    strategy.suggest_experiments(prev_res = previous)


def test_nm1():
    # Single-objective optimization problem with 3 dimensional input domain (only continuous inputs)
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='flowrate_b', description='flow of reactant b in mL/min', bounds=[0, 1])
    domain += ContinuousVariable(name='yield', description='relative conversion to xyz',
                                 bounds=[-1000,1000], is_objective=True, maximize=True)
    domain += Constraint(lhs="temperatureflowrate_a+flowrate_b-1", constraint_type="<=") #TODO: implement decoding of constraints
    constraint = False
    strategy =NelderMead(domain)

    # Simulating experiments with hypothetical relationship of inputs and outputs,
    # here Hartmann 3D function: https://www.sfu.ca/~ssurjano/hart3.html
    # Note that SNOBFIT treats constraints implicitly, i.e., for variable sets that
    # violate one of the constraints return NaN as function value (so-called: hidden constraints)
    def sim_fun(x_exp):
        if True:
            x_exp = x_exp[:3]
            A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
            P = np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])*10**(-4)
            alpha = np.array([1,1.2,3.0,3.2])
            d = np.zeros((4,1))
            for k in range(4):
                d[k] = np.sum(np.dot(A[k,:],(x_exp-P[k,:])**2))
            y_exp = - np.sum(np.dot(alpha,np.exp(-d)))
        else:
            y_exp = np.nan

        #x = x_exp
        #y_exp = ((x[0]**2 + x[1] - 11)**2+(x[0] + x[1]**2 -7)**2)-x[2]
        print(y_exp)
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
    initial_exp = pd.DataFrame(data={'temperature': [0.409,0.112,0.17,0.8], 'flowrate_a': [0.424,0.5,0.252,0.1],
                                     'flowrate_b': [0.13,0.8,0.255,0.01]})   # initial experimental points
    initial_exp.insert(3,'yield', test_fun(initial_exp.to_numpy()))   # initial results
    initial_exp = DataSet.from_df(initial_exp)

    # run SNOBFIT loop for fixed <num_iter> number of iteration with <num_experiments> number of experiments each
    # stop loop if <max_stop> consecutive iterations have not produced an improvement
    num_experiments = 4
    num_iter = 10
    max_stop = 1
    nstop = 0
    fbestold = float("inf")
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    #plt.axis([-1, 5, -1, 5, -1,5])
    patches = []
    points = []
    for i in range(num_iter):
        # initial run without history
        if i == 0:
            next_experiments, xbest, fbest, res = strategy.suggest_experiments(prev_res=initial_exp)
            next = initial_exp.data_to_numpy()
            print(next)
            for i in range(len(next)):
                points.append(np.asarray([next[i][:3].tolist()]))
            x = np.asarray([next[i][:3].tolist() for i in range(len(next))])
            print(x)
            polygon = Poly3DCollection(x,alpha=0.1)
            polygon.set_edgecolor('b')
            ax.add_collection3d(polygon)
            #polygon = Polygon(x, True, hatch='x')
            patches.append(polygon)
            next = next_experiments.data_to_numpy()
            points.append(next)
        # runs with history
        else:
            next_experiments_mod = next_experiments.data_to_numpy()
            if True:
                # This is the part where experiments take place
                exp_yield = test_fun(next_experiments.data_to_numpy())
                next_experiments['yield', 'DATA'] = exp_yield
                # Call of SNOBFIT
                next_experiments, xbest, fbest, res = \
                    strategy.suggest_experiments(prev_res=next_experiments, prev_param=res)
                #clear_output(wait=True)
                x1 = [res[0][i][0] for i in range(len(res[0]))]
                x2 = [res[0][i][1] for i in range(len(res[0]))]
                next = next_experiments.data_to_numpy()
                x = np.asarray([res[0][i].tolist() for i in range(len(res[0]))])
                print(x)
                polygon = Poly3DCollection(x, alpha=0.1)
                polygon.set_edgecolor('b')
                ax.add_collection3d(polygon)
                #polygon = Polygon(x, True,hatch= 'x')
                patches.append(polygon)
                points.append(next)
                #plt.plot(x, marker=11)
                #plt.show()
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
        print(next_experiments)
        print(res)
    black = (0, 0, 0, 1)
    #p = PatchCollection(patches, facecolors="None", edgecolors='lightgray')
    #ax.add_collection3d(p)
    for c in range(len(points)):
        print(points[c])
        ax.scatter(points[c][0][0], points[c][0][1], points[c][0][2])
        ax.text(points[c][0][0] + .1, points[c][0][1] + .1, points[c][0][2], c+1, fontsize=9)
    #for c, i in enumerate(points):
    #    ax.annotate("1", (points[c][0],points[c][1]))
    plt.show()
'''
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
        assert (xbest[0] >= 0.113 and xbest[0] <= 0.115) and (xbest[1] >= 0.555 and xbest[1] <= 0.557) and \
               (xbest[2] >= 0.851 and xbest[2] <= 0.853) and (fbest <= -3.85 and fbest >= -3.87)
    else:
        # Extrema of test function with constraint: tbd /TODO: determine optimum with constraint with other algorithms
        assert fbest <= -1
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
'''
#test_nm1()


def test_nm2():
    # Single-objective optimization problem with 3 dimensional input domain (only continuous inputs)
    domain = Domain()
    domain += ContinuousVariable(name='temperature', description='reaction temperature in celsius', bounds=[-4, 0])
    domain += ContinuousVariable(name='flowrate_a', description='flow of reactant a in mL/min', bounds=[-4, 0])
    domain += ContinuousVariable(name='yield', description='relative conversion to xyz',
                                 bounds=[-1000,1000], is_objective=True, maximize=True)
    domain += Constraint(lhs="temperatureflowrate_a+flowrate_b-1", constraint_type="<=") #TODO: implement decoding of constraints
    constraint = False
    strategy =NelderMead(domain)

    # Simulating experiments with hypothetical relationship of inputs and outputs,
    # here Hartmann 3D function: https://www.sfu.ca/~ssurjano/hart3.html
    # Note that SNOBFIT treats constraints implicitly, i.e., for variable sets that
    # violate one of the constraints return NaN as function value (so-called: hidden constraints)
    def sim_fun(x_exp):
        x = x_exp
        y_exp = ((x[0]**2 + x[1] - 11)**2+(x[0] + x[1]**2 -7)**2)
        print(y_exp)
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
    initial_exp = pd.DataFrame(data={'temperature': [-1,-0.5,-0.5], 'flowrate_a': [0,0,-1]})   # initial experimental points
    initial_exp.insert(2,'yield', test_fun(initial_exp.to_numpy()))   # initial results
    initial_exp = DataSet.from_df(initial_exp)

    # run SNOBFIT loop for fixed <num_iter> number of iteration with <num_experiments> number of experiments each
    # stop loop if <max_stop> consecutive iterations have not produced an improvement
    num_experiments = 4
    num_iter = 15
    max_stop = 1
    nstop = 0
    fbestold = float("inf")
    fig, ax = plt.subplots()
    plt.axis([-5, 1, -5, 1])
    patches = []
    points = []
    for i in range(num_iter):
        # initial run without history
        if i == 0:
            next_experiments, xbest, fbest, res = strategy.suggest_experiments(prev_res=initial_exp)
            next = initial_exp.data_to_numpy()
            print(next)
            for i in range(len(next)):
                points.append(np.asarray([next[i][:2].tolist()]))
            x = np.asarray([next[i][:2].tolist() for i in range(len(next))])
            polygon = Polygon(x, True, hatch='x')
            patches.append(polygon)
            next = next_experiments.data_to_numpy()
            points.append(next)
        # runs with history
        else:
            next_experiments_mod = next_experiments.data_to_numpy()
            if True:
                # This is the part where experiments take place
                exp_yield = test_fun(next_experiments.data_to_numpy())
                next_experiments['yield', 'DATA'] = exp_yield
                # Call of SNOBFIT
                next_experiments, xbest, fbest, res = \
                    strategy.suggest_experiments(prev_res=next_experiments, prev_param=res)
                #clear_output(wait=True)
                x1 = [res[0][i][0] for i in range(len(res[0]))]
                x2 = [res[0][i][1] for i in range(len(res[0]))]
                next = next_experiments.data_to_numpy()
                x = np.asarray([res[0][i].tolist() for i in range(len(res[0]))])
                polygon = Polygon(x, True,hatch= 'x')
                patches.append(polygon)
                points.append(next)
                #plt.plot(x, marker=11)
                #plt.show()
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
        print(next_experiments)
        #print(res)
    xlist = np.linspace(-5, 1, 1000)
    ylist = np.linspace(-5, 1, 1000)
    X, Y = np.meshgrid(xlist, ylist)
    Z = (((X**2 + Y - 11)**2+(X + Y**2 -7)**2))
    ax.contour(X,Y,Z, levels=[0.0, 0.5, 1, 1.5, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60 ,70, 80, 90, 100, 150, 200, 300], alpha=0.3)
    p = PatchCollection(patches, facecolors="None", edgecolors='grey', alpha=1)
    ax.add_collection(p)
    for c in range(len(points)):
        print(points[c])
        ax.scatter(points[c][0][0], points[c][0][1])
        ax.text(points[c][0][0] + .01, points[c][0][1] + .01, c+1, fontsize=7)
    ax.axvline(x=0, color='k', linestyle='--')
    ax.axhline(y=0, color='k', linestyle='--')
    plt.show()
'''
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
        assert (xbest[0] >= 0.113 and xbest[0] <= 0.115) and (xbest[1] >= 0.555 and xbest[1] <= 0.557) and \
               (xbest[2] >= 0.851 and xbest[2] <= 0.853) and (fbest <= -3.85 and fbest >= -3.87)
    else:
        # Extrema of test function with constraint: tbd /TODO: determine optimum with constraint with other algorithms
        assert fbest <= -1
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))
'''
test_nm2()