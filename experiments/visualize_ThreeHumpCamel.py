import pytest

from summit.benchmarks import *
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.utils.multiobjective import pareto_efficient, hypervolume
from summit.strategies import *

from fastprogress.fastprogress import progress_bar
import numpy as np
import os
import warnings

import matplotlib.pyplot as plt

def test_nm_thc(x_start,maximize,constraint, plot=False):

    thcamel = test_functions.ThreeHumpCamel(maximize=maximize, constraints=constraint)
    strategy = NelderMead(thcamel.domain, x_start=x_start, adaptive=False)

    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    #initial_exp = pd.DataFrame(data={'x_1': [4.0,4.0,2.0], 'x_2': [2.0,3.0,-6.0]})
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = himmelblau.run_experiments(initial_exp)  # initial results

    # run Nelder-Mead loop for fixed <num_iter> number of iteration
    num_iter = 17   # maximum number of iterations
    max_stop = 10   # allowed number of consecutive iterations w/o improvement
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
        next_experiments = \
            strategy.suggest_experiments(prev_res=next_experiments)\

        # This is the part where experiments take place
        next_experiments = thcamel.run_experiments(next_experiments)

        param = strategy.prev_param
        print(param)
        # save polygon points for plotting
        polygons_points.append(np.asarray([param[0]["sim"][i].tolist() for i in range(len(param[0]["sim"]))]))


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

        print(next_experiments)   # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    #assert fbest <= 0.1
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    # plot
    if plot:
        fig, ax = thcamel.plot(polygons=polygons_points)
        plt.show()

#test_nm_thc([1,1],False, False, True)
#test_nm_thc([-1,-2],False, False, True)


def test_snobfit_thc(num_experiments, maximize, constraints, plot=False):

    thcamel = test_functions.ThreeHumpCamel(maximize=maximize, constraints=constraints)
    strategy = SNOBFIT(thcamel.domain, probability_p=0.5, dx_dim=1E-5)

    initial_exp = None
    # Comment out to start without initial data
    #initial_exp = pd.DataFrame(data={'x_1': [0.409,0.112,0.17,0.8], 'x_2': [0.424,0.33,0.252,0.1],
    #                                 'x_3': [0.13,0.3,0.255,0.01]})   # initial experimental points
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = hartmann3D.run_experiments(initial_exp)   # initial results

    # run SNOBFIT loop for fixed <num_iter> number of iteration with <num_experiments> number of experiments each
    # stop loop if <max_stop> consecutive iterations have not produced an improvement
    num_iter = 5
    max_stop = 50//num_experiments
    nstop = 0
    fbestold = float("inf")

    #Initial experiments
    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        # Call of SNOBFIT
        next_experiments = \
            strategy.suggest_experiments(num_experiments, prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = thcamel.run_experiments(next_experiments)

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
        print(next_experiments)   # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)

    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    # plot
    if plot:
        fig, ax = thcamel.plot()
        plt.show()

#test_snobfit_thc(4,False,False,True)

def test_sobo_thc(num_experiments, maximize, constraint, plot=False):

    thcamel = test_functions.ThreeHumpCamel(maximize=maximize, constraints=constraint)
    strategy = SOBO(domain=thcamel.domain)

    # Uncomment to start algorithm with pre-defined initial experiments
    initial_exp = None
    # Uncomment to create test case which results in reduction dimension and dimension recovery
    #initial_exp = pd.DataFrame(data={'x_1': [0.1,0.1,0.4,0.3], 'x_2': [0.6,0.2,0.4,0.5], 'x_3': [1,1,1,0.3]})   # initial experimental points
    #initial_exp = DataSet.from_df(initial_exp)
    #initial_exp = hartmann3D.run_experiments(initial_exp)

    # run SOBO loop for fixed <num_iter> number of iteration
    num_iter = 5   # maximum number of iterations
    max_stop = 80//num_experiments   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")

    if initial_exp is not None:
        next_experiments = initial_exp
    else:
        next_experiments = None

    param = None
    for i in range(num_iter):
        next_experiments = \
            strategy.suggest_experiments(num_experiments=num_experiments, prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = thcamel.run_experiments(next_experiments)

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

        print(next_experiments)  # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    if plot:
        fig, ax = thcamel.plot()
        plt.show()

#stest_sobo_thc(4, False, False, True)

def test_gryffin_thc(num_experiments, maximize, constraint, plot=False):

    thcamel = test_functions.ThreeHumpCamel(maximize=maximize, constraints=constraint)
    strategy = GRYFFIN(domain=thcamel.domain, sampling_strategies=num_experiments)

    # run SOBO loop for fixed <num_iter> number of iteration
    num_iter = 20   # maximum number of iterations
    max_stop = 80   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    next_experiments = None
    for i in range(num_iter):
        next_experiments= \
            strategy.suggest_experiments(prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = thcamel.run_experiments(next_experiments)


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

        print(next_experiments)  # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    if plot:
        fig, ax = thcamel.plot()
        plt.show()

#test_gryffin_thc(1, False, False, True)


def test_dro_thc(num_experiments, maximize, constraint, plot=False):

    thcamel = test_functions.ThreeHumpCamel(maximize=maximize, constraints=constraint)
    strategy = DRO(domain=thcamel.domain)

    # run SOBO loop for fixed <num_iter> number of iteration
    num_iter = 20   # maximum number of iterations
    max_stop = 80   # allowed number of consecutive iterations w/o improvement
    nstop = 0
    fbestold = float("inf")
    next_experiments = None
    for i in range(num_iter):
        next_experiments= \
            strategy.suggest_experiments(prev_res=next_experiments)

        # This is the part where experiments take place
        next_experiments = thcamel.run_experiments(next_experiments)


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

        print(next_experiments)  # show next experiments
        print("\n")

    xbest = np.around(xbest, decimals=3)
    fbest = np.around(fbest, decimals=3)
    print("Optimal setting: " + str(xbest) + " with outcome: " + str(fbest))

    if plot:
        fig, ax = thcamel.plot()
        plt.show()

#test_dro_thc(1, False, False, True)

