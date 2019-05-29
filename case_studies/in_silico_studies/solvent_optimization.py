""" In-Silico Solvent Optimization"""
from summit.strategies import TSEMO, _pareto_front
from summit.models import GPyModel
from summit.data import solvent_ds, ucb_ds, DataSet
from summit.domain import Domain, DescriptorsVariable,ContinuousVariable
from summit.initial_design import LatinDesigner

import GPy
from deap import base, creator, tools, algorithms
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from tqdm import tqdm 
from notify_run import Notify

from random import Random
import time
from collections import namedtuple
import logging
import warnings
import sys
import json

#Default Constants
constants = {"RANDOM_SEED": 1000,
             "NUM_BATCHES" : 4,
             "BATCH_SIZE": 8,
             "NUM_COMPONENTS": 3,
             "NUM_REPEATS": 35,
             "NOTIFY_ENDPOINT": 'https://notify.run/x1W5cRESQDhdaJLt',
}


def create_pcs_ds(num_components, ucb_filter=True, verbose=False):
    '''Create dataset with principal components'''
    #Read in solubility data
    solubilities = pd.read_csv('inputs/solubilities.csv')
    solubilities = solubilities.set_index('cas_number')
    solubilities = DataSet.from_df(solubilities)

    #Merge data sets
    solvent_ds_full = solvent_ds.join(solubilities)
    if ucb_filter:
        solvent_ds_final = pd.merge(solvent_ds_full, ucb_ds, left_index=True,right_index=True)
    else:
        solvent_ds_final = solvent_ds_full
    if verbose:
        print(f"{solvent_ds_final.shape[0]} solvents for optimization")

    #Double check that there are no NaNs in the descriptors
    values = solvent_ds_final.data_to_numpy()
    values = values.astype(np.float64)
    check = np.isnan(values)
    assert check.all() == False

    #Transform to principal componets
    pca = PCA(n_components=num_components)
    pca.fit(solvent_ds_full.standardize())
    pcs = pca.fit_transform(solvent_ds_final.standardize())
    if verbose:
        explained_var = round(pca.explained_variance_ratio_.sum()*100)
        expl = f"{explained_var}% of variance is explained by {num_components} principal components."
        print(expl)

    #Create a new dataset with just the principal components
    metadata_df = solvent_ds_final.loc[:, solvent_ds_final.metadata_columns]
    pc_df = pd.DataFrame(pcs, columns = [f'PC_{i+1}' for i in range(num_components)], 
                        index=metadata_df.index)
    pc_ds = DataSet.from_df(pc_df)
    return pd.concat([metadata_df, pc_ds], axis=1), pca

class Experiments:
    ''' Generate data to simulate a stereoselective chemical reaction 
    
    Parameters
    ----------- 
    solvent_ds: Dataset
        Dataset with the solvent descriptors (must have cas numbers as index)
    random_state: `numpy.random.RandomState`, optional
        RandomState object. Creates a random state based ont eh computer clock
        if one is not passed
    pre_calculate: bool, optional
        If True, pre-calculates the experiments for all solvents. Defaults to False

    Notes
    -----
    Pre-calculating will ensure that multiple calls to experiments will give the same result
    (as long as a random state is specified).
    
    ''' 
    def __init__(self, solvent_ds, 
                 random_state=np.random.RandomState(),
                 initial_concentrations=[0.5,0.5],
                 pre_calculate=False):
        self.solvent_ds = solvent_ds
        self.initial_concentrations = initial_concentrations
        self.random_state = random_state
        self.pre_calculate = pre_calculate
        self.cas_numbers = self.solvent_ds.index.values.tolist()
        if pre_calculate:
            all_experiments = [self._run(cas) for cas in self.cas_numbers]
            self.all_experiments = np.array(all_experiments)                
        else:
            self.all_experiments = None
        
    def run(self, solvent_cas):
        if self.all_experiments is None:
            result = self._run(solvent_cas)
        else:
            index = self.cas_numbers.index(solvent_cas)
            result = self.all_experiments[index, :]
        return result

    def _run(self, solvent_cas, rxn_time=25200, step_size=200,
             time_series=False):
        '''Generate fake experiment data for a stereoselective reaction'''
        rxn_time = rxn_time + self.random_state.randn(1)*0.01*rxn_time
        x = self._integrate_rate(solvent_cas, rxn_time, step_size)
        cd1 = x[:,0] 
        cd2 = x[:,1]

        conversion = cd1/np.min(self.initial_concentrations)*100
        de = cd1/(cd1+cd2)*100

        if not time_series:
            conversion = conversion[-1]
            de = de[-1]
        return np.array([conversion, de])

    def _integrate_rate(self,solvent_cas, t=25200, step_size=200):
        '''Calculate extent of reaction for a particular solvent over a given time range'''
        t0=0 #Have to start from time 0
        x0 = 0 # Always start with 0 extent of reaction
        trange = np.linspace(t0, t, (t-t0)/step_size)
        x = np.zeros([len(trange), len(self.initial_concentrations)])
        r = integrate.ode(self._rate).set_integrator("vode")
        r.set_initial_value([x0, x0], t0)
        r.set_f_params(solvent_cas)
        for i in range(1, trange.size):
            x[i, :] = r.integrate(trange[i])
            if not r.successful():
                raise RuntimeError(f"Could not integrate: {solvent_cas}")
        return x

    def _rate(self, t, x, solvent_cas):
        '''Calculate  rates  for a given extent of reaction'''
        #Constants
        AD1 = 8.5e9   #L/(mol-s)
        AD2 = 8.3e9   #L/(mol-s)
        EAD1 = 105 # kJ/mol
        EAD2 = 110 # kJ/mol
        TRXN = 393 #K
        R = 8.314e-3 # kJ/mol/K
        Es1 = lambda pc1, pc2, pc3: -np.log(abs((pc2+0.73*pc1-4.46)*(pc2+2.105*pc1+11.367)))+ pc3
        Es2 = lambda pc1, pc2, pc3: -2*np.log(abs((pc2+0.73*pc1-4.46))) - 0.2*pc3**2

        #Solvent variable reaction rate coefficients
        pc_solvent = self.solvent_ds.loc[solvent_cas][self.solvent_ds.data_columns].to_numpy()
        es1 = Es1(pc_solvent[0], pc_solvent[1], pc_solvent[2])
        es2 = Es2(pc_solvent[0], pc_solvent[1], pc_solvent[2])
        # T = 0.5 * self.random_state.randn() + TRXN
        T=TRXN
        kd1 = AD1*np.exp(-(EAD1+es1)/(R*T))
        kd2 = AD2*np.exp(-(EAD2+es2)/(R*T))
        
        #Calculate rates
        x1dot = kd1*self.ca(x)*self.cb(x)
        x2dot = kd2*self.ca(x)*self.cb(x)
        return np.array([x1dot, x2dot])

    def ca(self, x):
        try:
            ca = self.initial_concentrations[0]-x[:,0]-x[:,1]   
        except IndexError:
            ca = self.initial_concentrations[0]-x[0]-x[1]
        return ca

    def cb(self, x):
        try:
            cb = self.initial_concentrations[1]-x[:, 0]-x[:,1]   
        except IndexError:
            cb = self.initial_concentrations[1]-x[0]-x[1]
        return cb

#Create  optimization domain
def create_domain(solvent_ds):
    '''Create solvent optimization domain'''
    domain = Domain()
    domain += DescriptorsVariable(name='solvent',
                                description='solvent for the borrowing hydrogen reaction',
                                ds=solvent_ds)
    domain += ContinuousVariable(name='conversion',
                                description='relative conversion to triphenylphosphine oxide determined by LCMS',
                                bounds=[0, 100],
                                is_output=True)
    domain += ContinuousVariable(name='de',
                                description='diastereomeric excess determined by ratio of LCMS peaks',
                                bounds=[0, 100],
                                is_output=True)
    return domain


def optimization_setup(domain, random_rate=0.0):
    '''Setup TSEMOish optimization'''
    input_dim = domain.num_continuous_dimensions()+domain.num_discrete_variables()
    kernels = [GPy.kern.Matern52(input_dim = input_dim, ARD=True)
            for _ in range(2)]
    models = [GPyModel(kernel=kernels[i]) for i in range(2)]
    return TSEMO(domain, models, random_rate=random_rate)

def generate_initial_experiment_data(domain, solvent_ds, batch_size, random_state,
                                     experiments, criterion='center'):
    '''Generate initial experimental design'''
    #Initial design
    lhs = LatinDesigner(domain,random_state)
    initial_design = lhs.generate_experiments(batch_size, criterion=criterion, unique=True)

    #Initial experiments
    initial_experiments = [experiments.run(cas) 
                           for cas in initial_design.to_frame()['cas_number']]
    initial_experiments = np.array(initial_experiments)
    initial_experiments = DataSet({('conversion', 'DATA'): initial_experiments[:, 0],
                                   ('de', 'DATA'): initial_experiments[:, 1],
                                   ('solvent', 'DATA'): initial_design.to_frame()['cas_number'].values,
                                   ('batch', 'METADATA'): np.zeros(batch_size, dtype=int)})
    initial_experiments.columns.names = ['NAME', 'TYPE']
    initial_experiments = initial_experiments.set_index(np.arange(0, batch_size))
    return initial_experiments

optimization_results = namedtuple('optimization_results', ('experiments', 'lengthscales', 'log_likelihoods', 'loo_cv_errors', 'hv_improvements'))

def run_optimization(tsemo, initial_experiments,solvent_ds,
                     batch_size, num_batches,
                     num_components, random_state, 
                     experiments,
                     normalize_inputs=False,
                     normalize_outputs=False):
    '''Run an optimization'''
    #Create storage arrays
    lengthscales = np.zeros([num_batches-1, num_components, 2])
    log_likelihoods = np.zeros([num_batches-1, 2])
    loo_errors = np.zeros([num_batches-1, 2])
    hv_improvements = np.zeros([num_batches-1])
    previous_experiments = initial_experiments
                    
    #Run the optimization
    for i in range(num_batches-1):
        #Generate batch of solvents
        last_batch_number = previous_experiments['batch'].max()
        logging.debug(f'Batch {i+1}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            design, hv_imp = tsemo.generate_experiments(previous_experiments, batch_size, 
                                                        normalize_inputs=normalize_inputs,
                                                        normalize_outputs=normalize_outputs)

        #Calculate model parameters for further analysis
        lengthscales[i, :, :] = np.array([model._model.kern.lengthscale.values for model in tsemo.models]).T
        log_likelihoods[i, :] = np.array([model._model.log_likelihood() for model in tsemo.models]).T
        hv_improvements[i] = hv_imp
        logging.debug(f'Calculating loo errors')
        loo_errors[i, :] = tsemo.loo_errors()
        

        #Run the "experiments"                                    
        new_experiments = [experiments.run(cas)
                           for cas in design.index.values]
        new_experiments = np.array(new_experiments)

        #Combine new experimental data with old data
        new_experiments = DataSet({('conversion', 'DATA'): new_experiments[:, 0],
                                   ('de', 'DATA'): new_experiments[:, 1],
                                   ('solvent', 'DATA'): design.index.values,
                                   ('batch', 'METADATA'): last_batch_number+1})
        last_index = previous_experiments.index.max()
        new_index = np.arange(last_index+1, last_index + 1 + batch_size)
        new_experiments = new_experiments.set_index(new_index)
        new_experiments.columns.names = ['NAME', 'TYPE']
        previous_experiments = previous_experiments.append(new_experiments)

    return optimization_results(experiments=previous_experiments, 
                                lengthscales=lengthscales,
                                log_likelihoods=log_likelihoods,
                                loo_cv_errors=loo_errors, 
                                hv_improvements=hv_improvements)

def descriptors_optimization(save_to_disk=True, **kwargs):
    '''Setup and run a descriptors optimization'''
    #Get keyword arguments
    ucb_filter = kwargs.get('ucb_filter', True)
    batch_size = kwargs.get('batch_size')
    num_batches = kwargs.get('num_batches')
    num_components = kwargs.get('num_components')
    num_initial_experiments = kwargs.get('num_initial_experiments')
    if num_initial_experiments is None:
        raise ValueError("Put a number of initial experiments")
    random_seed = kwargs.get('random_seed', constants['RANDOM_SEED'])
    normalize_inputs = kwargs.get('normalize_inputs', True)
    normalize_outputs = kwargs.get('normalize_outputs', True)
    design_criterion = kwargs.get('design_criterion', 'center')
    random_rate = kwargs.get('random_rate', 0.0)

    #Setup
    random_state = np.random.RandomState(random_seed)
    solvent_pcs_ds, _ = create_pcs_ds(num_components=num_components, ucb_filter=ucb_filter)
    exp = Experiments(solvent_ds=solvent_pcs_ds, random_state=random_state, pre_calculate=True)
    domain = create_domain(solvent_pcs_ds)
    tsemo = optimization_setup(domain, random_rate)
    initial_experiments = generate_initial_experiment_data(domain=domain,
                                                           solvent_ds=solvent_pcs_ds,
                                                           batch_size=num_initial_experiments,
                                                           random_state=random_state,
                                                           experiments=exp,
                                                           criterion=design_criterion)
    #Run optimization
    result = run_optimization(tsemo=tsemo, 
                              initial_experiments=initial_experiments, 
                              solvent_ds=solvent_pcs_ds,
                              experiments=exp,
                              num_batches=num_batches,
                              batch_size=batch_size, 
                              num_components=num_components,
                              random_state=random_state)

    # Write parameters to disk
    if save_to_disk:
        if save_to_disk is str:
            output_prefix = save_to_disk
        else:
            output_prefix = ''
        result.experiments.to_csv(f'outputs/{output_prefix}_in_silico_experiments.csv')
        np.save(f'outputs/{output_prefix}_in_silico_lengthscales', result.lengthscales)
        np.save(f'outputs/{output_prefix}_in_silico_log_likelihoods', result.log_likelihoods)
        np.save(f'outputs/{output_prefix}_in_silico_loo_errors', result.loo_cv_errors)
        np.save(f'outputs/{output_prefix}_in_silico_hv_improvements', result.hv_improvements)
        metadata = {"num_principal_components": num_components,
                    "num_batches": num_batches,
                    "batch_size": batch_size,
                    'normalize_inputs': normalize_inputs,
                    'normalize_outputs': normalize_outputs,
                    'repeat_iteration': i+1,
                    'design_criterion': design_criterion,
                    'num_initial_experiments': num_initial_experiments,
                    'random_rate': random_rate}
    
        with open(f'outputs/{output_prefix}_in_silico_metadata.json',  'w') as f:
            json.dump(metadata, f)

    return result

def repeat_test(num_repeats, callback=None, ucb_filter=True):
    '''Test various optimization parameters with repeats'''
    #Create a full factorial design
    num_components = constants['NUM_COMPONENTS']
    params = {'batch_size': [4, 8],
              'design_criterion': ['center', 'maximin'],
              'num_initial_experiments': [8, 16], 
              'random_rate': [0.0, 0.5]}
    
    levels = [len(params[key]) for key in params]
    doe = fullfact(levels)
    designs = [{key: params[key][int(index)] for key, index in zip(params, d)}
              for d in doe]

    for j, d in enumerate(designs):
        info = f'Starting design {j+1} out {len(designs)}.'
        print(info)
        if callback is not None and callable(callback):
            try:
                callback(info)
            except Exception as e:
                logging.debug(e)
          
        #Create arrays for summaries
        num_batches = (40-d['num_initial_experiments'])// d['batch_size'] + 1
        lengthscales = np.zeros([num_repeats, num_batches-1, num_components, 2])
        log_likelihoods = np.zeros([num_repeats, num_batches-1, 2])
        loo_cv_errors = np.zeros([num_repeats, num_batches-1, 2])
        hv_improvements = np.zeros([num_repeats, num_batches-1])
        

        for i in tqdm(range(num_repeats)):
            res =  descriptors_optimization(batch_size=d['batch_size'],
                                            num_batches=num_batches,
                                            num_initial_experiments=d['num_initial_experiments'],
                                            num_components = num_components,
                                            random_rate = d['random_rate'],
                                            random_seed=constants['RANDOM_SEED']+100*i,
                                            normalize_inputs=True,
                                            normalize_outputs=False,
                                            design_criterion=d['design_criterion'],
                                            ucb_filter=ucb_filter,
                                            save_to_disk=False)
            lengthscales[i, :, :, :] = res.lengthscales
            log_likelihoods[i, :, :] = res.log_likelihoods
            loo_cv_errors[i, :, :] = res.loo_cv_errors
            hv_improvements[i, :] = res.hv_improvements
        
            output_prefix=f'test_{j}'
            res.experiments.to_csv(f'outputs/{output_prefix}_iteration_{i}_in_silico_experiments.csv')
            np.save(f'outputs/{output_prefix}_in_silico_lengthscales', lengthscales)
            np.save(f'outputs/{output_prefix}_in_silico_log_likelihoods', log_likelihoods)
            np.save(f'outputs/{output_prefix}_in_silico_loo_errors', loo_cv_errors)
            np.save(f'outputs/{output_prefix}_in_silico_hv_improvements', hv_improvements)
            metadata = {"num_principal_components": num_components,
                        "num_batches": num_batches,
                        "batch_size": d['batch_size'],
                        'normalize_inputs': True,
                        'normalize_outputs': False,
                        'design_criterion': d['design_criterion'],
                        'repeat_iteration': i+1,
                        'num_initial_experiments': d['num_initial_experiments'], 
                        'random_rate': d['random_rate']}
        
            with open(f'outputs/{output_prefix}_in_silico_metadata.json',  'w') as f:
                json.dump(metadata, f)

def random_test(num_repeats, callback=None):
    '''Test various random rates'''
    #Create a full factorial design
    num_components = constants['NUM_COMPONENTS']
    params = {'batch_size': [4],
              'design_criterion': ['maximin'],
              'num_initial_experiments': [8], 
              'random_rate': [0.2, 0.4, 0.6, 0.8, 1]}
    
    levels = [len(params[key]) for key in params]
    doe = fullfact(levels)
    designs = [{key: params[key][int(index)] for key, index in zip(params, d)}
              for d in doe]

    for j, d in enumerate(designs):
        info = f'Starting design {j+1} out {len(designs)}.'
        print(info)
        if callback is not None and callable(callback):
            try:
                callback(info)
            except Exception as e:
                logging.debug(e)
          
        j += 16 #Start at sixteen since already ran designs #0 - 15
        #Create arrays for summaries
        num_batches = (40-d['num_initial_experiments'])// d['batch_size'] + 1
        lengthscales = np.zeros([num_repeats, num_batches-1, num_components, 2])
        log_likelihoods = np.zeros([num_repeats, num_batches-1, 2])
        loo_cv_errors = np.zeros([num_repeats, num_batches-1, 2])
        hv_improvements = np.zeros([num_repeats, num_batches-1])
        
        for i in tqdm(range(num_repeats)):
            res =  descriptors_optimization(batch_size=d['batch_size'],
                                            num_batches=num_batches,
                                            num_initial_experiments=d['num_initial_experiments'],
                                            num_components = num_components,
                                            random_rate = d['random_rate'],
                                            random_seed=constants['RANDOM_SEED']+100*i,
                                            normalize_inputs=True,
                                            normalize_outputs=False,
                                            design_criterion=d['design_criterion'],
                                            save_to_disk=False)
            lengthscales[i, :, :, :] = res.lengthscales
            log_likelihoods[i, :, :] = res.log_likelihoods
            loo_cv_errors[i, :, :] = res.loo_cv_errors
            hv_improvements[i, :] = res.hv_improvements
        
            output_prefix=f'test_{j}'
            res.experiments.to_csv(f'outputs/{output_prefix}_iteration_{i}_in_silico_experiments.csv')
            np.save(f'outputs/{output_prefix}_in_silico_lengthscales', lengthscales)
            np.save(f'outputs/{output_prefix}_in_silico_log_likelihoods', log_likelihoods)
            np.save(f'outputs/{output_prefix}_in_silico_loo_errors', loo_cv_errors)
            np.save(f'outputs/{output_prefix}_in_silico_hv_improvements', hv_improvements)
            metadata = {"num_principal_components": num_components,
                        "num_batches": num_batches,
                        "batch_size": d['batch_size'],
                        'normalize_inputs': True,
                        'normalize_outputs': False,
                        'design_criterion': d['design_criterion'],
                        'repeat_iteration': i+1,
                        'num_initial_experiments': d['num_initial_experiments'], 
                        'random_rate': d['random_rate']}
        
            with open(f'outputs/{output_prefix}_in_silico_metadata.json',  'w') as f:
                json.dump(metadata, f)

def fullfact(levels):
    """Full factorial design from pyDoE"""
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))
    
    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng
     
    return H

def calculate_effects(doe, params, outputs, include_names=False):
    '''Calculate primary and secondary effects of a DoE study'''
    num_params = len(params)
    pmary_effects = np.zeros(num_params)
    pmary_names = []
    sdary_effects = np.zeros((num_params-1)*num_params)
    sdary_names = (num_params-1)*num_params*['']
    ones = np.nonzero(doe)
    zeros = np.where(doe==0)
    j=0
    for param_index, key in enumerate(params):
        highs_where = np.where(ones[1] == param_index)[0]
        lows_where = np.where(zeros[1]==param_index)[0]
        highs_indices = ones[0][highs_where]
        lows_indices = zeros[0][lows_where]
        
        #Calculate primary effects
        highs = outputs[highs_indices]
        lows = outputs[lows_indices]
        pmary_effects[param_index] = 0.5*(np.sum(highs-lows))/highs.shape[0]
        pmary_names.append(key)

        #Calculate secondary effect
        for sdary_index, sdary_key in enumerate(params):
            if sdary_index == param_index: continue
            sdary_highs_where = np.where(ones[1] == sdary_index)[0]
            sdary_lows_where = np.where(zeros[1]==sdary_index)[0]
            sdary_highs_indices = ones[0][sdary_highs_where]
            sdary_lows_indices = zeros[0][sdary_lows_where]
            
            sdary_highs_indices = np.intersect1d(highs_indices, sdary_highs_indices)
            sdary_lows_indices  = np.intersect1d(lows_indices, sdary_lows_indices)
            
            sdary_highs = outputs[sdary_highs_indices]
            sdary_lows = outputs[sdary_lows_indices]
            sdary_effects[j] = 0.5*(np.sum(sdary_highs-sdary_lows))/sdary_highs.shape[0]
            sdary_names[j] = f"{key}*{sdary_key}"
            j+=1
    if include_names:
        result = pmary_effects, sdary_effects, pmary_names, sdary_names
    else:
        result = pmary_effects, sdary_effects
    return result

def send_warnings_to_log(message, category, filename, lineno, file=None, line=None):
    return ' %s:%s: %s:%s' % (filename, lineno, category.__name__, message)

if __name__ == '__main__':
    warnings.showwarning = send_warnings_to_log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(filename=f"outputs/in_silico_optimization_log.txt",level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    notify=Notify(constants['NOTIFY_ENDPOINT'])
    # repeat_test(constants['NUM_REPEATS'], callback=notify.send)
    # random_test(constants['NUM_REPEATS'], callback=notify.send)
    repeat_test(constants['NUM_REPEATS'], callback=notify.send, ucb_filter=False)