""" In-Silico Solvent Optimization"""
from surrogate_model_functions import loo_error
from summit.strategies import TSEMO, _pareto_front
from summit.models import GPyModel
from summit.data import solvent_ds, ucb_ds, DataSet
from summit.domain import Domain, DescriptorsVariable,ContinuousVariable
from summit.initial_design import LatinDesigner

import GPy
import inspyred
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import Random
import time
from collections import namedtuple

#Default Constants
constants = {"RANDOM_SEED": 1000,
             "NUM_BATCHES" : 4,
             "BATCH_SIZE": 8,
             "NUM_COMPONENTS": 3
}


def create_pcs_ds(num_components, verbose=False):
    '''Create dataset with principal components'''
    #Read in solubility data
    solubilities = pd.read_csv('inputs/solubilities.csv')
    solubilities = solubilities.set_index('cas_number')
    solubilities = DataSet.from_df(solubilities)

    #Merge data sets
    solvent_ds_full = solvent_ds.join(solubilities)
    solvent_ds_final = pd.merge(solvent_ds_full, ucb_ds, left_index=True,right_index=True)
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
    explained_var = round(pca.explained_variance_ratio_.sum()*100)
    if verbose:
        expl = f"{explained_var}% of variance is explained by {num_components} principal components."
        print(expl)

    #Create a new dataset with just the principal components
    metadata_df = solvent_ds_final.loc[:, solvent_ds_final.metadata_columns]
    pc_df = pd.DataFrame(pcs, columns = [f'PC_{i+1}' for i in range(num_components)], 
                        index=metadata_df.index)
    pc_ds = DataSet.from_df(pc_df)
    return pd.concat([metadata_df, pc_ds], axis=1)

# Set up test problem
AD1 = 8.5
AD2 = 0.7
EAD1 = 50
EAD2 = 70
R = 8.314
cd1 = lambda t, T, Es: AD1*t*np.exp(-(EAD1+Es)/T)
cd2 = lambda t, T, Es: AD2*t*np.exp(-(EAD2+Es)/T)
Es1 = lambda pc1, pc2, pc3: -20*pc2*abs(pc3) + 0.025*pc1**3
Es2 = lambda pc1, pc2, pc3: 15*pc2*pc3-40*pc3**2

def experiment(solvent_cas, solvent_ds, random_state=np.random.RandomState()):
    '''Generate fake experiment data for a stereoselective reaction'''
    pc_solvent = solvent_ds.loc[solvent_cas][solvent_ds.data_columns].to_numpy()
    es1 = Es1(pc_solvent[0], pc_solvent[1], pc_solvent[2])
    es2 = Es2(pc_solvent[0], pc_solvent[1], pc_solvent[2])
    T = 5 * random_state.randn(1) + 393
    t = 0.1 * random_state.randn(1) + 7
    exper_cd1 = cd1(t, T, es1)
    exper_cd2 = cd2(t, T, es2)

    #Conversion with noise
    conversion = exper_cd1 + exper_cd2
    max_conversion = 95.0 + random_state.randn(1)*2
    conversion = min(max_conversion[0], conversion[0])
    conversion = conversion + random_state.randn(1)*2
    conversion=conversion[0]
                     
    #Diasteromeric excess with noise
    de = abs(exper_cd1-exper_cd2)/(exper_cd1 +exper_cd2)
    max_de =  0.98 + random_state.randn(1)*0.02
    de = min(max_de[0], de[0])
    de = de + random_state.randn(1)*0.02
    de = de[0]
    return np.array([conversion, de*100])

#Create  optimization domain
def create_domain(solvent_ds):
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

def optimization_setup(domain):
    input_dim = domain.num_continuous_dimensions()+domain.num_discrete_variables()
    kernels = [GPy.kern.Matern52(input_dim = input_dim, ARD=True)
            for _ in range(2)]
    models = [GPyModel(kernel=kernels[i]) for i in range(2)]
    return TSEMO(domain, models)

def generate_initial_experiment_data(domain, solvent_ds, batch_size, 
                                     random_state, criterion='center'):
    #Initial design
    lhs = LatinDesigner(domain,random_state)
    initial_design = lhs.generate_experiments(batch_size, criterion=criterion)

    #Initial experiments
    initial_experiments = [experiment(cas, solvent_ds, random_state) 
                           for cas in initial_design.to_frame()['cas_number']]
    initial_experiments = np.array(initial_experiments)
    initial_experiments = DataSet({('conversion', 'DATA'): initial_experiments[:, 0],
                                   ('de', 'DATA'): initial_experiments[:, 1],
                                   ('solvent', 'DATA'): initial_design.to_frame()['cas_number'].values,
                                   ('batch', 'METADATA'): np.zeros(batch_size, dtype=int)})
    initial_experiments.columns.names = ['NAME', 'TYPE']
    initial_experiments = initial_experiments.set_index(np.arange(0, batch_size))
    return initial_experiments

optimization_results = namedtuple('optimization_analystics', ('experiments', 'lengthscales', 'log_likelihoods', 'loo_cv_errors', 'hv_improvements'))

def run_optimization(tsemo, initial_experiments,solvent_ds,
                     batch_size, num_batches,
                     num_components, random_state, 
                     normalize_inputs=False,
                     normalize_outputs=False):
    #Create storage arrays
    lengthscales = np.zeros([num_batches-1, num_components, 2])
    log_likelihoods = np.zeros([num_batches-1, 2])
    loo_errors = np.zeros([num_batches-1, 2])
    hv_improvements = np.zeros([num_batches-1])
    previous_experiments = initial_experiments


    #Run the optimization
    for i in range(num_batches-1):
        #Generate batch of solvents
        design, hv_imp = tsemo.generate_experiments(previous_experiments, batch_size, 
                                                    normalize_inputs=normalize_inputs,
                                                    normalize_outputs=normalize_outputs)

        #Calculate model parameters for further analysis
        lengthscales[i, :, :] = np.array([model._model.kern.lengthscale.values for model in tsemo.models]).T
        log_likelihoods[i, :] = np.array([model._model.log_likelihood() for model in tsemo.models]).T
        hv_improvements[i] = hv_imp
        for j in range(2):
            loo_errors[i, j] = loo_error(tsemo.x, np.atleast_2d(tsemo.y[:, j]).T)
        

        #Run the "experiments"                                    
        new_experiments = [experiment(cas, solvent_ds, random_state)
                           for cas in design.index.values]
        new_experiments = np.array(new_experiments)

        #Combine new experimental data with old data
        new_experiments = DataSet({('conversion', 'DATA'): new_experiments[:, 0],
                                   ('de', 'DATA'): new_experiments[:, 1],
                                   ('solvent', 'DATA'): design.index.values,
                                   ('batch', 'METADATA'): (i+1)*np.ones(batch_size, dtype=int)})
        new_experiments = new_experiments.set_index(np.arange(batch_size*(i+1), batch_size*(i+2)))
        new_experiments.columns.names = ['NAME', 'TYPE']
        previous_experiments = previous_experiments.append(new_experiments)

    return optimization_results(experiments=previous_experiments, 
                                lengthscales=lengthscales,
                                log_likelihoods=log_likelihoods,
                                loo_cv_errors=loo_errors, 
                                hv_improvements=hv_improvements)

def pareto_coverage(pareto_front, design):
    '''Calculate the percentage of the pareto front covered by a design'''
    pareto_size = pareto_front.shape[0]
    num_covered = 0
    for point in design:
        index = np.where(np.isclose(pareto_front, point).all(axis=1))[0]
        if len(index) == 0:
            continue
        else:
            num_covered += 1
    return num_covered/pareto_size

def descriptors_optimization(save_to_disk=True, **kwargs):
    batch_size = kwargs.get('batch_size', constants['BATCH_SIZE'])
    num_batches = kwargs.get('num_batches',constants['NUM_BATCHES'])
    num_components = kwargs.get('num_components',constants['NUM_COMPONENTS'])
    random_seed = kwargs.get('random_seed', constants['RANDOM_SEED'])
    normalize_inputs = kwargs.get('normalize_inputs', True)
    normalize_outputs = kwargs.get('normalize_outputs', True)
    design_criterion = kwargs.get('design_criterion', None)

    random_state = np.random.RandomState(random_seed)
    solvent_pcs_ds = create_pcs_ds(num_components=num_components)
    domain = create_domain(solvent_pcs_ds)
    tsemo = optimization_setup(domain)
    initial_experiments = generate_initial_experiment_data(domain,
                                                           solvent_pcs_ds,
                                                           batch_size,
                                                           random_state,
                                                           criterion=design_criterion)
    result = run_optimization(tsemo, initial_experiments, 
                              solvent_pcs_ds,
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
        to_print = [('Random seed', random_seed),
                    ("Number of principal components", num_components),
                    ("Number of batches", num_batches),
                    ("Batch size",batch_size)]
        
        with open(f'outputs/{output_prefix}_in_silico_metadata.txt',  'w') as f:
            for prefix, value in to_print:
                txt = f"{prefix}: {value}"
                print(txt)
                f.write(txt)
    
    return result

def repeat_test(num_repeats=1000):
    '''Test various optimization parameters'''
    #Create a full factorial design
    num_components = constants['NUM_COMPONENTS']
    params = {'normalize_inputs': [False, True],
              'normalize_outputs': [False, True],
              'batch_size': [4, 8],
              'design_criterion': ['center', 'maximin', 'centermaximin', 'correlation']}
    
    levels = [len(params[key]) for key in params]
    doe = fullfact(levels)
    design = [{key: params[key][int(index)] for key, index in zip(params, d)}
              for d in doe]

    for j, d in enumerate(design):
        #Create arrays for summaries
        num_batches = 40 // d['batch_size']
        lengthscales = np.zeros([num_repeats, num_batches-1, num_components, 2])
        log_likelihoods = np.zeros([num_repeats, num_batches-1, 2])
        loo_cv_errors = np.zeros([num_repeats, num_batches-1, 2])
        
        for i in range(num_repeats):
            res =  descriptors_optimization(batch_size=d['batch_size'],
                                            num_batches=num_batches,
                                            num_components = num_components,
                                            random_seed=int(time.time()),
                                            normalize_inputs=d['normalize_inputs'],
                                            normalize_outputs=d['normalize_outputs'],
                                            save_to_disk=False)
            lengthscales[i, :, :, :] = res.lengthscales
            log_likelihoods[i, :, :] = res.log_likelihoods
            loo_cv_errors[i, :, :] = res.loo_cv_errors
        
            output_prefix=f'design_{j}'
            res.experiments.to_csv(f'outputs/{output_prefix}_in_silico_experiments_iteration_{i}.csv')
            np.save(f'outputs/{output_prefix}_in_silico_lengthscales', lengthscales)
            np.save(f'outputs/{output_prefix}_in_silico_log_likelihoods', log_likelihoods)
            np.save(f'outputs/{output_prefix}_in_silico_loo_errors', loo_cv_errors)
            to_print = [("Number of principal components", num_components),
                        ("Number of batches", num_batches),
                        ("Batch size",d['batch_size']),
                        ('Normalize inputs', d['normalize_inputs']),
                        ('Normalize outputs', d['normalize_outputs']),
                        ('Iteration', i)]
        
            with open(f'outputs/{output_prefix}_in_silico_metadata.txt',  'w') as f:
                for prefix, value in to_print:
                    txt = f"{prefix}: {value}"
                    f.write(txt)


class SolventEvolutionaryOptimization:
    def __init__(self, batch_size, num_batches, seed, solvent_ds=None):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.solvent_ds = solvent_ds if solvent_ds is not None else create_pcs_ds(3)
        self.cas_numbers = self.solvent_ds.index.values
        self._prng = Random(seed)
        self._bounder = inspyred.ec.DiscreteBounder(values=[i for i in range(len(self.cas_numbers))])
        self._domain = create_domain(self.solvent_ds)
        initial_experiments = generate_initial_experiment_data(self._domain,
                                                                    self.solvent_ds,
                                                                    self.batch_size,
                                                                    np.random.RandomState(seed))
        self.initial_experiments = initial_experiments.data_to_numpy()[:, (0,1)].astype(np.float64).tolist()
        self.solvents_evaluated = []


    def _generator(self, random, args):
        return [random.randint(1, len(self.cas_numbers))]

    def _evaluator(self, candidates, args):
        fitness = []
        for index in candidates:
            cas = self.cas_numbers[index[0]]
            conversion, de = experiment(cas, self.solvent_ds)
            fitness.append(inspyred.ec.emo.Pareto([conversion, de]))
            self.solvents_evaluated.append(cas)
        return fitness
    
    def optimize(self):
        ea = inspyred.ec.emo.NSGA2(self._prng)  
        ea.termination = inspyred.ec.terminators.generation_termination
        ea.variator = [inspyred.ec.variators.blend_crossover, 
                    inspyred.ec.variators.gaussian_mutation]
        self._res = ea.evolve(generator=self._generator,
                              evaluator=self._evaluator,
                              pop_size=self.batch_size,
                              max_generations=self.num_batches,
                              maximize=True,
                            #   seeds=self.initial_experiments,
                              bounder=self._bounder,
                              observer=inspyred.ec.observers.population_observer)
        conversion = [f.fitness[0] for f in ea.archive]
        de  =[f.fitness[1] for f in ea.archive]
        return np.array([conversion, de]).T

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

if __name__ == '__main__':
    # descriptors_optimization()
    repeat_test(100)