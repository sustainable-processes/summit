from summit.data import DataSet
from summit.domain import Domain, DomainError
from summit.acquisition import HvI
from summit.optimizers import NSGAII

import GPy
import numpy as np

import warnings
import logging

class Strategy:
    def __init__(self, domain:Domain):
        self.domain = domain

    def get_inputs_outputs(self, ds: DataSet, copy=True):
        data_columns = ds.data_columns
        new_ds = ds.copy() if copy else ds

        #Determine input and output columns in dataset
        input_columns = []
        output_columns = []
        for variable in self.domain.variables:
            check_input = variable.name in data_columns and not variable.is_objective
                          
            if check_input and variable.variable_type != 'descriptors':
                input_columns.append(variable.name)
            elif check_input and variable.variable_type == 'descriptors':
                #Add descriptors to the dataset
                indices = new_ds[variable.name].values
                descriptors = variable.ds.loc[indices]
                new_metadata_name = descriptors.index.name
                descriptors.index = new_ds.index
                new_ds = new_ds.join(descriptors, how='inner')
                
                #Make the original descriptors column a metadata column
                column_list_1 = new_ds.columns.levels[0].to_list()
                ix = column_list_1.index(variable.name)
                column_list_1[ix] = new_metadata_name
                new_ds.columns.set_levels(column_list_1, level=0, inplace=True)
                column_codes_2 = list(new_ds.columns.codes[1])
                ix_code = np.where(new_ds.columns.codes[0]==ix)[0][0]
                column_codes_2[ix_code] = 1
                new_ds.columns.set_codes(column_codes_2, level=1, inplace=True)

                #add descriptors data columns to inputs
                input_columns += descriptors.data_columns
            elif variable.name in data_columns and variable.is_objective:
                if variable.variable_type == 'descriptors':
                    raise DomainError("Output variables cannot be descriptors variables.")
                output_columns.append(variable.name)
            else:
                raise DomainError(f"Variable {variable.name} is not in the dataset.")

        if output_columns is None:
            raise DomainError("No output columns in the domain.  Add at least one output column for optimization.")

        #Return the inputs and outputs as separate datasets
        return new_ds[input_columns].copy(), new_ds[output_columns].copy()
        
class TSEMO2(Strategy):
    ''' A modified version of Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO)
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the optimization
    models: summit.models.Model
        Any list of surrogate models to be used in the optimization
    maximize: bool, optional
        Whether optimization should be treated as a maximization or minimization problem.
        Defaults to maximization. 
    acquisition: summit.acquistion.Acquisition, optional
        The acquisition function used to select the next set of points from the pareto front
        (see optimizer).  Defaults to hypervolume improvement with the reference point set 
        as the upper bounds of the outputs in the specified domain and random rate 0.0
    optimizer: summit.optimizers.Optimizer, optional
        The internal optimizer for estimating the pareto front prior to maximization
        of the acquisition function. By default, NSGAII will be used if there is a combination
        of continuous, discrete and/or descriptors variables. If there is a single descriptors 
        variable, then all of the potential values of the descriptors will be evaluated.
    
    
    Examples
    --------
    domain += DescriptorsVariable('solvent',
                                  'solvents in the lab',
                                   solvent_ds)
    domain+= ContinuousVariable(name='yield',
                                description='relative conversion to triphenylphosphine oxide determined by LCMS',
                                bounds=[0, 100],
                                is_objective=True)
    domain += ContinuousVariable(name='de',
                                description='diastereomeric excess determined by ratio of LCMS peaks',
                                bounds=[0, 100],
                                is_objective=True)
    input_dim = domain.num_continuous_dimensions()+domain.num_discrete_variables()
    kernels = [GPy.kern.Matern52(input_dim = input_dim, ARD=True)
           for _ in range(2)]
    models = [GPyModel(kernel=kernels[i]) for i in range(2)]
    acquisition = HvI(reference=[100, 100], random_rate=0.25)
    tsemo = TSEMO(domain, models, acquisition=acquisition)
    previous_results = DataSet.read_csv('results.csv')
    design = tsemo.generate_experiments(previous_results, batch_size, 
                                        normalize_inputs=True)
 
    ''' 
    def __init__(self, domain, models, acquisition=None, optimizer=None):
        Strategy.__init__(self, domain)
        self.models = models
        if acquisition is None:
            reference = [v.upper_bound for v in self.domain.output_variables]
            self.acquisition = HvI(reference, random_rate=0.0)   
        else:
            self.acquisition = acquisition
        if not optimizer:
            self.optimizer = NSGAII(self.domain)
        else:
            self.optimizer = optimizer

    def generate_experiments(self, previous_results: DataSet, num_experiments, 
                             normalize_inputs=False, no_repeats=True, maximize=True):
        #Get inputs and outputs + standardize if needed
        inputs, outputs = self.get_inputs_outputs(previous_results)
        if normalize_inputs:
            self.x = inputs.standardize() #TODO: get this to work for discrete variables
        else:
            self.x = inputs.data_to_numpy()

        self.y = outputs.data_to_numpy()

        #Update surrogate models with new data
        for i, model in enumerate(self.models):
            Y = self.y[:, i]
            Y = np.atleast_2d(Y).T
            logging.debug(f'Fitting model {i+1}')
            model.fit(self.x, Y, num_restarts=3, max_iters=100,parallel=True)

        logging.debug("Running internal optimization")

        #If the domain consists of one descriptors variables, evaluate every candidate
        check_descriptors = [True if v.variable_type =='descriptors' else False 
                             for v in self.domain.input_variables]
        if all(check_descriptors) and len(check_descriptors)==1:
            descriptor_arr = self.domain.variables[0].ds.data_to_numpy()
            if no_repeats:
                points = self._mask_previous_points(self.x, descriptor_arr)
            else:
                points = self.x
            predictions = np.zeros([points.shape[0], len(self.models)])
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(points)[0][:,0]
            
            self.acquisition.data = self.y
            self.acq_vals, indices = self.acquisition.select_max(predictions, num_evals=num_experiments)
            indices = [np.where((descriptor_arr == points[ix]).all(axis=1))[0][0]
                       for ix in indices]
            result = self.domain.variables[0].ds.iloc[indices, :]
        #Else use modified nsgaII
        else:
            def problem(x):
                x = np.array(x)
                x = np.atleast_2d(x)
                y = [model.predict(x)
                     for model in self.models]
                y = np.array([yo[0,0] for yo in y])
                return y
            int_result = self.optimizer.optimize(problem)
            self.acquisition.data = self.y
            self.acq_vals, indices = self.acquisition.select_max(int_result.fun, 
                                                                 num_evals=num_experiments)
            result = int_result.x[indices, :]
        return result

    def _mask_previous_points(self, x, descriptor_arr):
        descriptor_mask = np.ones(descriptor_arr.shape[0], dtype=bool)
        for point in x:
            try: 
                index = np.where(descriptor_arr==point)[0][0]
                descriptor_mask[index] = False
            except IndexError:
                continue
        return descriptor_arr[descriptor_mask, :]

        