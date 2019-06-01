from summit.data import DataSet
from summit.domain import Domain, DomainError
from summit.objective import hypervolume
from summit.acquisition import HvI

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
            check_input = variable.name in data_columns and not variable.is_output 
                          
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
            elif variable.name in data_columns and variable.is_output:
                if variable.variable_type == 'descriptors':
                    raise DomainError("Output variables cannot be descriptors variables.")
                output_columns.append(variable.name)
            else:
                raise DomainError(f"Variable {variable.name} is not in the dataset.")

        if output_columns is None:
            raise DomainError("No output columns in the domain.  Add at least one output column for optimization.")

        #Return the inputs and outputs as separate datasets
        return new_ds[input_columns].copy(), new_ds[output_columns].copy()
        
class TSEMO(Strategy):
    def __init__(self, domain, models, 
                 objective=None, 
                 optimizer=None, 
                 acquisition=None):
        #TODO: check that the domain is only a descriptors variable
        super().__init__(domain)
        self.models = models
        self.optimizer = optimizer
        self.acquisition = acquisition

    def generate_experiments(self, previous_results: DataSet, num_experiments, 
                             normalize_inputs=False, no_repeats=True):
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
            # nsga = NSGAII(domain)
            # objectivefx = Objective(self.models)
            # results = nsga.optimize(objectivefx)
            # predictions = results.x
            raise NotImplementedError('When implemented, NSGAII optimizer should handle all other situations') 

        #Update models and take samples
        # samples_nadir = np.zeros(2)
        # new_samples = np.zeros([masked_descriptor_arr.shape[0], 2])
 
        logging.debug('Calculating hypervolume improvement')


        # hv_imp, indices = hypervolume_improvement_index(self.y, samples_nadir, predictions, 
        #                                                 batchsize=num_experiments, 
        #                                                 random_rate=self.random_rate)


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

        