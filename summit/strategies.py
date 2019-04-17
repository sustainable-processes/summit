from summit.data import DataSet
from summit.domain import Domain, DomainError

import GPy
import numpy as np

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
                new_ds = new_ds.join(descriptors)

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
        return new_ds[input_columns].copy(), ds[output_columns].copy()
        

class TSEMO(Strategy):
    def __init__(self, domain, models, objective=None, optimizer=None, acquisition=None):
        super().__init__(domain)
        self.models = models
        self.objective = objective
        self.optimizer = optimizer
        self.acquisition = acquisition

    def generate_experiments(self, previous_results: DataSet, num_experiments, normalize=False):
        #Get inputs and outputs + standardize if needed
        inputs, outputs = self.get_inputs_outputs(previous_results)
        if normalize:
            self.x = inputs.standardize()
            self.y = outputs.standardize()
        else:
            self.x = inputs.data_to_numpy()
            self.y = outputs.data_to_numpy()

        #Update models
        for i, model in enumerate(self.models):
            Y = self.y[:, i]
            Y = np.atleast_2d(Y).T
            model.fit(self.x, Y)

        # sample_funcs = [model.posterior_sample() for model in self.models]

        # pareto_sample = self.optimizer.optimizer(sample_funcs)

        # acquisition_values = np.zeros([len(pareto_sample), 1])
        # for i, sample in enumerate(pareto_sample):
        #     acquisition_values[i, 0] = self.acquisition.evaluate(sample)

        # acquisition_values_sorted = np.sort(acquisition_values)
        
        #Construct an experimental design somehow 
        # design = Design()
        # for variable in domain.variables:
        # ....add variables with correct values to design
        # return design


