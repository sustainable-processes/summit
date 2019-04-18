from summit.data import DataSet
from summit.domain import Domain, DomainError
from summit.objective import HV

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
        self.objective = objective if objective else HV()
        self.optimizer = optimizer
        self.acquisition = acquisition

    def generate_experiments(self, previous_results: DataSet, num_experiments,
                             normalize=False, num_spectral_samples=4000):
        #Get inputs and outputs + standardize if needed
        inputs, outputs = self.get_inputs_outputs(previous_results)
        if normalize:
            self.x = inputs.standardize()
            self.y = outputs.standardize()
        else:
            self.x = inputs.data_to_numpy()
            self.y = outputs.data_to_numpy()

        #Update models
        samples_nadir = np.zeros(2)
        sample_pareto = np.zeros([self.domain.variables[0].num_examples, 2])
        for i, model in enumerate(self.models):
            Y = self.y[:, i]
            Y = np.atleast_2d(Y).T
            model.fit(self.x, Y)
            samples = model._model.posterior_samples_f(self.domain.variables[0].ds.data_to_numpy(), size=1)
            sample_pareto[:, i] = samples[:,0,0]
            samples_nadir[i] = np.max(samples)

        hv_imp, indices = hypervolume_improvement_index(self.y, samples_nadir, sample_pareto, 
                                                      batchsize=num_experiments)

        return self.domain.variables[0].ds.iloc[indices, :]
        
def hypervolume_improvement_index(Ynew, sample_nadir, sample_pareto, batchsize):
    '''Returns the hypervolume improvment and index (in sample_pareto) of points selected from sample_pareto front '''
    #Get the reference point, r
    r = sample_nadir + 0.01*(np.max(sample_pareto)-np.min(sample_pareto)) 
    index = []
    # Number of samples to consider
    k, _ = np.shape(sample_pareto)
    num_gps = Ynew.shape[1]
    hvY0 = 0
    for i in range(batchsize):
        Yfront, _ = _pareto_front(Ynew)
        hv_improvement = []
        if num_gps == 2:
            hvY = hypervolume_2D(Yfront, r)
            #Determine hypervolume immprovement by including
            #piont j from sample_pareto
            for j in range(k):
                sample = sample_pareto[j, :].reshape(1,num_gps)
                A = np.append(Ynew, sample, axis=0)
                Afront, _ = _pareto_front(A)
                hv = hypervolume_2D(Afront, r)
                hv_improvement.append(hv-hvY)
        elif  num_gps == 3:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
        hvY0 = hvY if i==0 else hvY0
        #Choose the point that maximizes hypervolume improvement
        current_index =  hv_improvement.index(max(hv_improvement))
        max_improvement_point = sample_pareto[current_index, :].reshape(1, num_gps)
        Ynew = np.append(Ynew, max_improvement_point, axis=0)
        index.append(current_index)
        
    hv_imp = hv_improvement[index[-1]] + hvY-hvY0
    return hv_imp, index

def _pareto_front(B):
    pareto_front = np.zeros(np.shape(B))
    pareto_front_indices = []
    _, num_funcs = np.shape(B)
    j=0
    for i, exmpl in enumerate(B): 
        #If this example is smaller on any dimension 
        #than other examples, add it to the pareto front
        exmpl.shape = (1, num_funcs)
        diff = np.delete(B, i, 0) - exmpl
        if np.any(diff)>0:
            pareto_front[j, :] = exmpl
            j += 1
            pareto_front_indices.append(i)
    pareto_front = pareto_front[0:j, :]
    pareto_front.shape = (j, num_funcs)
    return pareto_front, pareto_front_indices

def _hypervolume_2D(F, ub):
    F = -F.transpose() + np.ones(np.shape(F.transpose()))
    L = sortrows(F.transpose(), 0).transpose()
    n, l = np.shape(L)
    ub = ub + np.ones(n)
    hypervolume = 0
    for i in range(l):
        hypervolume += ((L[0, i]-ub[0])*(L[1,i]-ub[1]))
        ub[1] = L[1, i]
    return hypervolume

def hypervolume_2D(Yfront, r):
    r = r[:, np.newaxis]
    AYfront = remove_points_above_reference(Yfront, r)
    if len(AYfront)>0:
        normvec = np.min(AYfront, axis=0)[:, np.newaxis]
        z, _ = np.shape(AYfront)
        A = AYfront-normvec.transpose()
        A = A@np.diagflat(np.divide(1,r-normvec))
        A = -A + np.ones(np.shape(A))
        A = sortrows(A, 0)
        hyp_percentage = _hypervolume_2D(A, [0, 0])
        hv = np.prod(r-normvec)*hyp_percentage
    else:
        hv = np.array([[]])
    return hv

def remove_points_above_reference(Afront, r):
    A = sortrows(Afront)
    for p in range(len(Afront[1, :])):
        A = A[A[:,p]<= r[p], :]
    return A

def sortrows(A, i=0):
    '''Sort rows from matrix A by column i'''
    I = np.argsort(A[:, i])
    return A[I, :]