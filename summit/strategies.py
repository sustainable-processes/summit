from summit.data import DataSet
from summit.domain import Domain, DomainError
from summit.objective import HV

import GPy
import numpy as np

import warnings

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
    def __init__(self, domain, models, objective=None, optimizer=None, acquisition=None):
        #TODO: check that the domain is only a descriptors variable
        super().__init__(domain)
        self.models = models
        self.objective = objective if objective else HV()
        self.optimizer = optimizer
        self.acquisition = acquisition

    def generate_experiments(self, previous_results: DataSet, num_experiments,
                             normalize_inputs=False, normalize_outputs=False, 
                             num_spectral_samples=4000):
        #Get inputs and outputs + standardize if needed
        inputs, outputs = self.get_inputs_outputs(previous_results)
        if normalize_inputs:
            self.x = inputs.standardize()
            descriptor_arr = self.domain.variables[0].ds.standardize()
            descriptor_arr = descriptor_arr.astype(np.float64)
        else:
            self.x = inputs.data_to_numpy()
            descriptor_arr = self.domain.variables[0].ds.data_to_numpy()
            descriptor_arr = descriptor_arr.astype(np.float64)

        if normalize_outputs:
            self.y = outputs.data_to_numpy()
        else:
            
            self.y = outputs.data_to_numpy()


        #Remove any points from the descriptors matrix that have already been suggested
        descriptor_mask = np.ones(descriptor_arr.shape[0], dtype=bool)
        for point in self.x:
            try: 
                index = np.where(descriptor_arr==point)[0][0]
                descriptor_mask[index] = False
            except IndexError:
                continue
        masked_descriptor_arr = descriptor_arr[descriptor_mask, :]

        #Update models and take samples
        samples_nadir = np.zeros(2)
        new_samples = np.zeros([masked_descriptor_arr.shape[0], 2])
        for i, model in enumerate(self.models):
            Y = self.y[:, i]
            Y = np.atleast_2d(Y).T
            model.fit(self.x, Y)
            samples = model._model.posterior_samples_f(masked_descriptor_arr, size=1)
            new_samples[:, i] = samples[:,0,0]
            samples_nadir[i] = np.max(samples)

        hv_imp, indices = hypervolume_improvement_index(self.y, samples_nadir, new_samples, 
                                                        batchsize=num_experiments)

        indices = [np.where((descriptor_arr == masked_descriptor_arr[ix]).all(axis=1))[0][0]
                   for ix in indices]

        return self.domain.variables[0].ds.iloc[indices, :]
        
def hypervolume_improvement_index(Ynew, samples_nadir, samples, batchsize):
    '''Returns the point(s) that maximimize hypervolume improvement '''
    #Get the reference point, r
    r = samples_nadir + 0.01*(np.max(samples)-np.min(samples)) 
    index = []
    mask = np.ones(samples.shape[0], dtype=bool)
    # Number of samples to consider
    k, _ = np.shape(samples)
    num_gps = Ynew.shape[1]

    for i in range(batchsize):
        masked_samples = samples[mask, :]
        Yfront, _ = _pareto_front(Ynew)
        hv_improvement = []
        if num_gps == 2:
            hvY = hypervolume_2D(Yfront, r)
            #Determine hypervolume improvement by including
            # each point from samples (masking previously selected poonts)
            for sample in masked_samples:
                sample = sample.reshape(1,num_gps)
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
        max_hvi_index =  hv_improvement.index(max(hv_improvement))
        samples_index = np.where((samples == masked_samples[max_hvi_index, :]).all(axis=1))[0][0]
        max_improvement_point = samples[samples_index, :].reshape(1, num_gps)
        Ynew = np.append(Ynew, max_improvement_point, axis=0)
        mask[samples_index] = False
        index.append(samples_index)

    if len(hv_improvement)==0:
        hv_imp = []
    else:
        hv_imp = hv_improvement[index[-1]] + hvY-hvY0
    return hv_imp, index

def _pareto_front(points):
    '''Calculate the pareto front of a 2 dimensional set'''
    try:
        assert points.all() == np.atleast_2d(points).all()
        assert points.shape[1] == 2
    except AssertionError:
        raise ValueError("Points must be 2 dimensional.")

    sorted_indices = np.argsort(points[:, 0])
    sorted = points[sorted_indices, :]
    front = np.atleast_2d(sorted[-1, :])
    front_indices = sorted_indices[-1]
    for i in range(2, sorted.shape[0]+1):
        if np.greater(sorted[-i, 1], front[:, 1]).all():
            front = np.append(front, 
                              np.atleast_2d(sorted[-i, :]),
                              axis=0)
            front_indices = np.append(front_indices,
                                      sorted_indices[-i])
    return front, front_indices


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