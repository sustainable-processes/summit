import numpy as np

class Strategy:
    def __init__(self, domain):
        self.domain = domain


class TSEMO(Strategy):
    def __init__(self, domain, models, objective, optimizer, acquisition):
        super().__init__(domain)
        self.models = models
        self.objective = objective
        self.optimizer = optimizer
        self.acquisition = acquisition

    def generate_experiemnts(self, previous_results, num_experiments):
        inputs = previous_results.inputs.standardize()
        outputs = previous_results.outputs.standardize()

        for model in self.models:
            model.fit(inputs, outputs)

        sample_funcs = [model.posterior_sample() for model in self.models]

        pareto_sample = self.optimizer.optimizer(sample_funcs)

        acquisition_values = np.zeros([len(pareto_sample), 1])
        for i, sample in enumerate(pareto_sample):
            acquisition_values[i, 0] = self.acquisition.evaluate(sample)

        acquisition_values_sorted = np.sort(acquisition_values)
        
        #Construct an experimental design somehow 
        # design = Design()
        # for variable in domain.variables:
        # ....add variables with correct values to design
        # return design


