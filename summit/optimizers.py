from typing import List
imort numpy as np.

class Optimizer(ABC):

    @abstractmethod
    def optimize(self):
        pass


class NSGAII(Optimizer):
    def __init__(self, objective, 
                 max_generations=100, 
                 crossover_rate=1, 
                 mutation_rate=0.1):
        super().__init__(objective)
        self.max_generations = max_generations
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate

    def optimize(self):
        pass

class EnumerationParetoOptimizer(Optimizer):
    def optimize(objectives: List, enumeration: np.ndarray):
        k, D = enumeration.shape #Number of examples in enumeration
        sample_pareto = np.zeros([k, Opt.NumGPs])
        sample_nadir = np.zeros(len(objectives))

        if D ==1:
            for i in range(k):
                x = np.array([[enumeration[i, 0]]])
                sample_pareto[i, :] = pareto_objective(x, objectives, [1,Opt.NumGPs])
        else:
            for i in range(k):
                sample_pareto[i, :] = pareto_objective(enumeration[i, :], objectives, [Opt.GPs[0].n_spectral_points, D])
                sample_nadir[i] = np.max(sample_pareto[:, i])

        for i in range(Opt.NumGPs):
            sample_nadir[i] = np.max(sample_pareto[:, i])
            
        return sample_pareto, sample_nadir

                           