from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, objective):
        self._objective = objective

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


class EnumerationOptimizer(Optimizer):
    def __init__(self, objective)
        self.objective = objective

    def optimize():
        pass

