
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

    def generate_experiemnts(self, X, Y, num_experiments):
        pass
