from summit.domain import Domain
from abc import ABC, abstractmethod

class Design(ABC):
    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def generate_experiments(self):
        raise NotImplementedError("Subclasses should implement this method.")


class ModifiedLatinDesign(Design):
    def __init__(self, domain: Domain):
        Design.__init__(self, domain)


    def generate_experiments(self, n_experiments, criterion='maximin'):
        
        