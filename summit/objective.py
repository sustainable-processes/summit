from abc import ABC, abstractmethod

class ObjectiveSet:
    def __init__(self, functions=[],):
        for f in functions:
            assert callable(f)
        self._functions = functions

    @property
    def num_objectives(self):
        return len(self._functions)
    
    def evaluate(self, X):
        outputs = self.num_objectives*[0]
        for i, f in enumerate(self._functions):
            outputs[i] = f(X)
        return outputs
            
class HV(ObjectiveSet):
    def __init__(self, *args):
        super().__init__(*args)

    def evaluate(self, X):
        func_vals = super().evaluate(X)