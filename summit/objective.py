
class ObjectiveSet:
    def __init__(self, *args):
        for f in args:
            assert callable(f)
        self._functions = [arg for arg in args]

    @property
    def num_objectives(self):
        return len(self_functions)
    
    def evaluate(self, X):
        outputs = self.num_objectives*[0]
        for i, f in enumerate(self._functions):
            output[i] = f(X)

            
class HV(ObjectiveSet):
    def __init__(self, *args):
        super().__init__(*args)

    def evaluate(self, X):
        func_vals = super().evaluate(X)
        