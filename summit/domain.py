import numpy as np
from summit.utils import check

class Variable:
    def __init__(self, name, description):
        Variable._check_name(name)
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        Variable._check_name(value)
        self._name = value

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value
    
    @staticmethod
    def _check_name(name):
        if type(name) != str:
            raise ValueError(f"""{name} is not a string. Variable names must be strings.""")

        test_name = name
        if name != test_name.replace(" ", ""):
            raise ValueError(f"""Error with variable name "{name}". Variable names cannot have spaces. Try replacing spaces with _ or -""")

    def __repr__(self):
        return f"Variable(name={self.name}, description={self.description})"


class ContinuousVariable(Variable):
    def __init__(self, name, description, bounds):
        Variable.__init__(self, name, description)
        # check('Bounds', bounds, [list, tuple])
        self._lower_bound = bounds[0]
        self._upper_bound = bounds[1]

    @property
    def bounds(self):
        return np.array([self.lower_bound, self.upper_bound])

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    
class DiscreteVariable(Variable):
    def __init__(self, name, description, levels):
        Variable.__init__(self, name, description)
        # check('Levels', levels, [list, tuple])
        if len(list({v for v in levels})) != len(levels):
            raise ValueError("Levels must have unique values.")
        self._levels = levels

    @property
    def levels(self):
        return np.array(self._levels)

    def add_level(self, level):
        if level in self._levels:
            raise ValueError("Levels must have unique values.")
        self._levels.append(level)

    def remove_level(self, level):
        try:
            remove_index = self._levels.index(level)
        except ValueError:
            raise ValueError(f"Level {level} is not in the list of levels.")


class DescriptorSet(Variable):
    def __init__(self, name, description, df, select_subset=None):
        Variable.__init__(self, name, description)
        self.df = df
        self.select_subset = select_subset

class Domain:
    def __init__(self, variables:[Variable]=[]):
        self._variables = variables

    @property
    def variables(self):
        return self._variables
    
    def __add__(self, var):
        # assert type(var) == Variable
        self._variables.append(var)
        return self
