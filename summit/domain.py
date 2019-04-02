import numpy as np
import pandas as pd
from typing import List, Optional

class Variable:
    """A base class for variables
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable

    Attributes
    ---------
    name
    description
    """
    def __init__(self, name: str, description: str):
        Variable._check_name(name)
        self._name = name
        self._description = description

    @property
    def name(self)-> str:
        """str: name of the variable"""
        return self._name

    @name.setter
    def name(self, value: str):
        Variable._check_name(value)
        self._name = value

    @property
    def description(self) -> str:
        """str: description of the variable"""
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value
    
    @staticmethod
    def _check_name(name: str):
        if type(name) != str:
            raise ValueError(f"""{name} is not a string. Variable names must be strings.""")

        test_name = name
        if name != test_name.replace(" ", ""):
            raise ValueError(f"""Error with variable name "{name}". Variable names cannot have spaces. Try replacing spaces with _ or -""")

    def __repr__(self):
        return f"Variable(name={self.name}, description={self.description})"


class ContinuousVariable(Variable):
    """Representation of a continuous variable
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable 
    bounds: list of float or int
        The lower and upper bounds (respectively) of the variable

    Attributes
    ---------
    name
    description
    bounds
    lower_bound
    upper_bound

    Examples
    --------
    >>> var = ContinuousVariable('temperature', 'reaction temperature', [1, 100])

    """

    def __init__(self, name: str, description: str, bounds: list):
        Variable.__init__(self, name, description)
        self._lower_bound = bounds[0]
        self._upper_bound = bounds[1]

    @property
    def bounds(self):
        """`numpy.ndarray`: Lower and upper bound of the variable"""
        return np.array([self.lower_bound, self.upper_bound])

    @property
    def lower_bound(self):
        """float or int: lower bound of the variable"""
        return self._lower_bound

    @property
    def upper_bound(self):
        """float or int: upper bound of the variable"""
        return self._upper_bound

    
class DiscreteVariable(Variable):
    """Representation of a discrete variable
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable 
    levels: list of any serializable object
        The potential values of the discrete variable

    Attributes
    ---------
    name
    description
    levels

    Examples
    --------
    >>> reactant = DiscreteVariable('reactant', 'aromatic reactant', ['benzene', 'toluene'])

    """
    def __init__(self, name, description, levels):
        Variable.__init__(self, name, description)
        
        #check that levels are unique
        if len(list({v for v in levels})) != len(levels):
            raise ValueError("Levels must have unique values.")
        self._levels = levels

    @property
    def levels(self) -> str:
        """`numpy.ndarray`: Potential values of the discrete variable"""
        return np.array(self._levels)

    def add_level(self, level):
        """ Add a level to the discrete variable

        Parameters
        ---------
        level
            Value to add to the levels of the discrete variable

        Raises
        ------
        ValueError
            If the level is already in the list of levels
        """
        if level in self._levels:
            raise ValueError("Levels must have unique values.")
        self._levels.append(level)

    def remove_level(self, level):
        """ Add a level to the discrete variable

        Parameters
        ---------
        level
            Level to remove from the discrete variable

        Raises
        ------
        ValueError
            If the level does not exist for the discrete variable
        """
        try:
            remove_index = self._levels.index(level)
        except ValueError:
            raise ValueError(f"Level {level} is not in the list of levels.")


class DescriptorsVariable(Variable):
    """Representation of a set of descriptors
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable 
    df: pandas.DataFrame
        A pandas dataframe with the values of descriptors
    select_subset: numpy.ndarray, optional
        A subset of index values in the df that should be selected from in all designs and optimizations

    Attributes
    ---------
    name
    description
    df
    select_subset

    Notes
    -----
    In other places, this type of variable is called a bandit variable. 

    Examples
    --------
    >>> solvent_df = pd.DataFrame([[5, 81],[-93, 111]], 
                                  index=['benzene', 'toluene'],
                                  columns=['melting_point', 'boiling_point'])
    >>> solvent = DescriptorsVariable('solvent', 'solvent descriptors', solvent_df)
    """    
    def __init__(self, name: str, description: str, 
                 df: pd.DataFrame, 
                 select_subset: np.ndarray= None):
        Variable.__init__(self, name, description)
        self.df = df
        self.select_subset = select_subset

class Domain:
    """Representation of the optimization domain

    Parameters
    ---------
    variables: list of `Variable` like objects, optional
        list of variable objects (i.e., `ContinuousVariable`, `DiscreteVariable`, `DescriptorsVariable`)

    Attributes
    ----------
    variables

    Examples
    --------
    >>> domain = Domain()
    >>> domain += ContinuousVariable('temperature', 'reaction temperature', [1, 100])

    """
    def __init__(self, variables:Optional[List[Type[Variable]]]=[]):
        self._variables = variables

    @property
    def variables(self):
        """[List[Type[Variable]]]: List of variables in the domain"""
        return self._variables
    
    def __add__(self, var):
        # assert type(var) == Variable
        self._variables.append(var)
        return self
