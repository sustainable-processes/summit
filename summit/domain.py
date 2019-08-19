from summit.data import DataSet
import numpy as np
import pandas as pd
from typing import List, Optional, Type, Dict
from abc import ABC, abstractmethod
import json

class Variable(ABC):
    """A base class for variables
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable
    is_objective: bool, optional
        If True, this variable is an output. Defaults to False (i.e., an input variable)
    maximize: bool, optional
        If True, the output will be maximized; if False, it will be minimized. 
        Defaults to True. 
    units: str, optional
        Units of the variable. Defaults to None.

    Attributes
    ---------
    name
    description
    """
    def __init__(self, name: str, description: str, variable_type: str, **kwargs):
        Variable._check_name(name)
        self._name = name
        self._description = description
        self._variable_type = variable_type
        self._is_objective = kwargs.get('is_objective', False)
        self._maximize = kwargs.get('maximize', True)
        self._units = kwargs.get('units', None)

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

    @property
    def variable_type(self) -> str:
        return self._variable_type

    @property
    def maximize(self) -> bool:
        return self._maximize
        
    @property
    def is_objective(self) -> bool:
        return self._is_objective

    @property
    def units(self) -> str:
        return self._units

    def to_dict(self):
        variable_dict = {
           'type': self._variable_type,
           'is_objective': self._is_objective,
           'name': self.name,
           'description': self.description,
           'units': self.units}
        return variable_dict

    @staticmethod
    @abstractmethod
    def from_dict():
        raise NotImplementedError('Must be implemented by subclasses of Variable')

    @staticmethod
    def _check_name(name: str):
        if type(name) != str:
            raise ValueError(f"""{name} is not a string. Variable names must be strings.""")

        test_name = name
        if name != test_name.replace(" ", ""):
            raise ValueError(f"""Error with variable name "{name}". Variable names cannot have spaces. Try replacing spaces with _ or -""")

    def __repr__(self):
        return f"Variable(name={self.name}, description={self.description})"

    @abstractmethod
    def _html_table_rows(self):
        pass

    def _make_html_table_rows(self, value):
        name_column = f"<td>{self.name}</td>"
        input_output = 'output' if self.is_objective else 'input'
        if self.is_objective:
            direction = 'maximize' if self.maximize else 'minimize'
            input_output = f'{direction} objective'
        else:
            input_output = 'input'
        type_column = f"<td>{self.variable_type}, {input_output}</td>"
        description_column = f"<td>{self.description}</td>"
        values_column = f"<td>{value}</td>"
        return f"<tr>{name_column}{type_column}{description_column}{values_column}</tr>"

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
    is_objective: bool, optional
        If True, this variable is an output. Defaults to False (i.e., an input variable)

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

    def __init__(self, name: str, description: str, bounds: list, **kwargs):
        Variable.__init__(self, name, description, 'continuous', **kwargs)
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

    def _html_table_rows(self):
        return self._make_html_table_rows(f"[{self.lower_bound},{self.upper_bound}]")
    
    def to_dict(self):
        variable_dict = super().to_dict()
        variable_dict.update({'bounds': [float(self.lower_bound), float(self.upper_bound)]})
        return variable_dict

    @staticmethod
    def from_dict(variable_dict):
        return ContinuousVariable(name= variable_dict['name'],
                           description=variable_dict['description'],
                           bounds=variable_dict['bounds'],
                           is_objective=variable_dict['is_objective'])

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return [self, other]

    
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
    is_objective: bool, optional
        If True, this variable is an output. Defaults to False (i.e., an input variable)

    Attributes
    ---------
    name
    description
    levels

    Examples
    --------
    >>> reactant = DiscreteVariable('reactant', 'aromatic reactant', ['benzene', 'toluene'])

    """
    def __init__(self, name, description, levels, **kwargs):
        Variable.__init__(self, name, description, 'discrete', **kwargs)
        
        #check that levels are unique
        if len(list({v for v in levels})) != len(levels):
            raise ValueError("Levels must have unique values.")
        self._levels = levels

    @property
    def levels(self) -> np.ndarray:
        """`numpy.ndarray`: Potential values of the discrete variable"""
        levels = np.array(self._levels)
        return np.atleast_2d(levels).T

    @property
    def num_levels(self) -> int:
        return len(self._levels)

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
    
    def to_dict(self):
        """ Return json encoding of the variable"""
        variable_dict = super().to_dict()
        variable_dict.update({'levels': self.levels.tolist()})
        return variable_dict

    @staticmethod
    def from_dict(variable_dict):
        return DiscreteVariable(name=variable_dict['name'],
                                description=variable_dict['description'],
                                levels=variable_dict['levels'],
                                is_objective=variable_dict['is_objective'])

    def _html_table_rows(self):
        return self._make_html_table_rows(f"{self.num_levels} levels")

class DescriptorsVariable(Variable):
    """Representation of a set of descriptors
    
    Parameters
    ----------
    name: str
        The name of the variable
    description: str
        A short description of the variable 
    ds: DataSet
        A dataset object
    is_objective: bool, optional
        If True, this variable is an output. Defaults to False (i.e., an input variable)

    Attributes
    ---------
    name
    description
    df
    num_descriptors
    num_examples

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
                 ds: DataSet, **kwargs):
        Variable.__init__(self, name, description, 'descriptors', **kwargs)
        self.ds = ds

    @property
    def num_descriptors(self) -> int:
        return len(self.ds.data_columns)

    @property
    def num_examples(self):
        return self.ds.shape[0]

    def to_dict(self):
        """ Return json encoding of the variable"""
        variable_dict = super().to_dict()
        variable_dict.update({'ds': self.ds.to_dict()})
        return variable_dict
    
    @staticmethod
    def from_dict(variable_dict):
        ds = DataSet(variable_dict['ds'])
        ds.columns.names = ['NAME', 'TYPE']
        return DescriptorsVariable(name=variable_dict['name'],
                                   description=variable_dict['description'],
                                   ds=ds,
                                   is_objective=variable_dict['is_objective'])

    def _html_table_rows(self):
        return self._make_html_table_rows(f"{self.num_examples} examples of {self.num_descriptors} descriptors")


class Constraint: 
    def __init__(self, expression):
        self._expression = expression

    @property
    def expression(self):
        return self._expression

    def _html_table_rows(self):
        columns = []
        columns.append("") #name column
        columns.append("constraint") #type column
        columns.append(self.expression) #description columns
        columns.append("") #value column
        return ''.join([f"<td>{column}</td>" for column in columns])

    
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
    def __init__(self, variables=[], constraints=[]):
        self._variables = variables
        self._constraints = constraints
        #Check that all the output variables continuous
        self.raise_noncontinuous_outputs()

    @property
    def variables(self):
        """[List[Type[Variable]]]: List of variables in the domain"""
        return self._variables

    @property
    def constraints(self):
        return self._constraints

    # def __getitem__(self, key):
    #     '''For accessing variables like a dictionary'''
    #     for v in self.variables:
    #         if v.name == key:
    #             return v    
    #     raise KeyError(f'No variable {key} found in the domain.')
    
    # def _ipython_key_completions_(self):
    #     return [v.name for v in self.variables]

    @property
    def input_variables(self):
        input_variables = []
        for v in self.variables:
            if v.is_objective:
                pass
            else:
                input_variables.append(v)
        return input_variables

    @property
    def output_variables(self):
        output_variables = []
        for v in self.variables:
            if v.is_objective:
                output_variables.append(v)
            else:
                pass
        return output_variables

    def raise_noncontinuous_outputs(self):
        '''Raise an error if the outputs are not continuous variables'''
        for v in self.output_variables:
            if v.variable_type != 'continuous':
                raise DomainError("All output variables must be continuous")

    def num_variables(self, include_outputs=False) -> int:
        ''' Number of variables in the domain 
        
        Parameters
        ---------- 
        include_outputs: bool, optional
            If True include output variables in the count.
            Defaults to False.
        
        Returns
        -------
        num_variables: int
            Number of variables in the domain
        ''' 
        k=0
        for v in self.variables:
            if v.is_objective and not include_outputs:
                continue
            k+=1
        return k

    def num_discrete_variables(self, include_outputs=False) -> int:
        ''' Number of discrete varibles in the domain 
        
        Parameters
        ---------- 
        include_outputs: bool, optional
            If True include output variables in the count.
            Defaults to False.
        
        Returns
        -------
        num_variables: int
            Number of discrete variables in the domain
        '''
        k=0
        for v in self._variables:
            if v.is_objective and not include_outputs:
                continue
            elif v.variable_type == 'discrete':
                k+= 1
        return k

    def num_continuous_dimensions(self, include_outputs=False) -> int:
        '''The number of continuous dimensions
        
        This includes dimensions of descriptors variables
        
        Parameters
        ---------- 
        include_outputs: bool, optional
            If True include output variables in the count.
            Defaults to False.
        
        Returns
        -------
        num_variables: int
            Number of variables in the domain
        '''
        k = 0
        for v in self._variables:
            if v.is_objective and not include_outputs:
                continue
            if v.variable_type == 'continuous':
                k+=1
            if v.variable_type == 'descriptors':
                k+= v.num_descriptors
        return k

    def to_dict(self):
        """Return a dictionary representation of the domain"""
        return [variable.to_dict() for variable in self.variables]

    def to_json(self):
        """Return the a json representation of the domain"""
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(domain_dict):
        variables = []
        for variable in domain_dict:
            if variable['type'] == "continuous":
                new_variable = ContinuousVariable.from_dict(variable)
            elif variable['type'] == "discrete":
                new_variable = DiscreteVariable.from_dict(variable)
            elif variable['type'] == 'descriptors':
                new_variable =  DescriptorsVariable.from_dict(variable)
            else:
                raise ValueError(f"Cannot load variable of type:{variable['type']}. Variable should be continuous, discrete or descriptors")
            variables.append(new_variable)
        return Domain(variables)


    def __add__(self, obj):
        #TODO: make this work with adding arrays of variable or constraints
        if isinstance(obj, Variable):
            if obj.is_objective and obj.variable_type != 'continuous':
                raise DomainError("Output variables must be continuous")
            return Domain(variables=self._variables + [obj], constraints=self.constraints) 
        elif isinstance(obj, Constraint):
            return Domain(variables=self.variables, constraints = self.constraints + [obj])
        else:
            raise RuntimeError('Not a supported domain object.')
    
    def _repr_html_(self):
        """Build html string for table display in jupyter notebooks.
        
        Notes
        -----
        Adapted from https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/domain.py
        """
        html = ["<table id='domain' width=100%>"]

        # Table header
        columns = ['Name', 'Type', 'Description', 'Values']
        header = "<tr>"
        header += ''.join(map(lambda l: "<td><b>{0}</b></td>".format(l), columns))
        header += "</tr>"
        html.append(header)

        # Add parameters
        html.append(self._html_table_rows())
        html.append("</table>")

        return ''.join(html)

    def _html_table_rows(self):
        variables = ''.join([v._html_table_rows() for v in self.variables])
        constraints = ''.join([c._html_table_rows() for c in self.constraints])
        return variables + constraints


class DomainError(Exception):
    pass