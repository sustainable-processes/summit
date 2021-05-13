__all__ = [
    "Variable",
    "ContinuousVariable",
    "CategoricalVariable",
    "Constraint",
    "Domain",
    "DomainError",
]

from summit.utils.dataset import DataSet
import numpy as np
from abc import ABC, abstractmethod
import json
from copy import deepcopy


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
        self._is_objective = kwargs.get("is_objective", False)
        self._maximize = kwargs.get("maximize", True)
        self._units = kwargs.get("units", None)

    @property
    def name(self) -> str:
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
            "type": self.__class__.__name__,
            "is_objective": self._is_objective,
            "name": self.name,
            "description": self.description,
            "units": self.units,
        }
        return variable_dict

    @staticmethod
    @abstractmethod
    def from_dict():
        raise NotImplementedError("Must be implemented by subclasses of Variable")

    @staticmethod
    def _check_name(name: str):
        # Check string
        if type(name) != str:
            raise ValueError(
                f"""{name} is not a string. Variable names must be strings."""
            )

        # No spaces
        test_name = name
        if name != test_name.replace(" ", ""):
            raise ValueError(
                f"""Error with variable name "{name}". Variable names cannot have spaces. Try replacing spaces with _ or -"""
            )

        # No python keywords
        kwds = [
            "as",
            "assert",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "False",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "None",
            "nonlocal",
            "pass",
            "raise",
            "return",
            "True",
            "try",
            "while",
            "with",
            "yield",
        ]
        if name in kwds:
            raise ValueError(
                f"Variable names cannot be python keywords ({name}). For a full list of python keywords, see https://www.w3schools.com/python/python_ref_keywords.asp"
            )

    def __repr__(self):
        return f"Variable(name={self.name}, description={self.description})"

    @abstractmethod
    def _html_table_rows(self):
        pass

    def _make_html_table_rows(self, value):
        name_column = f"<td>{self.name}</td>"
        input_output = "output" if self.is_objective else "input"
        if self.is_objective:
            direction = "maximize" if self.maximize else "minimize"
            input_output = f"{direction} objective"
        else:
            input_output = "input"
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
    maximize: bool, optional
        If True, the output will be maximized; if False, it will be minimized.
        Defaults to True.

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
        Variable.__init__(self, name, description, "continuous", **kwargs)
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
        variable_dict.update(
            {"bounds": [float(self.lower_bound), float(self.upper_bound)]}
        )
        return variable_dict

    @staticmethod
    def from_dict(variable_dict):
        return ContinuousVariable(
            name=variable_dict["name"],
            description=variable_dict["description"],
            bounds=variable_dict["bounds"],
            is_objective=variable_dict["is_objective"],
        )


class CategoricalVariable(Variable):
    """Representation of a categorical variable

    Categorical variables are discrete choices that do not have an ordering.
    Common examples are selections of catalysts, bases, or ligands.

    Each possible discrete choice is referred to as a level. These are added as a list
    using the `list` keyword argument.

    When available, descriptors can be added to a categorical variable. These might be values
    such as the melting point, logP, etc. of each level of the categorical variable. These descriptors
    can significantly improve the speed of optimization and also make many more strategies compatible
    with categorical variables (i.e., all that work with continuos variables).

    Parameters
    ----------
    name : str
        The name of the variable
    description : str
        A short description of the variable
    levels : list of any serializable object, optional
        The potential values of the Categorical variable. When descriptors
        are passed, this can be left empty, and the levels will be inferred from
        the index of the descriptors DataSet.
    descriptors : :class:`~summit.utils.dataset.DataSet`, optional
        A DataSet where the keys correspond to the levels and the data
        columns are descriptors.

    Attributes
    ---------
    name
    description
    levels
    ds : descriptors DataSet

    Raises
    ------
    ValueError
        When the levels are not unique
    TypeError
        When levels is not a list

    Examples
    --------
    The simplest way to use a CategoricalVariable is without descriptors:

    >>> base = CategoricalVariable('base', 'Organic Base', levels=['DBU', 'BMTG', 'TEA'])

    When descriptors are available, they can be used directly without specfying the levels:

    >>> solvent_df = DataSet([[5, 81],[-93, 111]], index=['benzene', 'toluene'], columns=['melting_point', 'boiling_point'])
    >>> solvent = CategoricalVariable('solvent', 'solvent descriptors', descriptors=solvent_df)

    It is also possible to specify a subset of the descriptors as possible choices by passing both descriptors and levels.
    The levels must match the index of the descriptors DataSet.

    >>> solvent_df = DataSet([[5, 81],[-93, 111]], index=['benzene', 'toluene'], columns=['melting_point', 'boiling_point'])
    >>> solvent = CategoricalVariable('solvent', 'solvent descriptors', levels=['benzene', 'toluene'],descriptors=solvent_df)
    """

    def __init__(self, name, description, **kwargs):
        """

        Returns
        -------
        object
        """
        Variable.__init__(self, name, description, "categorical", **kwargs)

        # Get descriptors DataSet
        self.ds = kwargs.get("descriptors")
        if self.ds is not None and not isinstance(self.ds, DataSet):
            raise TypeError("descriptors must be a DataSet")

        self._levels = kwargs.get("levels")
        # If levels and descriptors passed, check they match
        if self.ds is not None and self._levels is not None:
            index = self.ds.index
            for level in self._levels:
                assert (
                    level in index
                ), "Levels must be in the descriptors DataSet index."
        # If no levels passed but descriptors passed, make levels the whole index
        elif self.ds is not None and self._levels is None:
            self._levels = self.ds.index.to_list()
        elif self.ds is None and self._levels is None:
            raise ValueError("Levels, descriptors or both must be passed.")

        if type(self._levels) != list:
            raise TypeError("Levels must be a list")
        # check that levels are unique
        if len(self._levels) != len(set(self._levels)):
            raise ValueError("Levels must have unique values.")

    @property
    def levels(self) -> np.ndarray:
        """`numpy.ndarray`: Potential values of the discrete variable"""
        return self._levels

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    @property
    def num_descriptors(self) -> int:
        """Returns the number of descriptors"""
        if self.ds is not None:
            return len(self.ds.data_columns)

    def add_level(self, level):
        """Add a level to the discrete variable

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
        """Remove a level from the discrete variable

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
            self._levels.remove(level)
        except ValueError:
            raise ValueError(f"Level {level} is not in the list of levels.")

    def to_dict(self):
        """ Return json encoding of the variable"""
        variable_dict = super().to_dict()
        ds = self.ds.to_dict() if self.ds is not None else None
        variable_dict.update(dict(levels=self.levels, ds=ds))
        return variable_dict

    @staticmethod
    def from_dict(variable_dict):
        ds = variable_dict["ds"]
        ds = DataSet.from_dict(ds) if ds is not None else None
        return CategoricalVariable(
            name=variable_dict["name"],
            description=variable_dict["description"],
            levels=variable_dict["levels"],
            descriptors=ds,
            is_objective=variable_dict["is_objective"],
        )

    def _html_table_rows(self):
        """Return representation for Jupyter notebooks"""
        return self._make_html_table_rows(f"{self.num_levels} levels")


class Constraint:
    """A constraint for an optimization domain

    Parameters
    ----------
    lhs: str
        The left hand side of a constraint equation
    constraint_type: str
        The type of constraint. Must be <, <=, ==, > or >=. Default: "<="

    Raises
    ------
    ValueError

    Examples
    --------

    These should be constraints in the form "lhs constraint_type constraint 0"
    So for example, x+y=3 should be rewritten as x+y-3=0 and therefore:

    >>> domain = Domain()
    >>> domain += Constraint(lhs="x+y-3", constraint_type="==")

    Or x+y<0 would be:

    >>> domain = Domain()
    >>> domain += Constraint(lhs="x+y", constraint_type="<")

    """

    def __init__(self, lhs, constraint_type="<="):
        self._lhs = lhs
        self._constraint_type = constraint_type
        if self.constraint_type not in ["<", "<=", "==", ">", ">="]:
            raise ValueError("Constraint type must be <, <=, ==, > or >=")

    @property
    def lhs(self):
        return self._lhs

    @property
    def constraint_type(self):
        return self._constraint_type

    def _html_table_rows(self):
        columns = []
        columns.append("")  # name column
        columns.append("constraint")  # type column
        columns.append(self.lhs)  # description columns
        columns.append("")  # value column
        html = "".join([f"<td>{column}</td>" for column in columns])
        return f"<tr>{html}</tr>"


class Domain:
    """Representation of the optimization domain

    Parameters
    ---------
    variables: :class:`~summit.domain.Variable` or list of :class:`~summit.domain.Variable` like objects, optional
        list of variable objects (i.e., `ContinuousVariable`, `CategoricalVariable`)
    constraints: :class:`~summit.domain.Constraint` or  list of :class:`~summit.domain.Constraint` objects, optional
        list of constraints on the problem

    Attributes
    ----------
    variables

    Raises
    ------
    TypeError
        If variables or constraints are not lists or a single instance of the object
    ValueError
        If variable names are not unique

    Examples
    --------
    >>> domain = Domain()
    >>> domain += ContinuousVariable('temperature', 'reaction temperature', [1, 100])

    """

    def __init__(self, variables=[], constraints=[]):
        # Check types
        e = TypeError("variables must be Variable or list of Variable objects")
        if isinstance(variables, Variable):
            variables = [variables]
        elif not isinstance(variables, list):
            raise e
        else:
            for l in variables:
                if not isinstance(l, Variable):
                    raise e

        e = TypeError("constraints must be Constraint or list of Constraint objects")
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        elif not isinstance(constraints, list):
            raise e
        else:
            for l in constraints:
                if not isinstance(l, Constraint):
                    raise e

        self._variables = variables
        self._constraints = constraints
        # Check that all the output variables continuous
        # self._raise_noncontinuous_outputs()
        self._raise_names_not_unique()

    @property
    def variables(self):
        """[List[Type[Variable]]]: List of variables in the domain"""
        return self._variables

    @property
    def constraints(self):
        return self._constraints

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

    def get_categorical_combinations(self):
        """Get all combinations of categoricals using full factorial design

        Returns
        -------
        ds: DataSet
            A dataset containing the combinations of all categorical cvariables.
        """
        levels = [
            len(v.levels)
            for v in self.input_variables
            if v.variable_type == "categorical"
        ]
        doe = fullfact(levels)
        i = 0
        combos = {}
        for v in self.input_variables:
            if v.variable_type == "categorical":
                indices = doe[:, i]
                indices = indices.astype(int)
                combos[v.name, "DATA"] = [v.levels[i] for i in indices]
                i += 1
        return DataSet(combos)

    def _raise_noncontinuous_outputs(self):
        """Raise an error if the outputs are not continuous variables"""
        for v in self.output_variables:
            if v.variable_type != "continuous":
                raise DomainError("All output variables must be continuous")

    def _raise_names_not_unique(self):
        if len(set(self._variables)) != len(self._variables):
            raise ValueError("Variable names are not unique")

    def num_variables(self, include_outputs=False) -> int:
        """Number of variables in the domain

        Parameters
        ----------
        include_outputs: bool, optional
            If True include output variables in the count.
            Defaults to False.

        Returns
        -------
        num_variables: int
            Number of variables in the domain
        """
        k = 0
        for v in self.variables:
            if v.is_objective and not include_outputs:
                continue
            k += 1
        return k

    def num_discrete_variables(self, include_outputs=False) -> int:
        raise NotImplementedError(
            "num_discrete_variables has been deprecated due to the change of Discrete to Categorical variables"
        )

    def num_continuous_dimensions(
        self, include_descriptors=False, include_outputs=False
    ) -> int:
        """The number of continuous dimensions

        Parameters
        ----------
        include_descriptors : bool, optional
            If True, the number of descriptors columns are considered.
            Defaults to False.
        include_outputs : bool, optional
            If True include output variables in the count.
            Defaults to False.

        Returns
        -------
        num_variables: int
            Number of variables in the domain
        """
        k = 0
        for v in self.variables:
            if v.is_objective and not include_outputs:
                continue
            if isinstance(v, ContinuousVariable):
                k += 1
            if isinstance(v, CategoricalVariable) and include_descriptors:
                if v.num_descriptors is not None:
                    k += v.num_descriptors
        return k

    def to_dict(self):
        """Return a dictionary representation of the domain"""
        return [variable.to_dict() for variable in self.variables]

    def to_json(self):
        """Return the a json representation of the domain"""
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(domain_list):
        variables = []
        for variable in domain_list:
            if variable["type"] == "ContinuousVariable":
                new_variable = ContinuousVariable.from_dict(variable)
            elif variable["type"] == "CategoricalVariable":
                new_variable = CategoricalVariable.from_dict(variable)
            else:
                raise ValueError(
                    f"Cannot load variable of type:{variable['type']}. Variable should be continuous, discrete or descriptors"
                )
            variables.append(new_variable)
        return Domain(variables)

    def __add__(self, obj):
        # TODO: make this work with adding arrays of variable or constraints
        if isinstance(obj, Variable):
            if obj.is_objective and obj.variable_type != "continuous":
                raise DomainError("Output variables must be continuous")
            return Domain(
                variables=self._variables + [obj], constraints=self.constraints
            )
        elif isinstance(obj, Constraint):
            return Domain(
                variables=self.variables, constraints=self.constraints + [obj]
            )
        else:
            raise RuntimeError("Not a supported domain object.")

    def _repr_html_(self):
        """Build html string for table display in jupyter notebooks.

        Notes
        -----
        Adapted from https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/domain.py
        """
        html = ["<table id='domain' width=100%>"]

        # Table header
        columns = ["Name", "Type", "Description", "Values"]
        header = "<tr>"
        header += "".join(map(lambda l: "<td><b>{0}</b></td>".format(l), columns))
        header += "</tr>"
        html.append(header)

        # Add parameters
        html.append(self._html_table_rows())
        html.append("</table>")

        return "".join(html)

    def _html_table_rows(self):
        variables = "".join([v._html_table_rows() for v in self.variables])
        constraints = "".join([c._html_table_rows() for c in self.constraints])
        return f"{variables}{constraints}"

    def __getitem__(self, key):
        for v in self.variables:
            if v.name == key:
                return v
        raise ValueError("Variable not in domain")

    def __setitem__(self, key, value):
        for i, v in enumerate(self.variables):
            if v.name == key:
                self._variables.pop(i)
                self._variables.insert(i, value)

    def copy(self):
        return deepcopy(self)


class DomainError(Exception):
    pass


def fullfact(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor


    Notes
    ------
    This code is copied from pydoe2: https://github.com/clicumu/pyDOE2/blob/master/pyDOE2/doe_factorial.py

    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j] * level_repeat
        rng = lvl * range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H
