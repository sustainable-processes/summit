from summit.domain import (
    Domain,
    Variable,
    ContinuousVariable,
    DiscreteVariable,
    DescriptorsVariable,
    DomainError,
)
from summit.utils.models import ModelGroup
from summit.utils.dataset import DataSet

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod, abstractclassmethod
from typing import Type, Tuple
import json

__all__ = [
    "Transform",
    "Strategy",
    "Design",
    "MultitoSingleObjective",
    "LogSpaceObjectives",
    "Chimera",
]


class Transform:
    """  Pre/post-processing of data for strategies
    
    Parameters
    ---------- 
    domain: `sumit.domain.Domain``
        A domain for that is being used in the strategy

    Notes
    ------
    This class can be overridden to create custom transformations as necessary.    
    
    """

    def __init__(self, domain):
        self.transform_domain = domain.copy()
        self.domain = domain

    def transform_inputs_outputs(self, ds: DataSet, copy=True):
        """  Transform of data into inputs and outptus for a strategy
        
        Parameters
        ---------- 
        ds: `DataSet`
            Dataset with columns corresponding to the inputs and objectives of the domain.
        copy: bool, optional
            Copy the dataset internally. Defaults to True.

        Returns
        -------
        inputs, outputs
            Datasets with the input and output datasets  
        """
        data_columns = ds.data_columns
        new_ds = ds.copy() if copy else ds

        # Determine input and output columns in dataset
        input_columns = []
        output_columns = []

        for variable in self.domain.variables:
            check_input = variable.name in data_columns and not variable.is_objective

            if check_input and variable.variable_type != "descriptors":
                input_columns.append(variable.name)
            elif check_input and variable.variable_type == "descriptors":
                # Add descriptors to the dataset
                indices = new_ds[variable.name].values
                descriptors = variable.ds.loc[indices]
                new_metadata_name = descriptors.index.name
                descriptors.index = new_ds.index
                new_ds = new_ds.join(descriptors, how="inner")

                # Make the original descriptors column a metadata column
                column_list_1 = new_ds.columns.levels[0].to_list()
                ix = column_list_1.index(variable.name)
                column_list_1[ix] = new_metadata_name
                new_ds.columns.set_levels(column_list_1, level=0, inplace=True)
                column_codes_2 = list(new_ds.columns.codes[1])
                ix_code = np.where(new_ds.columns.codes[0] == ix)[0][0]
                column_codes_2[ix_code] = 1
                new_ds.columns.set_codes(column_codes_2, level=1, inplace=True)

                # add descriptors data columns to inputs
                input_columns += descriptors.data_columns
            elif variable.name in data_columns and variable.is_objective:
                if variable.variable_type == "descriptors":
                    raise DomainError(
                        "Output variables cannot be descriptors variables."
                    )
                output_columns.append(variable.name)
            else:
                raise DomainError(f"Variable {variable.name} is not in the dataset.")

        if output_columns is None:
            raise DomainError(
                "No output columns in the domain.  Add at least one output column for optimization."
            )

        # Return the inputs and outputs as separate datasets
        return new_ds[input_columns].copy(), new_ds[output_columns].copy()

    def un_transform(self, ds):
        """ Transform data back into its original represetnation
            after strategy is finished 
        
        Parameters
        ---------- 
        ds: `DataSet`
            Dataset with columns corresponding to the inputs and objectives of the domain.

        Notes
        -----
        Override this class to achieve custom untransformations 
        """
        return ds

    def to_dict(self, **kwargs):
        """ Output a dictionary representation of the transform"""
        return dict(
            transform_domain=self.transform_domain.to_dict(),
            name=self.__class__.__name__,
            domain=self.domain.to_dict(),
            transform_params=kwargs
        )

    @classmethod
    def from_dict(cls, d):
        t = cls(Domain.from_dict(d["domain"]), **d["transform_params"])
        t.transform_domain = Domain.from_dict(d["transform_domain"])
        return t


def transform_from_dict(d):
    if d["name"] == "MultitoSingleObjective":
        return MultitoSingleObjective.from_dict(d)
    elif d["name"] == "LogSpaceObjectives":
        return LogSpaceObjectives.from_dict(d)
    elif d["name"] == "Chimera":
        return Chimera.from_dict(d)
    elif d["name"] == "Transform":
        return Transform.from_dict(d)


class MultitoSingleObjective(Transform):
    """  Transform a multiobjective problem into a single objective problems
    
    Parameters
    ---------- 
    domain: `sumit.domain.Domain``
        A domain for that is being used in the strategy
    expression: str
        An expression in terms of variable names used to
        convert the multiobjective problem into a single
        objective problem
    
    Returns
    -------
    result: `bool`
        description
    
    Raises
    ------
    ValueError
        If domain does not have at least two objectives
    
    """

    def __init__(self, domain: Domain, expression: str, maximize=True):
        super().__init__(domain)
        objectives = [v for v in self.transform_domain.variables if v.is_objective]
        num_objectives = len(objectives)
        if num_objectives <= 1:
            raise ValueError(
                f"Domain must have at least two objectives; it currently has {num_objectives} objectives."
            )
        self.expression = expression

        # Replace objectives in transform domain
        for v in objectives:
            i = self.transform_domain.variables.index(v)
            self.transform_domain.variables.pop(i)
        self.transform_domain += ContinuousVariable(
            "scalar_objective",
            description=expression,
            bounds=[-np.inf, np.inf],
            is_objective=True,
            maximize=maximize,
        )
        self.maximize = maximize

    def transform_inputs_outputs(self, ds, copy=True):
        inputs, outputs = super().transform_inputs_outputs(ds, copy=copy)
        outputs = outputs.eval(self.expression, resolvers=[outputs])
        outputs = DataSet(outputs, columns=["scalar_objective"])
        return inputs, outputs

    def to_dict(self):
        """ Output a dictionary representation of the transform"""
        transform_params = dict(expression=self.expression, maximize=self.maximize)
        d = super().to_dict(**transform_params)
        return d


class LogSpaceObjectives(Transform):
    """  Log transform objectives
    
    Parameters
    ---------- 
    domain: `sumit.domain.Domain``
        A domain for that is being used in the strategy

    Raises
    ------
    ValueError
        When the domain has no objectives.
    
    """

    def __init__(self, domain: Domain):
        super().__init__(domain)
        objectives = [
            (i, v)
            for i, v in enumerate(self.transform_domain.variables)
            if v.is_objective
        ]

        # Check that the domain has objectives
        num_objectives = len(objectives)
        if num_objectives == 0:
            raise ValueError(
                f"The domain must have objectives. Currently has {num_objectives} objectives."
            )

        # Rename objectives in new domain
        for i, v in objectives:
            v.name = "log_" + v.name

    def transform_inputs_outputs(self, ds, copy=True):
        """  Transform of data into inputs and outptus for a strategy
        
        This will do a log transform on the objectives (outputs).

        Parameters
        ---------- 
        ds: `DataSet`
            Dataset with columns corresponding to the inputs and objectives of the domain.
        copy: bool, optional
            Copy the dataset internally. Defaults to True.

        Returns
        -------
        inputs, outputs
            Datasets with the input and output datasets  
        """
        inputs, outputs = super().transform_inputs_outputs(ds, copy=copy)
        if (outputs.any() < 0).any():
            raise ValueError("Cannot complete log transform for values less than zero.")
        outputs = outputs.apply(np.log)
        columns = [v.name for v in self.transform_domain.variables if v.is_objective]
        outputs = DataSet(outputs.data_to_numpy(), columns=columns)
        return inputs, outputs

    def un_transform(self, ds):
        """ Untransform objectives from log space to
        
        Parameters
        ---------- 
        ds: `DataSet`
            Dataset with columns corresponding to the inputs and objectives of the domain.

        Notes
        -----
        Override this class to achieve custom untransformations 
        """
        ds = super().un_transform(ds)
        for v in self.domain.variables:
            if v.is_objective and ds.get("log_" + v.name):
                ds[v.name] = np.exp(ds["log_" + v.name])
        return ds


class Chimera(Transform):
    """ Scalarize a multiobjective problem using Chimera.

    Chimera is a hiearchical multiobjective scalarazation function developed by
    Häse et al[1]_[2]_. You set the parameter `loss_tolerances` to weight the importance
    of each objective.

    Parameters
    ---------- 
    domain : `sumit.domain.Domain``
        A domain for that is being used in the strategy
    hierarchy : dict
        Dictionary with keys as the names of the objectives and values as dictionaries
        with the keys hierarchy and tolerance for the ranking and tolerance on each objective.
    softness : float, optional
        Smoothing parameter. Defaults to 1e-3 as recommended by Häse et al [1]_. 
        Larger values result in a more smooth objective while smaller values
        will give a disjointed objective.
    absolutes : array-like, optional
        Default is zeros.s
    
    Examples
    --------

    Notes
    ------
    This code is based on the code for Griffyn[2]_, which can be found on `Github <https://github.com/aspuru-guzik-group/gryffin/blob/d7443bf374e5d1fee2424cb49f5008ce4248d432/src/gryffin/observation_processor/chimera.py://www.example.com>`_
    
    Chimera turns problems into minimization problems. This is done automatically by reading the type 
    of objective from the domain.
    
    References
    ----------
    .. [1] Häse, F., Roch, L. M., & Aspuru-Guzik, A. "Chimera: enabling hierarchy based multi-objective
           optimization for self-driving laboratories." Chemical Science, 2018, 9,7642-7655
    .. [2] Häse, F., Roch, L.M. and Aspuru-Guzik, A., 2020. Gryffin: An algorithm for Bayesian 
           optimization for categorical variables informed by physical intuition with applications to chemistry. 
           arXiv preprint arXiv:2003.12127.
    
    """

    def __init__(self, domain: Domain, hierarchy: dict, softness=1e-3, absolutes=None):
        super().__init__(domain)

        # Sort objectives
        # {'y_0': {'hiearchy': 0, 'tolerance': 0.2}}
        objectives = self.transform_domain.output_variables
        self.hierarchy = hierarchy
        self.tolerances = np.zeros_like(objectives)
        self.directions = np.zeros_like(objectives)
        self.ordered_objective_names = len(objectives) * [""]
        for name, v in hierarchy.items():
            h = v["hierarchy"]
            self.ordered_objective_names[h] = name
            self.tolerances[h] = v["tolerance"]
            self.directions[h] = -1 if self.domain[name].maximize else 1

        # Pop objectives from transform domain
        for v in objectives:
            i = self.transform_domain.variables.index(v)
            self.transform_domain.variables.pop(i)

        # Add chimera objective to transform domain
        self.transform_domain += ContinuousVariable(
            "chimera",
            "chimeras scalarized objectived",
            bounds=[0, 1],
            is_objective=True,
            maximize=False,
        )

        # Set chimera parameters
        self.absolutes = absolutes
        if self.absolutes is None:
            self.absolutes = np.zeros(len(self.tolerances)) + np.nan
        self.softness = softness

    def transform_inputs_outputs(self, ds, copy=True):
        # Get inputs and outputs
        inputs, outputs = super().transform_inputs_outputs(ds, copy=copy)

        # Scalarize using Chimera
        outputs_arr = outputs[self.ordered_objective_names].to_numpy()
        outputs_arr = outputs_arr*self.directions #Change maximization to minimization
        scalarized_array = self._scalarize(outputs_arr)

        # Write scalarized objective back to DataSEt
        outputs = DataSet(scalarized_array, columns=["chimera"])
        return inputs, outputs

    def _scalarize(self, raw_objs):
        res_objs, res_abs = self._rescale(raw_objs)
        shifted_objs, abs_tols = self._shift_objectives(res_objs, res_abs)
        scalarized_obj = self._scalarize_objs(shifted_objs, abs_tols)
        return scalarized_obj

    def _scalarize_objs(self, shifted_objs, abs_tols):
        scalar_obj = shifted_objs[-1].copy()
        for index in range(0, len(shifted_objs) - 1)[::-1]:
            scalar_obj *= self._step(-shifted_objs[index] + abs_tols[index])
            scalar_obj += (
                self._step(shifted_objs[index] - abs_tols[index]) * shifted_objs[index]
            )
        return scalar_obj.transpose()

    def _soft_step(self, value):
        arg = -value / self.softness
        return 1.0 / (1.0 + np.exp(arg))

    def _hard_step(self, value):
        result = np.empty(len(value))
        result = np.where(value > 0.0, 1.0, 0.0)
        return result

    def _step(self, value):
        if self.softness < 1e-5:
            return self._hard_step(value)
        else:
            return self._soft_step(value)

    def _rescale(self, raw_objs):
        """Min-Max scale objectives and absolutes by between 0 and 1"""
        res_objs = np.empty(raw_objs.shape)
        res_abs = np.empty(self.absolutes.shape)
        for index in range(raw_objs.shape[1]):
            min_objs, max_objs = (
                np.amin(raw_objs[:, index]),
                np.amax(raw_objs[:, index]),
            )
            if min_objs < max_objs:
                res_abs[index] = (self.absolutes[index] - min_objs) / (
                    max_objs - min_objs
                )
                res_objs[:, index] = (raw_objs[:, index] - min_objs) / (
                    max_objs - min_objs
                )
            else:
                res_abs[index] = self.absolutes[index] - min_objs
                res_objs[:, index] = raw_objs[:, index] - min_objs
        return res_objs, res_abs

    def _shift_objectives(self, objs, res_abs):
        transposed_objs = objs.transpose()
        shapes = transposed_objs.shape
        shifted_objs = np.empty((shapes[0] + 1, shapes[1]))

        mins, maxs, tols = [], [], []
        domain = np.arange(shapes[1])
        shift = 0
        for obj_index, obj in enumerate(transposed_objs):
            # get absolute tolerances
            minimum = np.amin(obj[domain])
            maximum = np.amax(obj[domain])
            mins.append(minimum)
            maxs.append(maximum)
            tolerance = minimum + self.tolerances[obj_index] * (maximum - minimum)
            if np.isnan(tolerance):
                tolerance = res_abs[obj_index]

            # adjust region of interest
            interest = np.where(obj[domain] < tolerance)[0]
            if len(interest) > 0:
                domain = domain[interest]

            # apply shift
            tols.append(tolerance + shift)
            shifted_objs[obj_index] = transposed_objs[obj_index] + shift

            # compute new shift
            if obj_index < len(transposed_objs) - 1:
                shift -= np.amax(transposed_objs[obj_index + 1][domain]) - tolerance
            else:
                shift -= np.amax(transposed_objs[0][domain]) - tolerance
                shifted_objs[obj_index + 1] = transposed_objs[0] + shift
        return shifted_objs, tols
    
    def to_dict(self):
        transform_params = dict(hierarchy=self.hierarchy,
                                softness=self.softness,
                                absolutes=self.absolutes)
        return super().to_dict(**transform_params)


class Strategy(ABC):
    """ Base class for strategies 
    
    Parameters
    ---------- 
    domain: `summit.domain.Domain`
        A summit domain containing variables and constraints
    transform: `summit.strategies.base.Transform`, optional
        A transform class (i.e, not the object itself). By default
        no transformation will be done the input variables or
        objectives.
    
    """

    def __init__(self, domain: Domain, transform: Transform = None, **kwargs):
        if transform is None:
            self.transform = Transform(domain)
        elif isinstance(transform, Transform):
            self.transform = transform
        else:
            raise TypeError("transform must be a Transform class")
        self.domain = self.transform.transform_domain

    @abstractmethod
    def suggest_experiments(self):
        raise NotImplementedError(
            "Strategies should inhereit this class and impelemnt suggest_experiments"
        )

    def to_dict(self, **strategy_params):
        """Convert strategy to jsonable format
        
        You can pass in as keyword arguments any custom parameters
        for a strategy, which will be stored under the key strategy_params.
        """
        return dict(
            name=self.__class__.__name__,
            transform=self.transform.to_dict(),
            strategy_params=strategy_params,
        )

    @classmethod
    def from_dict(cls, d):
        """Create a strategy from a dictionary"""
        transform = transform_from_dict(d["transform"])
        return cls(domain=transform.domain, transform=transform, **d["strategy_params"])

    def save(self, filename):
        """Save a strategy to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        """Load a strategy from a JSON file"""
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)


class Design:
    """Representation of an experimental design
    
    Parameters
    ---------- 
    domain: summit.domain.Domain
        The domain of the design
    num_samples: int
        Number of samples in the design 
    design_type: str
        The name of the design type 

    Examples
    --------
    >>> from summit.domain import Domain, ContinuousVariable
    >>> domain = Domain()
    >>> domain += ContinuousVariable('temperature','reaction temperature', [1, 100])
    >>> initial_design = Design(domain, 10, 'example_design')
    >>> initial_design.add_variable('temperature',  np.array([[100, 120, 150]]))

    """

    def __init__(self, domain: Domain, num_samples, design_type: str, exclude=[]):
        self._variable_names = [variable.name for variable in domain.variables]
        self._indices = domain.num_variables() * [0]
        self._values = domain.num_variables() * [0]
        self.num_samples = num_samples
        self.design_type = design_type
        self.exclude = exclude
        self._domain = domain

    def add_variable(
        self, variable_name: str, values: np.ndarray, indices: np.ndarray = None
    ):
        """ Add a variable to a design 
        
        Parameters
        ---------- 
        variable_name: str
            Name of the variable to be added. Must already be in the domain.
        values: numpy.ndarray
            Values of the design points in the variable. 
            Should be an nxd array, where n is the number of samples and 
            d is the number of dimensions of the variable.
        indices: numpy.ndarray, optional
            Indices of the design points in the variable
        
        Raises
        ------
        ValueError
            If indices or values are not a two-dimensional array.
        """
        variable_index = self._get_variable_index(variable_name)
        if values.ndim < 2:
            raise ValueError("Values must be 2 dimensional. Use np.atleast_2d.")
        if indices is not None:
            if indices.ndim < 2:
                raise ValueError("Indices must be 2 dimensional. Use np.atleast_2d.")
            self._indices[variable_index] = indices
        self._values[variable_index] = values

    def get_indices(self, variable_name: str) -> np.ndarray:
        """ Get indices of designs points  
        
        Parameters
        ---------- 
        variable_name: str, optional
            Get only the indices for a specific variable name.
        
        Returns
        -------
        indices: numpy.ndarray
            Indices of the design pionts
        
        Raises
        ------
        ValueError
            If the variable name is not in the list of variables
        """
        variable_index = self._get_variable_index(variable_name)
        indices = self._indices[variable_index]
        return indices

    def get_values(self, variable_name: str = None) -> np.ndarray:
        """ Get values of designs points  
        
        Parameters
        ---------- 
        variable_name: str, optional
            Get only the values for a specific variable name.
        
        Returns
        -------
        values: numpy.ndarray
            Values of the design pionts
        
        Raises
        ------
        ValueError
            If the variable name is not in the list of variables
        """
        if variable_name is not None:
            variable_index = self._get_variable_index(variable_name)
            values = self._values[variable_index].T
        else:
            values = np.concatenate(self._values, axis=0).T

        return values

    def to_dataset(self) -> DataSet:
        """ Get design as a pandas dataframe 
        Returns
        -------
        ds: summit.utils.dataset.Dataset
        """
        df = pd.DataFrame([])
        i = 0
        for variable in self._domain.variables:
            if variable.is_objective or variable.name in self.exclude:
                continue
            if variable.variable_type == "descriptors":
                descriptors = variable.ds.iloc[self.get_indices(variable.name)[:, 0], :]
                descriptors = descriptors.rename_axis(variable.name)
                df = pd.concat([df, descriptors.index.to_frame(index=False)], axis=1)
                i += variable.num_descriptors
            else:
                df.insert(i, variable.name, self.get_values(variable.name)[:, 0])
                i += 1

        return DataSet.from_df(df)

    def _get_variable_index(self, variable_name: str) -> int:
        """Method for getting the internal index for a variable"""
        if not variable_name in self._variable_names:
            raise ValueError(f"Variable {variable_name} not in domain.")
        return self._variable_names.index(variable_name)

    # def coverage(self, design_indices, search_matrix=None,
    #              metric=closest_point_distance):
    #     ''' Get coverage statistics for a design based
    #     Arguments:
    #         design_indices: Indices in the search matrix of the design points
    #         search_matrix (optional): A matrix of descriptors used for calculating the coverage. By default, the
    #                                   descriptor matrix in the instance of solvent select will be used as the search
    #                                   matrix
    #         metric (optional): A function for calculating the coverage. By default this is the closest point.
    #                            The function should take a design point as its first argument and a candidate matrix
    #                            as its second argument.
    #     Notes:
    #         Coverage statistics are calculated by finding the distance between each point in the search matrix
    #         and the closest design point. The statistics are mean, standard deviation, median, maximum, and minimum
    #         of the distances.
    #     Returns
    #         An instance of `DesignCoverage`

    #     '''
    #     if search_matrix is None:
    #         search_matrix = self.descriptor_df.values

    #     mask = np.ones(search_matrix.shape[0], dtype=bool)
    #     mask[design_indices] = False
    #     distances = [metric(row, search_matrix[design_indices, :])
    #                 for row in search_matrix[mask, ...]]
    #     mean = np.average(distances)
    #     std_dev = np.std(distances)
    #     median = np.median(distances)
    #     max = np.max(distances)
    #     min = np.min(distances)
    #     return DesignCoverage(
    #                     mean=mean,
    #                     std_dev=std_dev,
    #                     median=median,
    #                     max = max,
    #                     min = min
    #                     )

    def _repr_html_(self):
        return self.to_frame().to_html()


class DesignCoverage:
    properties = ["mean", "std_dev", "median", "max", "min"]

    def __init__(self, mean=None, std_dev=None, median=None, max=None, min=None):
        self._mean = mean
        self._std_dev = std_dev
        self._median = median
        self._max = max
        self._min = min

    @property
    def mean(self):
        return self._mean

    @property
    def std_dev(self):
        return self._std_dev

    @property
    def median(self):
        return self._median

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    def __repr__(self):
        values = "".join(
            [f"{property}:{getattr(self, property)}, " for property in self.properties]
        )
        return f"""DesignCoverage({values.rstrip(", ")})"""

    def get_dict(self):
        return {property: getattr(self, property) for property in self.properties}

    def get_array(self):
        return [getattr(self, property) for property in self.properties]

    @staticmethod
    def average_coverages(coverages):
        """Average multiple design coverages
        
        Arguments:
            coverages: a list of `DesignCoverage` objects.
        """
        # Check that argument is  a list of coverages
        for coverage in coverages:
            assert isinstance(coverage, DesignCoverage)

        avg_mean = np.average([coverage.mean for coverage in coverages])
        avg_std_dev = np.average([coverage.std_dev for coverage in coverages])
        avg_median = np.average([coverage.median for coverage in coverages])
        avg_max = np.average([coverage.max for coverage in coverages])
        avg_min = np.average([coverage.min for coverage in coverages])
        return DesignCoverage(
            mean=avg_mean,
            std_dev=avg_std_dev,
            median=avg_median,
            max=avg_max,
            min=avg_min,
        )


def _closest_point_indices(design_points, candidate_matrix, unique=False):
    """Return the indices of the closest point in the candidate matrix to each design point"""
    if unique:
        mask = np.ones(candidate_matrix.shape[0], dtype=bool)
        indices = [0 for i in range(len(design_points))]
        for i, design_point in enumerate(design_points):
            masked_candidates = candidate_matrix[mask, :]
            point_index = _closest_point_index(design_point, masked_candidates)
            actual_index = np.where(
                candidate_matrix == masked_candidates[point_index, :]
            )[0][0]
            indices[i] = actual_index
            mask[actual_index] = False
    else:
        indices = [
            _closest_point_index(design_point, candidate_matrix)
            for design_point in design_points
        ]
    indices = np.array(indices)
    return np.atleast_2d(indices).T


def _closest_point_index(design_point, candidate_matrix):
    """Return the index of the closest point in the candidate matrix"""
    distances = _design_distances(design_point, candidate_matrix)
    return np.argmin(np.atleast_2d(distances))


def _design_distances(design_point, candidate_matrix):
    """ Return the distances between a design_point and all candidates"""
    diff = design_point - candidate_matrix
    squared = np.power(diff, 2)
    summed = np.sum(squared, axis=1)
    root_square = np.sqrt(summed)
    return root_square
