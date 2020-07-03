from abc import ABC, abstractmethod

import os
import os.path as osp
import numpy as np
import json

from sklearn.model_selection import train_test_split as sklearn_train_test_split
import sklearn.preprocessing

import matplotlib.pyplot as plt

class Emulator(ABC):
    """Base class for emulator training

    Parameters
    ---------
    domain: summit.domain.Domain
        The domain of the experiment
    dataset: summit.utils.dataset.Dataset
        The data points obtained from an experiment the emulator
        is trained on.

    Notes
    -----
    Developers that subclass `Experiment` need to implement
    `_run`, which runs the experiments.

    """

    def __init__(self, model, **kwargs):
        self._model = model

    @property
    def domain(self):
        """The  domain for the experiment"""
        return self._domain

    @property
    def dataset(self):
        """Dataset of all experiments trained on"""
        return self._dataset

    @property
    def model(self):
        """Model that is trained"""
        return self._model

    @abstractmethod
    def _setup_model(self, **kwargs):
        """ Setup model structure.

        Arguments
        ---------

        Returns
        -------
        model
            Should return a regression model that can be trained on experimental data.
        """

        raise NotImplementedError("_steup_model be implemented by subclasses of Emulator")

    @abstractmethod
    def train_model(self, verbose=True, parity_plot=False):
        """ Train model on a given Summit Dataset.

        Arguments
        ---------

        Returns
        -------
        model
            Should return a regression model that is trained on experimental data.
        """

        raise NotImplementedError("_train_model be implemented by subclasses of Emulator")

    @abstractmethod
    def validate_model(self):
        """ Validate a model on a given Summit Dataset.

        Arguments
        ---------

        Returns
        -------
        model
            Should return evaluation values w.r.t. the accuracy of the
             regression model.
        """

        raise NotImplementedError("_validate_model be implemented by subclasses of Emulator")

    @abstractmethod
    def infer_model(self):
        raise NotImplementedError("_infer_model be implemented by subclasses of Emulator")

    @abstractmethod
    def _save_model(self, **kwargs):
        raise NotImplementedError("_save_model be implemented by subclasses of Emulator")

    def _domain_preprocess(self):

        self.input_dim = 0
        self.output_dim = 0
        self.out_mean = np.asarray([])
        self.output_models = {}
        self.input_names = []

        for i, v in enumerate(self._domain.variables):
            if not v.is_objective:
                if v.variable_type == "continuous":
                    self.input_dim += 1
                    self.input_names.append(v.name)
                elif v.variable_type == "discrete":
                    self.input_dim += len(v.levels)
                    self.input_names.append(v.name)
                    # create one-hot tensor for discrete inputs
                elif v.variable_type == "descriptors":
                    raise TypeError(
                        "Regressor not trainable on descriptor variables. Please redefine {} as continuous or discrete variable".format(
                            v.name))
                else:
                    raise TypeError("Unknown variable type: {}.".format(v.variable_type))
            else:
                if v.variable_type == "continuous":
                    self.output_dim += 1
                    self.output_models[v.name] = ""
                elif v.variable_type == "discrete":
                    raise TypeError(
                        "{} is a discrete variable. Regressor not trainable for discrete outputs.".format(v.name))
                elif v.variable_type == "descriptors":
                    raise TypeError(
                        "{} is a descriptor variable. Regressor not trainable for descriptor outputs.".format(
                            v.name))
                else:
                    raise TypeError("Unknown variable type: {}.".format(v.variable_type))

    def _data_preprocess(
            self, inference=False, infer_dataset=None, validate=False, transform_input="standardize",
            transform_output="standardize", test_size=0.1, shuffle=False, kwargs={}
    ):
        if not inference:
            np_dataset = self._dataset.data_to_numpy()
            data_column_names = self._dataset.data_columns
        else:
            np_dataset = infer_dataset.data_to_numpy()
            if not validate:
                data_column_names = [c[0] for c in infer_dataset.data_columns]
            else:
                data_column_names = infer_dataset.data_columns

        self.input_data_continuous = []
        self.input_data_discrete = []
        self.output_data = []
        if not inference:
            self.data_transformation_dict = {}

        # this loop makes sure that the inputs are always in the same order and only
        # data with the same column names as in the domain is considered
        for v in self._domain.variables:
            v_in_dataset = False
            for i, c_name in enumerate(data_column_names):
                if c_name == v.name:
                    v_in_dataset = True
                    if not v.is_objective:
                        if v.variable_type == "continuous":
                            # Standardize continuous inputs
                            tmp_cont_inp = np.asarray(np_dataset[:, i], dtype=float)
                            if not inference:
                                tmp_cont_inp, _reduce, _divide = self._transform_data(data=tmp_cont_inp, transformation_type=transform_input)
                                self.data_transformation_dict[v.name] = [_reduce, _divide]
                            else:
                                tmp_cont_inp, _, _ = self._transform_data(data=tmp_cont_inp, reduce=self.data_transformation_dict[v.name][0], divide=self.data_transformation_dict[v.name][1])
                            self.input_data_continuous.append(tmp_cont_inp)
                        elif v.variable_type == "discrete":
                            # create one-hot tensor for discrete inputs
                            one_hot_enc = sklearn.preprocessing.OneHotEncoder(categories=[v.levels])
                            tmp_disc_inp_one_hot = one_hot_enc.fit_transform(np_dataset[:, i].reshape(-1, 1)).toarray()
                            self.input_data_discrete.append(np.asarray(tmp_disc_inp_one_hot))
                        elif v.variable_type == "descriptors":
                            raise TypeError(
                                "Regressor not trainable on descriptor variables. Please redefine {} as continuous or discrete variable".format(
                                    v.name))
                        else:
                            raise TypeError("Unknown variable type: {}.".format(v.variable_type))
                    elif not inference:
                        if v.variable_type == "continuous":
                            tmp_cont_out = np.asarray(np_dataset[:, i], dtype=float)
                            if not inference:
                                tmp_cont_out, _reduce, _divide = self._transform_data(data=tmp_cont_out, transformation_type=transform_output)
                                self.data_transformation_dict[v.name] = [_reduce, _divide]
                            self.output_data.append(tmp_cont_out)
                        elif v.variable_type == "discrete":
                            raise TypeError(
                                "{} is a discrete variable. Regressor not trainable for discrete outputs.".format(
                                    v.name))
                        elif v.variable_type == "descriptors":
                            raise TypeError(
                                "{} is a descriptor variable. Regressor not trainable for descriptor outputs.".format(
                                    v.name))
                        else:
                            raise TypeError("Unknown variable type: {}.".format(v.variable_type))
            if v_in_dataset == False:
                raise ValueError("Variable {} defined in the domain is missing in the given dataset.".format(v.name))

        self.input_data_continuous = np.asarray(self.input_data_continuous).transpose()
        self.input_data_discrete = np.asarray(self.input_data_discrete[0])
        self.output_data = np.asarray(self.output_data).transpose()

        # Set up training and test data
        if not inference:
            final_np_dataset = np.concatenate([self.input_data_continuous, self.input_data_discrete, self.output_data],
                                              axis=1)
            X, y = final_np_dataset[:, :-self.output_dim], final_np_dataset[:, -self.output_dim:]
            X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=test_size, shuffle=shuffle)
            return [X_train.astype(dtype=float), y_train.astype(dtype=float)], [X_test.astype(dtype=float),
                                                                                y_test.astype(dtype=float)]
        else:
            X = np.concatenate([self.input_data_continuous, self.input_data_discrete], axis=1)
            return X.astype(dtype=float)

    def _transform_data(self, data, transformation_type=None, reduce=None, divide=None, infer=False, kwargs={}):
        """ Transform data according to transformation type (standardize, normalize)"""
        if not infer:
            if transformation_type == "standardize":
                tmp_reduce = data.mean()
                tmp_divide = data.std()
            elif transformation_type == "normalize":
                tmp_reduce = np.float64(0)
                tmp_divide = data.mean()
            elif transformation_type == "min_max":
                min, max = kwargs.get("min", 0), kwargs.get("max", 1)
                tmp_reduce = np.float64(min)
                tmp_divide = np.float64(max - min)
            else:
                tmp_reduce = reduce if reduce else np.float64(0)
                tmp_divide = divide if divide else np.float64(1)
        else:
            tmp_reduce, tmp_divide = reduce, divide
        transf_data = (data - tmp_reduce) / tmp_divide
        return transf_data, tmp_reduce, tmp_divide

    def _untransform_data(self, data, reduce=None, divide=None):
        """ Untransform data -> revert _transform_data"""
        tmp_reduce, tmp_divide = reduce, divide
        untransf_data = data * tmp_divide + tmp_reduce
        return untransf_data

    def _save_model(self):
        filename = osp.join(self.save_path, self.model_name + ".json")
        """Save a strategy to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.output_models, f)

    def _load_model(self, model_name):
        filename = osp.join(self.save_path, model_name + ".json")
        """Load a strategy from a JSON file"""
        with open(filename, "r") as f:
            output_model = json.load(f)
        return output_model

    def _check_file_path(self, file_path):
        """ Check whether a file with the path <file_path> already exist. If yes, it asks the user, whether the existing file should be overwritten. """
        if osp.isfile(file_path):
            print("Warning: The file {} already exist.".format(file_path))
            valid_input = False
            while not valid_input:
                tmp_input = str(input("Do you want to overwrite this file? If yes, type \'y\' or \'yes\', else type \'n\' or \'no\': "))
                if tmp_input in ["y", "yes"]:
                    overwrite = True
                    valid_input = True
                elif tmp_input in ["n", "no"]:
                    overwrite = False
                    valid_input = True
            if not overwrite:
                return False
        return True


    def create_parity_plot(self, datasets_real=None, datasets_pred=None, **kwargs):
        """  Make a parity plot of the training and test dataset

        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to
        y_pred: np-array, optional
            Prediction values for y.
        y_real: np-array, optional (required if y_pred != None)
            Real values for y.

        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis

        if ax is a matplotlib axis, returns only the axis

        Raises
        ------
        ValueError
            If there are no points to plot
        """
        if datasets_pred == None or datasets_real == None:
            raise ValueError("No points to plot.")
        if (len(datasets_pred) != len(datasets_real)):
            raise ValueError("Number of datasets with real points does not correspond to number of datasets with prediction points.")


        ax = kwargs.get("ax", None)

        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False

        marker_symbols = ["o", "x", "s", "p", "h", "+", "8"]
        for i in range(len(datasets_pred)):
            y_pred, y_real = datasets_pred[i], datasets_real[i]
            if len(y_pred) != len(y_real):
                raise ValueError("Number of real data points does not correspond to number of prediction data points.")
            marker_symbol = marker_symbols[i] if i < len(marker_symbols) else marker_symbols[0]
            ax.scatter(np.asarray(y_real), np.asarray(y_pred), marker=marker_symbol)

        ax.set_xlabel("Experimental y", fontsize=16)
        ax.set_ylabel("Predicted y", fontsize=16)

        if return_fig:
            return fig, ax
        else:
            return ax