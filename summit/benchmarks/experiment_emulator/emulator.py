from abc import ABC, abstractmethod
from summit.domain import Domain
from summit.utils.dataset import DataSet

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split as sklearn_train_test_split
import sklearn.preprocessing

from experimental_datasets import load_reizman_suzuki
#=======================================================================
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

    def __init__(self, train_dataset, test_dataset, model, **kwargs):
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
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

        raise NotImplementedError("_steup_model be implemented by subclasses of EmulatorTrainer")

    @abstractmethod
    def train_model(self):
        """ Train model on a given Summit Dataset.

        Arguments
        ---------

        Returns
        -------
        model
            Should return a regression model that is trained on experimental data.
        """

        raise NotImplementedError("_train_model be implemented by subclasses of EmulatorTrainer")


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

        raise NotImplementedError("_validate_model be implemented by subclasses of EmulatorTrainer")

    @abstractmethod
    def infer_model(self):
        raise NotImplementedError("_infer_model be implemented by subclasses of EmulatorTrainer")

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


    def _data_preprocess(self, inference=False, infer_dataset=None):

        if not inference:
            np_dataset = self._dataset.data_to_numpy()
            data_column_names = self._dataset.data_columns
        else:
            np_dataset = infer_dataset.data_to_numpy()
            data_column_names = [c[0] for c in infer_dataset.data_columns]

        self.input_data_continuous = []
        self.input_data_discrete = []
        self.output_data = []
        if not inference:
            self.out_mean = np.asarray([])

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
                            tmp_cont_inp = np_dataset[:, i]
                            if False:
                                tmp_cont_inp = sklearn.preprocessing.scale(tmp_cont_inp)
                            self.input_data_continuous.append(tmp_cont_inp)
                        elif v.variable_type == "discrete":
                            # create one-hot tensor for discrete inputs
                            one_hot_enc = sklearn.preprocessing.OneHotEncoder(categories = [v.levels])
                            tmp_disc_inp_one_hot = one_hot_enc.fit_transform(np_dataset[:, i].reshape(-1,1)).toarray()
                            self.input_data_discrete.append(np.asarray(tmp_disc_inp_one_hot))
                        elif v.variable_type == "descriptors":
                            raise TypeError(
                                "Regressor not trainable on descriptor variables. Please redefine {} as continuous or discrete variable".format(
                                    v.name))
                        else:
                            raise TypeError("Unknown variable type: {}.".format(v.variable_type))
                    elif not inference:
                        if v.variable_type == "continuous":
                            self.out_mean = np.concatenate((self.out_mean, np.asarray([1])))
                            self.output_data.append(np_dataset[:, i])
                        elif v.variable_type == "discrete":
                            raise TypeError(
                                "{} is a discrete variable. Regressor not trainable for discrete outputs.".format(v.name))
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
        if not inference:
            final_np_dataset = np.concatenate([self.input_data_continuous, self.input_data_discrete, self.output_data], axis=1)
            X, y = final_np_dataset[:, :-self.output_dim], final_np_dataset[:, -self.output_dim:]
            X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size = 0.1, shuffle=False)
            return [X_train.astype(dtype=float), y_train.astype(dtype=float)], [X_test.astype(dtype=float), y_test.astype(dtype=float)]
        else:
            X = np.concatenate([self.input_data_continuous, self.input_data_discrete], axis=1)
            return X.astype(dtype=float)