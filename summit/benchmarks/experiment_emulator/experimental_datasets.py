"""
This code follows scikit-learn: https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/datasets/_base.py#L948
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
import os
import csv
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
import os.path as osp

import numpy as np


def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.
    Parameters
    ----------
    module_path : string
        The module path.
    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example "wine_data.csv".
    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.
    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].
    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(osp.join(module_path, "data", data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names


def load_reizman_suzuki(*, return_X_y=False, case=1):
    """Load and return the suzuki dataset (regression).
    ==============   ===============
    Samples total            96 (97) 
    Dimensionality                4
    Features         real, positive
    Targets                  real 2
    ==============   ===============
    Parameters
    ----------
    return_X_y : bool, optional, default=False
        If True, returns ``(data, target)`` instead of a `data` dict object.
        See below for more information about the `data` and `target` object.
    case: int, optional, default=1
        Reizman et al. (2016) reported experimental data for 4 different
        cases. The case number refers to the cases they reported.
        Please see their paper for more information on the cases.
    Returns
    -------
    data : Dictionary-like object, with the following attributes.
        data : ndarray of shape (samples total, dimensionality)
            The data matrix.
        target : ndarray of shape (samples total, targets)
            The regression targets - TON and yield.
        feature_names : ndarray
            The names of features
        filename : str
            The physical location of reizman_suzuki csv dataset.
        DESCR : str
            The full description of the dataset.
    Dataset: Summit Dataset
    Domain: Summit Domain
    (data, target) : tuple if ``return_X_y`` is True
    Examples
    --------
    >>> from experimental_datasets import load_reizman_suzuki
    >>> X, y = load_reizman_suzuki(return_X_y=True, case=1)
    >>> print(X.shape)
    (96, 4)
    """
    module_path = osp.dirname(osp.realpath(__file__))

    def get_case_file(case):
        return {
           1: "reizman_suzuki_case1_train_test.csv",
           2: "reizman_suzuki_case2_train_test.csv",
           3: "reizman_suzuki_case3_train_test.csv",
           4: "reizman_suzuki_case4_train_test.csv",
       }.get(case, "reizman_suzuki_case1_train_test.csv")

    data_file_case = get_case_file(case)
    data_file_name = osp.join(module_path, "data", data_file_case)

    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        descr = str(temp[2])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,2))
        temp = next(data_file)
        # only return first word as variable name to avoid units or similar in variable names
        #for t in range(len(temp)):
        #    tmp[t] = tmp[t].strip.split(" ", 1)[0]
        #feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            ligand_type = np.asarray([int(str(d[0])[1]+str(d[0])[4])], dtype=np.int32)   # categorical input: convert string of ligand type into numerical identifier
            cont_data = np.asarray(d[1:-2], dtype=np.float64)    # continuous input
            data[i] = np.concatenate((ligand_type, cont_data), axis=0)
            target[i] = np.asarray(d[-2:], dtype=np.float64)

    if return_X_y:
        return data, target

    data_dict = {
            "data": data,
            "target": target,
            "feature_names": feature_names,
            "filename": data_file_name,
            "DESCR": descr
            }

    # TODO: returns a Summit dataset and a Domain

    return data_dict


