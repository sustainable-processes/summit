import os
import os.path as osp

from summit.strategies.base import Transform
from summit.experiment import Experiment
from summit.domain import *
from summit.utils.dataset import DataSet
from summit.benchmarks.experiment_emulator import experimental_datasets

import numpy as np
from scipy.integrate import solve_ivp

import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


class ReizmanSuzukiEmulator(Experiment):
    """ Reizman Suzuki Emulator

    Virtual experiments representing the Suzuki-Miyaura Cross-Coupling reaction
    similar to Reizman et al. (2016). Experimental outcomes are based on an
    emulator that is trained on the experimental data published by Reizman et al. 
    
    Examples
    --------
    >>> b = ReizmanSuzukiEmulator()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.6*(v.bounds[1]-v.bounds[0]) if v.variable_type == 'continuous' else v.levels[-1] for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)

    
    Notes
    -----
    This benchmark is based on Reizman et al. React. Chem. Eng. 2016. 
    https://doi.org/10.1039/C6RE00153J
    
    """

    def __init__(self, model_name="reizman_suzuki_case1", **kwargs):
        domain = self._setup_domain()
        super().__init__(domain)

        self.model_name = model_name
        self.emulator_type = kwargs.get("emulator_type", "BNN")

    def _setup_domain(self):
        domain = Domain()

        # Decision variables
        des_1 = "Catalyst type - different ligands"
        domain += DiscreteVariable(
            name="catalyst", description=des_1, levels=["P1-L1", "P2-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7"])

        des_2 = "Residence time in seconds (s)"
        domain += ContinuousVariable(
            name="t_res", description=des_2, bounds=[60, 600]
        )

        des_3 = "Reactor temperature in degrees Celsius (ºC)"
        domain += ContinuousVariable(
            name="temperature", description=des_3, bounds=[30, 110]
        )

        des_4 = "Catalyst loading in mol%"
        domain += ContinuousVariable(
            name="catalyst_loading", description=des_4, bounds=[0.5, 2.5]
        )

        # Objectives
        des_5 = "Turnover number - moles product generated divided by moles catalyst used"
        domain += ContinuousVariable(
            name="ton",
            description=des_5,
            bounds=[0, 200],   # TODO: not sure about bounds, maybe redefine
            is_objective=True,
            maximize=True,
        )

        des_6 = "Yield"
        domain += ContinuousVariable(
            name="yield",
            description=des_6,
            bounds=[0, 100],
            is_objective=True,
            maximize=True,
        )

        return domain

    def _run(self, conditions, **kwargs):
        catalyst = str(conditions["catalyst"].iloc[0])
        t_res = float(conditions["t_res"])
        temperature = float(conditions["temperature"])
        catalyst_loading = float(conditions["catalyst_loading"])
        X = self._convert_inputs(catalyst, t_res, temperature, catalyst_loading)
        pred_ton, pred_yield = self._emulator_predict(X)
        conditions[("ton", "DATA")] = pred_ton
        conditions[("yield", "DATA")] = pred_yield
        return conditions, None

    def _get_original_data(self):
        self.X, self.y = experimental_datasets.load_reizman_suzuki(return_X_y=True, case=int(self.model_name[-1]))

    def _convert_inputs(self, catalyst, t_res, temperature, catalyst_loading):
        self._get_original_data()

        # convert categorical variables to one-hot tensors
        tmp_ligand_type = np.asarray([int(str(catalyst[1])+str(catalyst[4]))], dtype=np.int32)   # categorical input: convert string of ligand type into numerical unique identifier (i.e. hash value)
        tmp_ligand_type = torch.tensor(tmp_ligand_type).int()

        ## get ligand types of real experimental data in order to get number of classes the model is trained on (make sure ligand type of new point was in original data, otherwise this will lead to false predictions)
        origin_data_ligand_type = torch.tensor(self.X[:,0]).int()
        
        ## unify ligand type of current point and ligand types of original data, in order to create one-hot tensor
        _uni_ligand_types = torch.cat((tmp_ligand_type, origin_data_ligand_type), axis=0)
        tmp_ligand_type = torch.unique(_uni_ligand_types, True, True)[1][0].view(-1)

        ## get number of different ligand types and the corresponding unique values (the identifiers - hash values)
        origin_data_ligand_type = torch.unique(origin_data_ligand_type, True, True)[1]
        num_types = int(origin_data_ligand_type.max().item() + 1)

        ## generate one-hot vector
        tmp_ligand_type = F.one_hot(tmp_ligand_type, num_classes=num_types).to(torch.float)


        # standardize continuous input variables
        tmp_inp_cont = torch.tensor([t_res, temperature, catalyst_loading]).float().resize_((1, 3))
        
        ## get mean and std of real experiment data to get the numbers the model is trained on
        original_data_inp_cont = torch.tensor(self.X[:,1:]).float()
        original_data_inp_mean = original_data_inp_cont.mean(axis=0)
        original_data_inp_std = original_data_inp_cont.std(axis=0)

        ## standardize inputs
        tmp_inp_cont = (tmp_inp_cont - original_data_inp_mean) / original_data_inp_std 

        # X - input: concatenate one-hot caterogical variables and continuous variables
        X = torch.cat((tmp_ligand_type, tmp_inp_cont), axis=1)
        self.tmp_inp_dim = X.shape[1]

        # get mean of real experiment target property data the model is trained on
        original_data_y = torch.tensor(self.y).float()
        self.original_data_out_mean = original_data_y.mean(axis=0)
        return X

    def _setup_emulator_structure(self):
        # make sure this structure corresponds to the structure of the trained model
        @variational_estimator
        class BayesianRegressor(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()

                self.blinear1 = BayesianLinear(input_dim, 24)
                self.blinear2 = BayesianLinear(24, 24)
                self.blinear3 = BayesianLinear(24, output_dim)
        
            def forward(self, x):
                x = F.leaky_relu(self.blinear1(x))
                x = F.leaky_relu(self.blinear2(x))
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(self.blinear3(x))
                y = x
                return y.view(-1)

        return BayesianRegressor


    def _emulator_predict(self, X):
        # Initiliaze emulator
        Emulator = self._setup_emulator_structure()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emulator = Emulator(self.tmp_inp_dim, 1).to(device)
        emulator.eval()   # set to evaluation mode (may be redundant)
        emulator.freeze_()   # freeze the model, in order to predict using only their weight distribution means
        
        # TON prediction
        ton_path = osp.join(osp.dirname(osp.realpath(__file__)), "experiment_emulator/trained_models", self.emulator_type, self.model_name + "_TON_BNN_model.pt")
        emulator.load_state_dict(torch.load(ton_path, map_location=torch.device(device)))
        data = X.to(device)
        ## model predicts average data, so multiply with average
        pred_ton = float(emulator(data).item() * self.original_data_out_mean[0].item())

        # Yield prediction
        yield_path = osp.join(osp.dirname(osp.realpath(__file__)), "experiment_emulator/trained_models", self.emulator_type, self.model_name + "_yield_BNN_model.pt")
        emulator.load_state_dict(torch.load(yield_path, map_location=torch.device(device)))
        emulator.eval()
        data = X.to(device)
        ## model predicts average data, so multiply with average
        pred_yield = float(emulator(data).item() * self.original_data_out_mean[1].item())
    
        return pred_ton, pred_yield


