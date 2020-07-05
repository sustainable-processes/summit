import os
import os.path as osp

import numpy as np

from summit.benchmarks.experiment_emulator.emulator import Emulator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.metrics import r2_score

# =======================================================================

class BNNEmulator(Emulator):
    """ BNN Emulator

    A Bayesian Neural Network (BNN) emulator.


    """

# =======================================================================

    def __init__(self, domain, dataset, model_name, kwargs={}):
        self._domain = domain
        self._domain_preprocess()

        self._dataset = dataset

        regression_model = self._setup_model()
        super().__init__(regression_model)

        # Set model name for saving
        self.save_path = kwargs.get("save_path", osp.join(osp.dirname(osp.realpath(__file__)), "trained_models/BNN"))
        self.model_name = str(model_name)

        # Set up training hyperparameters
        self.set_training_hyperparameters()

# =======================================================================

    def _setup_model(self, **kwargs):
        """ Setup the BNN model """

        @variational_estimator
        class BayesianRegressor(nn.Module):
            def __init__(self, input_dim):
                super().__init__()

                self.blinear1 = BayesianLinear(input_dim, 24)
                self.blinear2 = BayesianLinear(24, 24)
                self.blinear3 = BayesianLinear(24, 24)
                self.blinear4 = BayesianLinear(24, 1)

            def forward(self, x):
                x = F.leaky_relu(self.blinear1(x))
                x = F.leaky_relu(self.blinear2(x))
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.leaky_relu(self.blinear3(x))
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(self.blinear4(x))
                y = x
                return y.view(-1)

            # Training of model on given dataloader
            def _train(self, regressor, device, optimizer, criterion, X_train, loader):
                regressor.train()

                for i, (datapoints, labels) in enumerate(loader):
                    optimizer.zero_grad()
                    loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                                 labels=labels.to(device),
                                                 criterion=criterion,
                                                 sample_nbr=3,
                                                 complexity_cost_weight=1 / X_train.shape[0])
                    loss.backward()
                    optimizer.step()

            # Evaluate model for given dataloader
            def _evaluate_regression(self, regressor, device, loader, fun_untransform_data, out_transform, get_predictions=False):
                regressor.eval()
                regressor.freeze_()

                mae = 0
                pred_data = []
                real_data = []
                for i, (datapoints, labels) in enumerate(loader):
                    data = datapoints.to(device)
                    pred = regressor(data)
                    tmp_pred_data = fun_untransform_data(data=pred, reduce=out_transform[0], divide=out_transform[1])
                    tmp_real_data = fun_untransform_data(data=labels, reduce=out_transform[0], divide=out_transform[1])
                    mae += (tmp_pred_data - tmp_real_data).abs().sum(0).item()

                    if get_predictions:
                        pred_data.extend(tmp_pred_data.tolist())
                        real_data.extend(tmp_real_data.tolist())

                if get_predictions:
                    return pred_data, real_data

                regressor.unfreeze_()

                return mae / len(loader.dataset)

        regression_model = BayesianRegressor(self.input_dim)
        return regression_model

# =======================================================================

    def set_training_hyperparameters(self, kwargs={}):
        # Setter method for hyperparameters of training
        self.epochs = kwargs.get("epochs", 300) # number of max epochs the model is trained
        self.initial_lr = kwargs.get("initial_lr", 0.001)  # initial learning rate
        self.min_lr = kwargs.get("min_lr", 0.00001)
        self.lr_decay = kwargs.get("lr_decay", 0.7)  # learning rate decay
        self.lr_decay_patience = kwargs.get("lr_decay_patience", 3) # number of epochs before learning rate is reduced by lr_decay
        self.early_stopping_epochs = kwargs.get("early_stopping_epochs", 30)  # number of epochs before early stopping
        self.batch_size_train = kwargs.get("batch_size_train", 4)
        self.transform_input = kwargs.get("transform_input", "standardize")
        self.transform_output = kwargs.get("transform_output", "standardize")
        self.test_size = kwargs.get("test_size", 0.1)
        self.shuffle = kwargs.get("shuffle", False)

# =======================================================================

    def train_model(
            self, dataset=None, verbose=True, kwargs={}
    ):
        torch.set_printoptions(precision=10)
        # Manual call of training -> overwrite dataset with new dataset for training
        if dataset is not None:
            self._dataset = dataset

        # Data preprocess
        train_dataset, test_dataset = \
            self._data_preprocess(transform_input=self.transform_input, transform_output=self.transform_output, test_size=self.test_size, shuffle=self.shuffle)

        X_train, y_train = torch.tensor(train_dataset[0]).float(), torch.tensor(train_dataset[1]).float()
        X_test, y_test = torch.tensor(test_dataset[0]).float(), torch.tensor(test_dataset[1]).float()

        if verbose:
            print("\n<---- Start training of BNN model ---->")
            print("  --- Length of train dataset: {} ---".format(X_train.shape[0]))
            print("  --- Length of test dataset: {} ---".format(X_test.shape[0]))
        for i, k in enumerate(self.output_models):
            if verbose:
                print("\n  <-- Start training of BNN regressor on objective: {} -->\n".format(k))

            # Set training details
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            regressor = self._setup_model().to(device)
            optimizer = optim.Adam(regressor.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.lr_decay, patience=self.lr_decay_patience, min_lr=self.min_lr)
            criterion = torch.nn.MSELoss()
            model_save_name = self.model_name + "_" + str(k) + "_BNN_model.pt"
            model_save_dir = osp.join(self.save_path, model_save_name)
            storable = self._check_file_path(model_save_dir)
            if not storable:
                self.output_models[k] = self._load_model(self.model_name)[k]
                continue

            out_transform = self.data_transformation_dict[k]
            y_train_obj, y_test_obj = y_train[:, i], y_test[:, i]
            ds_train = torch.utils.data.TensorDataset(X_train, y_train_obj)
            dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=self.batch_size_train, shuffle=True)
            ds_test = torch.utils.data.TensorDataset(X_test, y_test_obj)
            dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)

            max_iter_stop = self.early_stopping_epochs  # maximum number of consecutive iteration w/o improvement after which training is stopped
            tmp_iter_stop = 0
            best_train_mae = float("inf")
            for epoch in range(self.epochs):

                lr = scheduler.optimizer.param_groups[0]["lr"]

                # train model
                self._model._train(regressor, device, optimizer, criterion, X_train, dataloader_train)

                # TODO: define stopping criterion! To use training mae is not the best way to do this (-> overfitting, usually we have an extra validation set, problem: small dataset size, cross-validation is exhaustive at this point)
                train_mae = self._model._evaluate_regression(regressor, device, dataloader_train, self._untransform_data, out_transform)
                scheduler.step(train_mae)

                if verbose:
                    print("   -- Epoch: {:03d}, LR: {:7f}, Train MAE: {:4f}".format(epoch, lr, train_mae))

                # if prediction accuracy was improved in current epoch, reset <tmp_iter_stop> and save model
                if best_train_mae > train_mae:
                    best_train_mae = train_mae
                    tmp_iter_stop = 0
                    torch.save(regressor.state_dict(), model_save_dir)
                    if verbose:
                        test_mae = self._model._evaluate_regression(regressor, device, dataloader_test, self._untransform_data, out_transform)
                        print("      -> Train MAE improved, current Test MAE: {:4f}".format(test_mae))
                # if prediction accuracy was not imporved in current epoch, increase <tmp_iter_stop> and stop training if <max_iter_stop> is reached
                else:
                    tmp_iter_stop += 1
                    if tmp_iter_stop >= max_iter_stop:
                        break

            # load final model from epoch with lowest prediction accuracy
            regressor.load_state_dict(torch.load(model_save_dir))
            # freeze the model, in order to predict using only their weight distribution means
            regressor.freeze_()
            # get final model predictions for training and test data
            final_train_mae = self._model._evaluate_regression(regressor=regressor, device=device, loader=dataloader_train, fun_untransform_data=self._untransform_data,
                                                        out_transform=out_transform)
            final_test_mae = self._model._evaluate_regression(regressor=regressor, device=device, loader=dataloader_test, fun_untransform_data=self._untransform_data,
                                                        out_transform=out_transform)
            y_train_pred, y_train_real = self._model._evaluate_regression(regressor=regressor, device=device, loader=torch.utils.data.DataLoader(ds_train, shuffle=False), fun_untransform_data=self._untransform_data,
                                                        out_transform=out_transform, get_predictions=True)
            y_test_pred, y_test_real = self._model._evaluate_regression(regressor=regressor, device=device, loader=torch.utils.data.DataLoader(ds_test, shuffle=False), fun_untransform_data=self._untransform_data,
                                                        out_transform=out_transform, get_predictions=True)
            self.output_models[k] = {"model_save_dir": model_save_name,
                                     "data_transformation_dict": self.data_transformation_dict,
                                     "X_train": X_train.tolist(), "y_train_real": y_train_real, "y_train_pred": y_train_pred,
                                     "X_test": X_test.tolist(), "y_test_real": y_test_real, "y_test_pred": y_test_pred}

            if verbose:
                print("\n  <-- Finished training of BNN model on objective: {} -->\n"
                      "   -- Final Train MAE: {:4f}, Final Test MAE: {:4f} --\n"
                      "   -- Model saved at: {} --\n".format(k, final_train_mae, final_test_mae, model_save_dir))

        self._save_model()

        if verbose:
            print("<---- End training of BNN regressor ---->\n")

# =======================================================================

    def validate_model(self, dataset=None, parity_plots=False, kwargs={}):
        self.output_models = self._load_model(self.model_name)

        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means
        self._model.eval()  # set to evaluation mode (may be redundant)

        val_dict = {}
        lst_parity_plots=None
        if parity_plots:
            lst_parity_plots = []

        if dataset is not None:
            for i, (k, v) in enumerate(self.output_models.items()):
                model_load_dir = osp.join(self.save_path, v["model_save_dir"])
                self.data_transformation_dict = v["data_transformation_dict"]
                out_transform = self.data_transformation_dict[k]

                X_val = self._data_preprocess(inference=True, infer_dataset=dataset, validate=True)
                X_val = torch.tensor(X_val).float()
                y_val = torch.tensor(dataset[(k, "DATA")].to_numpy()).float()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model.load_state_dict(torch.load(model_load_dir, map_location=torch.device(device)))
                data = X_val.to(device)
                predictions = self._model(data).detach()
                predictions = self._untransform_data(data=predictions, reduce=out_transform[0], divide=out_transform[1])
                val_dict[k] = {"MAE": (predictions - y_val).abs().mean().item(),
                               "RMSE": ((((predictions - y_val)**2).mean())**(1/2)).item(),
                               "r2": r2_score(y_val, predictions)}

                if parity_plots:
                    parity_plot = self.create_parity_plot(datasets_pred=[predictions], datasets_real=[y_val], kwargs=kwargs)
                    lst_parity_plots.append(parity_plot)
        else:
            for i, (k, v) in enumerate(self.output_models.items()):
                y_train_real, y_train_pred, y_test_real, y_test_pred = \
                    torch.tensor(v["y_train_real"]).float(), torch.tensor(v["y_train_pred"]).float(), \
                    torch.tensor(v["y_test_real"]).float(), torch.tensor(v["y_test_pred"]).float()
                val_dict[k] = {"Train":
                                   {"MAE": (y_train_real - y_train_pred).abs().mean().item(),
                                    "RMSE": ((((y_train_real - y_train_pred) ** 2).mean()) ** (1 / 2)).item(),
                                    "r2": r2_score(y_train_real, y_train_pred)},
                               "Test":
                                   {"MAE": (y_test_real - y_test_pred).abs().mean().item(),
                                    "RMSE": ((((y_test_real - y_test_pred) ** 2).mean()) ** (1 / 2)).item(),
                                    "r2": r2_score(y_test_real, y_test_pred)}
                               }
                if parity_plots:
                    parity_plot = self.create_parity_plot(datasets_pred=[y_train_pred, y_test_pred], datasets_real=[y_train_real, y_test_real], kwargs=kwargs)
                    lst_parity_plots.append(parity_plot)

        return val_dict, lst_parity_plots

# =======================================================================

    def infer_model(self, dataset):

        self.output_models = self._load_model(self.model_name)

        self._model.eval()  # set to evaluation mode (may be redundant)
        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means

        infer_dict = {}
        for i, (k, v) in enumerate(self.output_models.items()):
            model_load_dir = osp.join(self.save_path, v["model_save_dir"])
            self.data_transformation_dict = v["data_transformation_dict"]
            out_transform = self.data_transformation_dict[k]

            X_infer = self._data_preprocess(inference=True, infer_dataset=dataset)
            X_infer = torch.tensor(X_infer).float()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.load_state_dict(torch.load(model_load_dir, map_location=torch.device(device)))
            data = X_infer.to(device)
            predictions = self._model(data).item()
            predictions = self._untransform_data(data=predictions, reduce=out_transform[0], divide=out_transform[1])
            infer_dict[k] = predictions

        return infer_dict