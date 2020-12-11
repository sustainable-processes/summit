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
    """BNN Emulator

    A Bayesian Neural Network (BNN) emulator.

    Parameters
    ---------
    domain: summit.domain.Domain
        The domain of the experiment
    dataset: class:~summit.utils.dataset.DataSet, optional
        A DataSet with data for training where the data columns correspond to the domain and the data rows correspond to the training points.
        By default: None
    model_name: string, optional
        Name of the model that is used for saving model parameters. Should be unique.
        By default: "dataset_emulator_model_name"
    """

    # =======================================================================

    def __init__(self, domain, dataset, model_name, kwargs={}):
        super().__init__(domain, dataset, model_name, kwargs)
        self._model = self._setup_model()

        # Set model name for saving
        self.save_path = kwargs.get(
            "save_path",
            osp.join(osp.dirname(osp.realpath(__file__)), "trained_models/BNN"),
        )

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
                    loss = regressor.sample_elbo(
                        inputs=datapoints.to(device),
                        labels=labels.to(device),
                        criterion=criterion,
                        sample_nbr=3,
                        complexity_cost_weight=1 / X_train.shape[0],
                    )
                    loss.backward()
                    optimizer.step()

            # Evaluate model for given dataloader
            def _evaluate_regression(
                self,
                regressor,
                device,
                loader,
                fun_untransform_data,
                out_transform,
                get_predictions=False,
            ):
                regressor.eval()
                regressor.freeze_()

                mae = 0
                pred_data = []
                real_data = []
                for i, (datapoints, labels) in enumerate(loader):
                    data = datapoints.to(device)
                    pred = regressor(data)
                    tmp_pred_data = fun_untransform_data(
                        data=pred, reduce=out_transform[0], divide=out_transform[1]
                    )
                    tmp_real_data = fun_untransform_data(
                        data=labels, reduce=out_transform[0], divide=out_transform[1]
                    )
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
        self.epochs = kwargs.get(
            "epochs", 300
        )  # number of max epochs the model is trained
        self.initial_lr = kwargs.get("initial_lr", 0.001)  # initial learning rate
        self.min_lr = kwargs.get("min_lr", 0.00001)
        self.lr_decay = kwargs.get("lr_decay", 0.7)  # learning rate decay
        self.lr_decay_patience = kwargs.get(
            "lr_decay_patience", 3
        )  # number of epochs before learning rate is reduced by lr_decay
        self.early_stopping_epochs = kwargs.get(
            "early_stopping_epochs", 30
        )  # number of epochs before early stopping
        self.batch_size_train = kwargs.get("batch_size_train", 4)
        self.transform_input = kwargs.get("transform_input", "standardize")
        self.transform_output = kwargs.get("transform_output", "standardize")
        self.test_size = kwargs.get("test_size", 0.1)
        self.shuffle = kwargs.get("shuffle", False)

    # =======================================================================

    def train_model(self, dataset=None, verbose=True, kwargs={}):
        # Manual call of training -> overwrite dataset with new dataset for training
        if dataset is not None:
            self._dataset = dataset

        # #<cv_fold>-fold cross-validation
        cv_fold = kwargs.get("cv_fold", 10)

        # Data preprocess
        train_dataset, test_dataset = self._data_preprocess(
            transform_input=self.transform_input,
            transform_output=self.transform_output,
            test_size=self.test_size,
            shuffle=self.shuffle,
        )

        X_train_init, y_train_init = (
            torch.tensor(train_dataset[0]).float(),
            torch.tensor(train_dataset[1]).float(),
        )
        X_test, y_test = (
            torch.tensor(test_dataset[0]).float(),
            torch.tensor(test_dataset[1]).float(),
        )

        shuffle_train = kwargs.get("shuffle_train", False)
        if shuffle_train:
            perm = torch.randperm(len(y_train_init))
            train_data = torch.cat([X_train_init, y_train_init], axis=1)[perm]
            X_train, y_train = (
                train_data[:, : -self.output_dim],
                train_data[:, -self.output_dim :],
            )
        else:
            X_train, y_train = X_train_init, y_train_init

        if verbose:
            print("\n<---- Start training of BNN model ---->")
            print("  --- Length of train dataset: {} ---".format(X_train.shape[0]))
            print("  --- Length of test dataset: {} ---".format(X_test.shape[0]))
        for i, k in enumerate(self.output_models):
            if verbose:
                print(
                    "\n  <-- Start {}-fold cross-validation training of BNN regressor on objective: {} -->\n".format(
                        cv_fold, k
                    )
                )

            train_acc, val_acc, test_acc = [], [], []
            y_train_pred_l, y_train_real_l, y_test_pred_l, y_test_real_l = (
                [],
                [],
                [],
                [],
            )
            for j in range(cv_fold):
                if verbose:
                    print("  ---------------- Split {} ----------------".format(j + 1))

                # Set training details
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                regressor = self._setup_model().to(device)
                optimizer = optim.Adam(regressor.parameters(), lr=self.initial_lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.lr_decay,
                    patience=self.lr_decay_patience,
                    min_lr=self.min_lr,
                )
                criterion = torch.nn.MSELoss()
                model_save_name = (
                    self.model_name + "_" + str(k) + "_" + str(j + 1) + "_BNN_model.pt"
                )
                model_save_dir = osp.join(self.save_path, model_save_name)
                storable = self._check_file_path(model_save_dir)
                if not storable:
                    self.output_models[k] = self._load_model(self.model_name)[k]
                    continue

                # Setup train and val dataset for cross-validation
                if cv_fold <= 1:
                    raise ValueError(
                        "{}-fold Cross-Validation not possible. Increase cv_fold.".format(
                            cv_fold
                        )
                    )
                if len(X_train) < cv_fold:
                    raise ValueError(
                        "Too few data points ({}) for training provided. Decrease cv_fold.".format(
                            len(X_train)
                        )
                    )
                n = len(X_train) // cv_fold
                r = len(X_train) % cv_fold
                val_mask = torch.zeros(len(X_train), dtype=torch.uint8)
                # make sure every data point is included in the validation set once
                if j < r:
                    val_mask[j * (n + 1) : (j + 1) * (n + 1)] = 1
                else:
                    val_mask[j * n + r : (j + 1) * n + r] = 1
                X_val_cv, y_val_cv = X_train[val_mask], y_train[val_mask]
                X_train_cv, y_train_cv = X_train[1 - val_mask], y_train[1 - val_mask]

                out_transform = self.data_transformation_dict[k]
                y_train_obj, y_val_obj, y_test_obj = (
                    y_train_cv[:, i],
                    y_val_cv[:, i],
                    y_test[:, i],
                )
                ds_train = torch.utils.data.TensorDataset(X_train_cv, y_train_obj)
                dataloader_train = torch.utils.data.DataLoader(
                    ds_train, batch_size=self.batch_size_train, shuffle=True
                )
                ds_val = torch.utils.data.TensorDataset(X_val_cv, y_val_obj)
                dataloader_val = torch.utils.data.DataLoader(
                    ds_val, batch_size=16, shuffle=False
                )
                ds_test = torch.utils.data.TensorDataset(X_test, y_test_obj)
                dataloader_test = torch.utils.data.DataLoader(
                    ds_test, batch_size=16, shuffle=False
                )

                max_iter_stop = (
                    self.early_stopping_epochs
                )  # maximum number of consecutive iteration w/o improvement after which training is stopped
                tmp_iter_stop = 0
                best_train_mae, best_val_mae, best_test_mae = (
                    float("inf"),
                    float("inf"),
                    float("inf"),
                )
                for epoch in range(self.epochs):

                    lr = scheduler.optimizer.param_groups[0]["lr"]

                    # train model
                    self._model._train(
                        regressor,
                        device,
                        optimizer,
                        criterion,
                        X_train_cv,
                        dataloader_train,
                    )

                    train_mae = self._model._evaluate_regression(
                        regressor,
                        device,
                        dataloader_train,
                        self._untransform_data,
                        out_transform,
                    )
                    val_mae = self._model._evaluate_regression(
                        regressor,
                        device,
                        dataloader_val,
                        self._untransform_data,
                        out_transform,
                    )
                    scheduler.step(val_mae)

                    if verbose:
                        print(
                            "   -- Epoch: {:03d}, LR: {:7f}, Train MAE: {:4f}, Val MAE: {:4f}".format(
                                epoch, lr, train_mae, val_mae
                            )
                        )

                    # if prediction accuracy was improved in current epoch, reset <tmp_iter_stop> and save model
                    if best_val_mae > val_mae:
                        best_val_mae = val_mae
                        tmp_iter_stop = 0
                        torch.save(regressor.state_dict(), model_save_dir)
                        test_mae = self._model._evaluate_regression(
                            regressor,
                            device,
                            dataloader_test,
                            self._untransform_data,
                            out_transform,
                        )
                        best_train_mae, best_test_mae = train_mae, test_mae
                        if verbose:
                            print(
                                "      -> Val MAE improved, current Test MAE: {:4f}".format(
                                    test_mae
                                )
                            )
                    # if prediction accuracy was not imporved in current epoch, increase <tmp_iter_stop> and stop training if <max_iter_stop> is reached
                    else:
                        tmp_iter_stop += 1
                        if tmp_iter_stop >= max_iter_stop:
                            break

                train_acc.append(best_train_mae)
                val_acc.append(best_val_mae)
                test_acc.append(best_test_mae)

                y_train_obj = y_train_init[:, i]
                ds_train_all = torch.utils.data.TensorDataset(X_train_init, y_train_obj)

                # load final model from epoch with lowest prediction accuracy
                regressor.load_state_dict(torch.load(model_save_dir))

                # get final model predictions for training and test data
                y_train_pred, y_train_real = self._model._evaluate_regression(
                    regressor=regressor,
                    device=device,
                    loader=torch.utils.data.DataLoader(ds_train_all, shuffle=False),
                    fun_untransform_data=self._untransform_data,
                    out_transform=out_transform,
                    get_predictions=True,
                )
                y_test_pred, y_test_real = self._model._evaluate_regression(
                    regressor=regressor,
                    device=device,
                    loader=torch.utils.data.DataLoader(ds_test, shuffle=False),
                    fun_untransform_data=self._untransform_data,
                    out_transform=out_transform,
                    get_predictions=True,
                )
                y_train_pred_l.append(y_train_pred), y_train_real_l.append(y_train_real)
                y_test_pred_l.append(y_test_pred), y_test_real_l.append(y_test_real)

            train_acc, val_acc, test_acc = (
                torch.tensor(train_acc),
                torch.tensor(val_acc),
                torch.tensor(test_acc),
            )
            y_train_pred_l, y_train_real_l, y_test_pred_l, y_test_real_l = (
                torch.tensor(y_train_pred_l),
                torch.tensor(y_train_real_l),
                torch.tensor(y_test_pred_l),
                torch.tensor(y_test_real_l),
            )

            X_train_final = np.asarray(X_train_init.tolist())
            X_test_final = np.asarray(X_test.tolist())
            for ind, inp_var in enumerate(self.input_names_transformable):
                tmp_inp_transform = self.data_transformation_dict[inp_var]
                X_train_final[:, ind] = self._untransform_data(
                    data=X_train_final[:, ind],
                    reduce=tmp_inp_transform[0],
                    divide=tmp_inp_transform[1],
                )
                X_test_final[:, ind] = self._untransform_data(
                    data=X_test_final[:, ind],
                    reduce=tmp_inp_transform[0],
                    divide=tmp_inp_transform[1],
                )

            self.output_models[k] = {
                "model_save_dirs": [
                    self.model_name + "_" + str(k) + "_" + str(j + 1)
                    for j in range(cv_fold)
                ],
                "Final train MAE": train_acc.mean().tolist(),
                "Final validation MAE": val_acc.mean().tolist(),
                "Final test MAE": test_acc.mean().tolist(),
                "data_transformation_dict": self.data_transformation_dict,
                "X variable names": self.input_names,
                "X_train": X_train_final.tolist(),
                "y_train_real": y_train_real_l.mean(axis=0).tolist(),
                "y_train_pred_average": y_train_pred_l.mean(axis=0).tolist(),
                "X_test": X_test_final.tolist(),
                "y_test_real": y_test_real_l.mean(axis=0).tolist(),
                "y_test_pred_average": y_test_pred_l.mean(axis=0).tolist(),
            }

            if verbose:
                print(
                    "\n  <-- Finished training of BNN model on objective: {} -->\n"
                    "   -- Final Train MAE: {:4f}, Final Val MAE: {:4f}, Final Test MAE: {:4f} --\n"
                    "   -- Model saved at: {} --\n".format(
                        k,
                        train_acc.mean(),
                        val_acc.mean(),
                        test_acc.mean(),
                        model_save_dir,
                    )
                )

        self._save_model()

        if verbose:
            print("<---- End training of BNN regressor ---->\n")

    # =======================================================================

    def validate_model(
        self, dataset=None, parity_plots=False, get_pred=False, kwargs={}
    ):
        self.output_models = self._load_model(self.model_name)

        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means
        self._model.eval()  # set to evaluation mode (may be redundant)

        val_dict = {}
        lst_parity_plots = None
        if parity_plots:
            lst_parity_plots = []

        if dataset is not None:
            for i, (k, v) in enumerate(self.output_models.items()):
                model_load_dirs = v["model_save_dirs"]
                self.data_transformation_dict = v["data_transformation_dict"]
                out_transform = self.data_transformation_dict[k]

                X_val = self._data_preprocess(
                    inference=True, infer_dataset=dataset, validate=True
                )
                X_val = torch.tensor(X_val).float()
                y_val = torch.tensor(dataset[(k, "DATA")].to_numpy()).float()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                prediction_l = []
                for m in model_load_dirs:
                    model_load_dir = osp.join(self.save_path, m + "_BNN_model.pt")
                    self._model.load_state_dict(
                        torch.load(model_load_dir, map_location=torch.device(device))
                    )
                    data = X_val.to(device)
                    predictions = self._model(data).detach()
                    predictions = self._untransform_data(
                        data=predictions,
                        reduce=out_transform[0],
                        divide=out_transform[1],
                    )
                    prediction_l.append(predictions)
                prediction_l = torch.tensor(prediction_l)
                predictions = prediction_l.mean(axis=0)
                val_dict[k] = {
                    "MAE": (predictions - y_val).abs().mean().item(),
                    "RMSE": ((((predictions - y_val) ** 2).mean()) ** (1 / 2)).item(),
                    "r2": r2_score(y_val, predictions)
                    if y_val.shape[0] > 1
                    else "Too few data points to calculate r2.",
                }

                if parity_plots:
                    parity_plot = self.create_parity_plot(
                        datasets_pred=[predictions],
                        datasets_real=[y_val],
                        kwargs=kwargs,
                    )
                    lst_parity_plots.append(parity_plot)
        else:
            for i, (k, v) in enumerate(self.output_models.items()):
                y_train_real, y_train_pred, y_test_real, y_test_pred = (
                    torch.tensor(v["y_train_real"]).float(),
                    torch.tensor(v["y_train_pred_average"]).float(),
                    torch.tensor(v["y_test_real"]).float(),
                    torch.tensor(v["y_test_pred_average"]).float(),
                )
                val_dict[k] = {
                    "Train": {
                        "MAE": (y_train_real - y_train_pred).abs().mean().item(),
                        "RMSE": (
                            (((y_train_real - y_train_pred) ** 2).mean()) ** (1 / 2)
                        ).item(),
                        "r2": r2_score(y_train_real, y_train_pred)
                        if y_train_pred.shape[0] > 1
                        else "Too few data points to calculate r2.",
                    },
                    "Test": {
                        "MAE": (y_test_real - y_test_pred).abs().mean().item(),
                        "RMSE": (
                            (((y_test_real - y_test_pred) ** 2).mean()) ** (1 / 2)
                        ).item(),
                        "r2": r2_score(y_test_real, y_test_pred)
                        if y_test_pred.shape[0] > 1
                        else "Too few data points to calculate r2.",
                    },
                }
                if parity_plots:
                    parity_plot = self.create_parity_plot(
                        datasets_pred=[y_train_pred, y_test_pred],
                        datasets_real=[y_train_real, y_test_real],
                        kwargs=kwargs,
                    )
                    lst_parity_plots.append(parity_plot)
        if get_pred:
            return predictions
        return val_dict, lst_parity_plots

    # =======================================================================

    def infer_model(self, dataset):

        self.output_models = self._load_model(self.model_name)

        self._model.eval()  # set to evaluation mode (may be redundant)
        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means

        infer_dict = {}
        for i, (k, v) in enumerate(self.output_models.items()):
            model_load_dirs = v["model_save_dirs"]
            self.data_transformation_dict = v["data_transformation_dict"]
            out_transform = self.data_transformation_dict[k]

            X_infer = self._data_preprocess(inference=True, infer_dataset=dataset)
            X_infer = torch.tensor(X_infer).float()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            prediction_l = []
            for m in model_load_dirs:
                model_load_dir = osp.join(self.save_path, m + "_BNN_model.pt")
                self._model.load_state_dict(
                    torch.load(model_load_dir, map_location=torch.device(device))
                )
                data = X_infer.to(device)
                predictions = self._model(data).item()
                predictions = self._untransform_data(
                    data=predictions, reduce=out_transform[0], divide=out_transform[1]
                )
                prediction_l.append(predictions)
            prediction_l = torch.tensor(prediction_l)
            predictions = prediction_l.mean(axis=0).item()
            infer_dict[k] = predictions

        return infer_dict
