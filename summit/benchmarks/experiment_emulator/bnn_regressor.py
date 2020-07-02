import os
import os.path as osp

from summit.benchmarks.experiment_emulator.emulator import Emulator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import json

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split as sklearn_train_test_split
import sklearn.preprocessing

from experimental_datasets import load_reizman_suzuki


# =======================================================================

class BNNEmulator(Emulator):
    """ BNN Trainer

    A Bayesian Neural Network (BNN) emulator trainer.

    Examples
    --------
    >>> emulator = BNNEmulator(domain, dataset)

    Notes
    -----
    This benchmark is based on Hone et al. Reac Engr. & Chem. 2016.

    """

    def __init__(self, domain, dataset, model_name, kwargs={}):
        self._domain = domain
        self._domain_preprocess()

        if dataset is not None:
            self._dataset = dataset
            # Preprocess dataset

        regression_model = self._setup_model()
        super().__init__(regression_model)

        # Set model name for saving
        self.save_path = kwargs.get("save_path", osp.join(osp.dirname(osp.realpath(__file__)), "trained_models/BNN"))
        self.model_name = str(model_name)

        # Set up training hyperparameters
        self.set_training_hyperparameters()

    def _setup_model(self, **kwargs):
        """ Setup the BNN model """

        @variational_estimator
        class BayesianRegressor(nn.Module):
            def __init__(self, input_dim):
                super().__init__()

                self.blinear1 = BayesianLinear(input_dim, 24)
                self.blinear2 = BayesianLinear(24, 24)
                self.blinear3 = BayesianLinear(24, 1)

            def forward(self, x):
                x = F.leaky_relu(self.blinear1(x))
                x = F.leaky_relu(self.blinear2(x))
                x = F.dropout(x, p=0.1, training=self.training)
                x = F.relu(self.blinear3(x))
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
            def _evaluate_regression(self, regressor, device, loader, fun_untransform_data, out_transform,):
                regressor.eval()

                mae = 0
                for i, (datapoints, labels) in enumerate(loader):
                    data = datapoints.to(device)
                    tmp_pred_data = fun_untransform_data(data=regressor(data), reduce=out_transform[0], divide=out_transform[1])
                    tmp_real_data = fun_untransform_data(data=labels, reduce=out_transform[0], divide=out_transform[1])
                    mae += (tmp_pred_data - tmp_real_data).abs().mean()

                return mae

        regression_model = BayesianRegressor(self.input_dim)
        return regression_model

    def set_training_hyperparameters(self, kwargs={}):
        # Setter method for hyperparameters of training
        self.epochs = kwargs.get("epochs", 300) # number of max epochs the model is trained
        self.initial_lr = kwargs.get("initial_lr", 0.001)  # initial learning rate
        self.min_lr = kwargs.get("min_lr", 0.00001)
        self.lr_decay = kwargs.get("lr_decay", 0.7)  # learning rate decay
        self.lr_decay_patience = kwargs.get("lr_decay_patience", 3) # number of epochs before learning rate is reduced by lr_decay
        self.early_stopping_epochs = kwargs.get("early_stopping_epochs", 20)  # number of epochs before early stopping
        self.batch_size_train = kwargs.get("batch_size_train", 4)
        self.transform_input = kwargs.get("transform_input", "standardize")
        self.transform_output = kwargs.get("transform_output", "standardize")


    def train_model(
            self, dataset=None, verbose=True, kwargs={}
    ):

        # Manual call of training -> overwrite dataset with new dataset for training
        if dataset is not None:
            self._dataset = dataset

        # Data preprocess
        self._train_dataset, self._test_dataset = \
            self._data_preprocess(transform_input=self.transform_input, transform_output=self.transform_output)

        X_train, y_train = torch.tensor(self._train_dataset[0]).float(), torch.tensor(self._train_dataset[1]).float()
        X_test, y_test = torch.tensor(self._test_dataset[0]).float(), torch.tensor(self._test_dataset[1]).float()

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
            model_save_dir = osp.join(self.save_path, self.model_name + "_" + str(k) + "_BNN_model.pt")
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

                # if prediction accuracy was improved in current epoch, reset <tmp_iter_stop> and save model
                if best_train_mae > train_mae:
                    best_train_mae = train_mae
                    tmp_iter_stop = 0
                    torch.save(regressor.state_dict(), model_save_dir)
                # if prediction accuracy was not imporved in current epoch, increase <tmp_iter_stop> and stop training if <max_iter_stop> is reached
                else:
                    tmp_iter_stop += 1
                    if tmp_iter_stop >= max_iter_stop:
                        break

                # print mean absolute error (MAE) on training set for current epoch (same for test set every 100th epoch)
                if verbose:
                    print("   -- Epoch: {:03d}, LR: {:7f}, Train MAE: {:4f}".format(epoch, lr, train_mae))
                    if epoch % 100 == 0:
                        test_mae = self._model._evaluate_regression(regressor, device, dataloader_test, self._untransform_data, out_transform)
                        print("   -> Epoch: {:03d}, Test MAE: {:4f}".format(epoch, test_mae))

            self.output_models[k] = {"model_save_dir": model_save_dir, "data_transformation_dict": self.data_transformation_dict}
            if verbose:
                print("\n  <-- Finished training of BNN model on objective: {} -->\n"
                      "   -- Model saved at: {} --\n".format(k, model_save_dir))

        self._save_model()
        if verbose:
            print("<---- End training of BNN regressor ---->\n")

    def validate_model(self, dataset, kwargs={}):
        self.output_models = self._load_model(self.model_name)

        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means
        self._model.eval()  # set to evaluation mode (may be redundant)

        val_dict = {}
        for i, (k, v) in enumerate(self.output_models.items()):
            model_load_dir = v["model_save_dir"]
            self.data_transformation_dict = v["data_transformation_dict"]
            out_transform = self.data_transformation_dict[k]

            X_val = self._data_preprocess(inference=True, infer_dataset=dataset, validate=True)
            X_val = torch.tensor(X_val).float()
            y_val = dataset[(k, "DATA")]
            y_val = torch.tensor(dataset[(k, "DATA")].to_numpy()).float()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.load_state_dict(torch.load(model_load_dir, map_location=torch.device(device)))
            data = X_val.to(device)
            predictions = self._model(data).detach()
            predictions = self._untransform_data(data=predictions, reduce=out_transform[0], divide=out_transform[1])
            val_dict[k] = {"MAE": (predictions - y_val).abs().mean().item(),
                           "RMSE": ((((predictions - y_val)**2).mean())**(1/2)).item()}

        return val_dict

    def infer_model(self, dataset):

        self.output_models = self._load_model(self.model_name)

        self._model.eval()  # set to evaluation mode (may be redundant)
        self._model.freeze_()  # freeze the model, in order to predict using only their weight distribution means

        infer_dict = {}
        for i, (k, v) in enumerate(self.output_models.items()):
            model_load_dir = v["model_save_dir"]
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



'''


# set dataset
dataset_name = "reizman_suzuki"
case = 2
target = "TON"
X, y = load_reizman_suzuki(return_X_y=True, case=case)

# set hyperparameters
epochs = 300
initial_lr = 0.001
early_stopping_epochs = 20

# adapt target (only if multiple targets), comment out for single-objective dataset
target_dim = 0
if target == "TON":
    target_dim = 0
elif target == "yield":
    target_dim = 1

y = y[:,target_dim]

# adapt model name
model_name = str(dataset_name) + "_case" + str(case) + "_" + str(target)

# adapt save directory
save_path = osp.join(osp.dirname(osp.realpath(__file__)), "trained_models/BNN")

print("<---- Dataset: {} case {}, Target property: {} ---->".format(dataset_name, case, target))

#=======================================================================

# convert categorical variables to one-hot tensors
tmp_ligand_type = torch.tensor(X[:,0]).int()
tmp_ligand_type = torch.unique(tmp_ligand_type, True, True)[1]
num_types = int(tmp_ligand_type.max().item() + 1)
tmp_ligand_type = F.one_hot(tmp_ligand_type, num_classes=num_types).to(torch.float)

# standardize continuous input variables
tmp_inp_cont = torch.tensor(X[:,1:]).float()
inp_mean = tmp_inp_cont.mean(axis=0)
inp_std = tmp_inp_cont.std(axis=0)
tmp_inp_cont = (tmp_inp_cont - inp_mean) / inp_std 

# X - input: concatenate one-hot caterogical variables and continuous variables
X = torch.cat((tmp_ligand_type, tmp_inp_cont), axis=1)
inp_dim = X.shape[1]

# divide target variable by average
y = torch.tensor(y).float()
out_mean = y.mean(axis=0)
y = y / out_mean 

# split data into training and test set
## random split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,shuffle=True)
## predefined split (takes the #<test_size> last points of the dataset csv file)
test_size = 8
X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True)

ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=16, shuffle=True)

print("<---- Length of train dataset: {} ---->".format(X_train.shape[0]))
print("<---- Length of test dataset: {} ---->".format(X_test.shape[0]))

#=======================================================================

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.blinear1 = BayesianLinear(input_dim, 24)
        self.blinear2 = BayesianLinear(24, 24)
        self.blinear3 = BayesianLinear(24, output_dim)
        #self.linear = nn.Linear(24, output_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.blinear1(x))
        #x = F.dropout(x, p=0.1, training=self.training)
        x = F.leaky_relu(self.blinear2(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.blinear3(x))
        #x = self.linear(x)
        y = x
        return y.view(-1)

#=======================================================================

# Training of model on given dataloader
def train(loader):
    regressor.train()

    for i, (datapoints, labels) in enumerate(loader):
        data = datapoints.to(device)
        optimizer.zero_grad()
        loss = regressor.sample_elbo(inputs=datapoints.to(device),
                           labels=labels.to(device),
                           criterion=criterion,
                           sample_nbr=3,
                           complexity_cost_weight=1/X_train.shape[0])
        loss.backward()
        optimizer.step()


# Evaluate model for given dataloader
def evaluate_regression(loader):
    regressor.eval()

    mae = 0
    for i, (datapoints, labels) in enumerate(loader):
        data = datapoints.to(device)
        tmp_pred_data = regressor(data) * out_mean
        tmp_real_data = labels * out_mean
        mae += (tmp_pred_data - tmp_real_data).abs().mean()

    return mae

#=======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regressor = BayesianRegressor(inp_dim, 1).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.7, patience=3, min_lr=0.00001)
criterion = torch.nn.MSELoss()

#=======================================================================

print("\n<---- Start training of BNN model ---->\n")

max_iter_stop = early_stopping_epochs   # maximum number of consecutive iteration w/o improvement after which training is stopped
tmp_iter_stop = 0
best_train_mae = float("inf")
for epoch in range(300):

    lr = scheduler.optimizer.param_groups[0]["lr"]

    # train model
    train(dataloader_train)

    # TODO: define stopping criterion! To use training mae is not the best way to do this (-> overfitting, usually we have an extra validation set, problem: small dataset size, cross-validation is exhaustive at this point)
    train_mae = evaluate_regression(dataloader_train)
    scheduler.step(train_mae)

    # if prediction accuracy was improved in current epoch, reset <tmp_iter_stop> and save model
    if best_train_mae > train_mae:
        best_train_mae = train_mae
        tmp_iter_stop = 0
        save_model_weights = osp.join(save_path, model_name + "_BNN_model.pt")
        torch.save(regressor.state_dict(), save_model_weights)
    # if prediction accuracy was not imporved in current epoch, increase <tmp_iter_stop> and stop training if <max_iter_stop> is reached
    else:
        tmp_iter_stop += 1
        if tmp_iter_stop >= max_iter_stop:
            break

    # print mean absolute error (MAE) on training set for current epoch (same for test set every 100th epoch)
    print("   -- Epoch: {:03d}, LR: {:7f}, Train MAE: {:4f}".format(epoch, lr, train_mae))
    if epoch%100==0:
        test_mae = evaluate_regression(dataloader_test)
        print("   -> Epoch: {:03d}, Test MAE: {:4f}".format(epoch, test_mae))

print("\n<---- End training of BNN model ---->\n")

#=======================================================================

print("<---- Postprocessing ---->\n")

# load final model from epoch with lowest prediction accuracy 
regressor.load_state_dict(torch.load(osp.join(save_path, model_name + "_BNN_model.pt")))
# freeze the model, in order to predict using only their weight distribution means
regressor.freeze_()

# get final model predictions for training and test data
y_train_pred = regressor(X_train) * out_mean
y_train = y_train * out_mean
y_test_pred = regressor(X_test) * out_mean
y_test = y_test * out_mean

# Write model performance to general csv file
path_csv = (osp.join(save_path, model_name + "_exp_pred.csv"))
with open(path_csv,"w+", newline="") as result_file:
    wr = csv.writer(result_file, quoting=csv.QUOTE_ALL)
    wr.writerow(["Exp", "Pred"])
    wr.writerow(["Training set"])
    for i in range(y_train.shape[0]):
        wr.writerow([y_train[i].item(), y_train_pred[i].item()]) 
    wr.writerow(["Test set"])
    for i in range(y_test.shape[0]):
        wr.writerow([y_test[i].item(), y_test_pred[i].item()]) 

# Create parity plots
plt.figure(figsize=(5,5))
plt.plot(y_train.detach().numpy(), y_train_pred.detach().numpy(), "o")
plt.plot(y_test.detach().numpy(), y_test_pred.detach().numpy(), "x")
plt.xlabel("Experimental y", fontsize=16)
plt.ylabel("Predicted y", fontsize=16)
plt.savefig(osp.join(save_path, model_name + "_ParityPlot"))

print("<---- Finished! ---->\n")

'''
