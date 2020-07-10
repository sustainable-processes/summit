from summit.utils.models import GPyModel, ModelGroup
from summit.utils.dataset import DataSet

import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# arameters
n_training_matlab = 30
num_restarts=100
n_spectral_points=4000
use_spectral_sample = False

# Read in data from one Matlab experiment
X = pd.read_csv('data/matlab/experiment_20/X.csv', names=[f"x_{i}" for i in range(6)])
y = pd.read_csv('data/matlab/experiment_20/Y.csv', names=['y_0', 'y_1'])
X = DataSet.from_df(X)
y = DataSet.from_df(y)

# Train-test split
X_train = X.iloc[:n_training_matlab, :]
X_test =  X.iloc[n_training_matlab:, :]
y_train = y.iloc[:n_training_matlab, :]
y_test = y.iloc[n_training_matlab:, :]
print("Number of training data:", X_train.shape[0])
print("Number of test data:", X_test.shape[0])

# Scale decision variables between 0 and 1
# X_min = X_train.min()
# X_max = X_train.max()

# Scale objectives to 0 mean and unit variance
y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train-y_mean)/y_std

# Train model
print("Fitting models")
if use_spectral_sample:
    print("Number of spectral points:", n_spectral_points)
kerns = [GPy.kern.Exponential(input_dim=6,ARD=True) for _ in range(2)]
models = ModelGroup({'y_0': GPyModel(kernel=kerns[0]),
                     'y_1': GPyModel(kernel=kerns[1])})
models.fit(X_train, y_train_scaled, 
           num_restarts=num_restarts,
           n_spectral_points=n_spectral_points, 
           spectral_sample=use_spectral_sample)

# Model validation
rmse = lambda pred, actual: np.sqrt(np.mean((pred-actual)**2, axis=0))

y_pred_train_scaled = models.predict(X_train, 
                         use_spectral_sample=use_spectral_sample)
y_pred_train_scaled = DataSet(y_pred_train_scaled, columns=['y_0', 'y_1'])
y_pred_train = y_pred_train_scaled*y_std+y_mean
rmse_train = rmse(y_pred_train.to_numpy(), y_train.to_numpy())

y_pred_test_scaled = models.predict(X_test, 
                                    use_spectral_sample=use_spectral_sample)
y_pred_test_scaled = DataSet(y_pred_test_scaled, columns=['y_0', 'y_1'])
y_pred_test = y_pred_test_scaled*y_std+y_mean
rmse_test = rmse(y_pred_test.to_numpy(), y_test.to_numpy())

# Plots
fig, axes = plt.subplots(1,2)
fig.suptitle("With Spectral Sampling" if use_spectral_sample else "Without Spectral Sampling")
for i, name in enumerate(models.models.keys()):
    axes[i].scatter(y_train[name], y_pred_train[name], 
                    label=f"Training: RMSE = {rmse_train[i].round(2)}")
    axes[i].scatter(y_test[name], y_pred_test[name],
                    label=f"Test: RMSE = {rmse_test[i].round(2)}")
    axes[i].plot([0,2], [0,2])
    axes[i].legend()
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].set_title(name)
plt.savefig('20200709_train_gp_matlab_data.png',dpi=300)
plt.show()

