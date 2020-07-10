from summit.utils.models import GPyModel, ModelGroup
from summit.utils.dataset import DataSet

import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore', category=RuntimeWarning)

# arameters
n_training_matlab = 30
num_restarts=100
n_spectral_points=1500
use_spectral_sample = False

def fit_and_test(n_training_matlab, num_restarts=100, max_iters=2000, n_spectral_points=4000, 
                use_spectral_sample=True, plot=True):
    # Read in data from one Matlab experiment
    X = pd.read_csv('data/matlab/experiment_1/X.csv', names=[f"x_{i}" for i in range(6)])
    y = pd.read_csv('data/matlab/experiment_1/Y.csv', names=['y_0', 'y_1'])
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
    X_min = X_train.min()
    X_max = X_train.max()
    X_train_scaled = (X_train-X_min)/(X_max-X_min)
    X_test_scaled = (X_test-X_min)/(X_max-X_min)

    # Scale objectives to 0 mean and unit variance
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train-y_mean)/y_std

    # Train model
    print(f"Fitting models (number of optimization restarts={num_restarts})")
    kerns = [GPy.kern.Exponential(input_dim=6,ARD=True) for _ in range(2)]
    models = ModelGroup({'y_0': GPyModel(kernel=kerns[0]),
                        'y_1': GPyModel(kernel=kerns[1])})
    models.fit(X_train_scaled, y_train_scaled, 
            num_restarts=num_restarts,
            max_iters=max_iters,
            n_spectral_points=n_spectral_points, 
            spectral_sample=False)
    for name, model in models.models.items():
        hyp = model.hyperparameters
        print(f"Model {name} lengthscales: {hyp[0]}")
        print(f"Model {name} variance: {hyp[1]}")
        print(f"Model {name} noise: {hyp[2]}")
    if use_spectral_sample:
        print(f"Spectral sampling with {n_spectral_points} spectral points.")
        for model in models.models.values():
            model.spectral_sample(X_train_scaled, y_train_scaled, 
                                n_spectral_points=n_spectral_points)

    # Model validation
    rmse = lambda pred, actual: np.sqrt(np.mean((pred-actual)**2, axis=0))

    y_pred_train_scaled = models.predict(X_train_scaled, 
                            use_spectral_sample=use_spectral_sample)
    y_pred_train_scaled = DataSet(y_pred_train_scaled, columns=['y_0', 'y_1'])
    y_pred_train = y_pred_train_scaled*y_std+y_mean
    rmse_train = rmse(y_pred_train.to_numpy(), y_train.to_numpy())

    y_pred_test_scaled = models.predict(X_test_scaled, 
                                        use_spectral_sample=use_spectral_sample)
    y_pred_test_scaled = DataSet(y_pred_test_scaled, columns=['y_0', 'y_1'])
    y_pred_test = y_pred_test_scaled*y_std+y_mean
    rmse_test = rmse(y_pred_test.to_numpy(), y_test.to_numpy())

    # Plots
    if plot:
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
        plt.savefig('20200710_train_gp_matlab_data.png',dpi=300)
        plt.show()
    
    objectives = [m._model.objective_function() for m in models.models.values()]
    return dict(rmse_train_y0=rmse_train[0],
                rmse_train_y1=rmse_train[1],
                rmse_test_y0=rmse_test[0],
                rmse_test_y1=rmse_test[1],
                objective_y0=objectives[0],
                objective_y1=objectives[1])

if __name__=='__main__':
    results = []
    for i in range(20):
        res = fit_and_test(
                    n_training_matlab=30,
                    max_iters=int(1e4),
                    num_restarts=100,
                    use_spectral_sample=False,
                    plot=False)
        results.append(res) 
    df = pd.DataFrame(results)
    df.to_csv('20200710_train_gp_matlab_data_comparison_data_100_restarts.csv')

    # Make plot
    fig,ax = plt.subplots()
    columns =['rmse_train_y0', 'rmse_train_y0', 'rmse_test_y0', 'rmse_test_y1']
    df.boxplot(ax=ax, column=columns)
    plt.show()
    fig.savefig('20200710_train_gp_matlab_data_boxplot_100_restarts.png', dpi=300)