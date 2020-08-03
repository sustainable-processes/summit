# import pytest

# from summit.utils.models import GPyModel

# import numpy as np
# import matplotlib.pyplot as plt
# import warnings

# @pytest.mark.parametrize('n_dim', [1,6])
# def test_gpy_model(n_dim, n_points=100, n_repeats=5, plot=False):
#     noisy_fun = lambda x: np.mean(np.sin(x), axis=1) + np.random.randn(x.shape[0])*0.05

#     for i in range(n_repeats):
#         X_train = np.random.uniform(-3.,3.,(n_points,n_dim))
#         Y_train = noisy_fun(X_train)

#         #Scaling
#         X_min = np.min(X_train, axis=0)
#         X_max = np.max(X_train, axis=0)
#         X_train_scaled = (X_train-X_min)/(X_max-X_min)

#         Y_mean = np.mean(Y_train)
#         Y_std = np.std(Y_train)
#         Y_train_scaled = (Y_train-Y_mean)/Y_std
#         Y_train_scaled = np.atleast_2d(Y_train_scaled).T

#         # Fit model
#         warnings.filterwarnings('ignore', category=DeprecationWarning)
#         m = GPyModel(input_dim=n_dim)
#         m.fit(X_train_scaled, Y_train_scaled, spectral_sample=True)
#         Y_train_pred_scaled = m.predict(X_train_scaled, use_spectral_sample=True)
#         Y_train_pred = Y_train_pred_scaled[:,0]*Y_std + Y_mean
#         square_error = (Y_train_pred-Y_train)**2
#         train_rmse = np.sqrt(np.mean(square_error))
#         print("Training root mean squared error:", train_rmse)
#         assert train_rmse < 0.1

#         # Model validation
#         X_valid = np.random.uniform(-3, 3, (n_points,n_dim))
#         Y_valid = noisy_fun(X_valid)
#         X_valid_scaled = (X_valid-X_min)/(X_max-X_min)

#         Y_valid_pred_scaled = m.predict(X_valid_scaled, use_spectral_sample=True)
#         Y_valid_pred = Y_valid_pred_scaled[:,0]*Y_std+Y_mean

#         square_error = (Y_valid-Y_valid_pred)**2
#         valid_rmse = np.sqrt(np.mean(square_error))
#         print("Validation root mean squared error:",valid_rmse)

#         assert valid_rmse < 0.3

#         if plot and n_dim == 1:
#             fig, ax, = plt.subplots(1)
#             ax.scatter(X_valid[:,0], Y_valid_pred, label="Prediction")
#             ax.scatter(X_valid[:,0], Y_valid, label="True")
#             ax.legend()
#             plt.show()
