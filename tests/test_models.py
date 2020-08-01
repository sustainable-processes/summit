# import pytest
# from summit.utils.models import GPyModel
# import matplotlib.pyplot as plt
# import numpy as np

# Add back in once fixed GPyModel

# def test_gpy_model():
#     X = np.random.uniform(-3.,3.,(20,1))
#     Y = np.sin(X) + np.random.randn(20,1)*0.05
#     m = GPyModel(input_dim=1)
#     m.fit(X, Y)
#     sampled_f = m.spectral_sample(X, Y)
#     predict_Y = m.predict(X)
#     sample_Y = sampled_f(X)
#     mae_sample = np.mean(np.abs(sample_Y[:,0]-Y[:,0]))
#     mae_pred = np.mean(np.abs(predict_Y[:,0]-Y[:,0]))
#     assert mae_sample < 0.1
#     assert mae_pred < 0.1

