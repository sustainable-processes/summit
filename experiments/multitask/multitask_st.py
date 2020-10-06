import streamlit as st
import botorch
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import torch
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("Multitask GP Model")

# Sliders
n_t1 = st.slider("Task 1 Observations", min_value=0, max_value=50, value=20)

n_t2 = st.slider("Task 2 Observations", min_value=0, max_value=50, value=20)
noise = st.slider(
    "Observation Noise", min_value=0.0, max_value=1.0, value=0.0, step=0.1
)
use_fixed_noise = st.checkbox("Use Fixed Noise Model", False)

f1 = lambda X: torch.cos(5 * X[:, 0]) ** 2
f2 = lambda X: 1.5 * torch.cos(5 * X[:, 0]) ** 2
gen_inputs = lambda n: torch.rand(n, 1)
gen_obs = lambda X, f, noise: f(X) + noise * torch.rand(X.shape[0])

### Data Generation
X1, X2 = gen_inputs(n_t1), gen_inputs(n_t2)
i1, i2 = torch.zeros(n_t1, 1), torch.ones(n_t2, 1)

train_X = torch.cat([torch.cat([X1, i1], -1), torch.cat([X2, i2], -1)])

train_Y_f1 = gen_obs(X1, f1, noise)
train_Y_f2 = gen_obs(X2, f2, noise)
train_Y = torch.cat([train_Y_f1, train_Y_f2]).unsqueeze(-1)
train_Y_mean = train_Y.mean()
train_Y_std = train_Y.std()
train_Y_norm = (train_Y - train_Y_mean) / train_Y_std
train_Yvar = noise * torch.rand(train_X.shape[0], 1)

### Model Training
if not use_fixed_noise:
    model = botorch.models.MultiTaskGP(train_X, train_Y_norm, task_feature=-1)
else:
    model = botorch.models.FixedNoiseMultiTaskGP(
        train_X, train_Y_norm, train_Yvar, task_feature=-1
    )
mll = ExactMarginalLogLikelihood(model.likelihood, model)
botorch.fit.fit_gpytorch_model(mll)

### Plotting
st.markdown(
    "Red is for Task 1, and blue is for task 2. Dotted lines are the ground truth. Dots are training observations, and translucent lines are posterior samples of the trained model for each task."
)

fig, ax = plt.subplots(1)

# Ground truth
X_plot = np.atleast_2d(np.linspace(0, 1, 100)).T
X_plot = torch.tensor(X_plot).float()
ax.plot(X_plot, f1(X_plot), "--", label="Task 1", alpha=1, c="r")
ax.plot(X_plot, f2(X_plot), "--", label="Task 2", alpha=1, c="b")

# Observations
ax.scatter(X1, train_Y_f1, c="r")
ax.scatter(X2, train_Y_f2, c="b")

# Posterior of model
with torch.no_grad():
    posterior = model.posterior(X_plot)
    for i in range(100):
        y = posterior.sample()[0, :, 0] * train_Y_std + train_Y_mean
        ax.plot(X_plot, y, alpha=0.01, color="r")
    for i in range(100):
        y = posterior.sample()[0, :, 1] * train_Y_std + train_Y_mean
        ax.plot(X_plot, y, alpha=0.01, color="b")
    # ax.plot(X_plot, posterior.mean.detach())
ax.legend(loc="best")

st.write(fig)
