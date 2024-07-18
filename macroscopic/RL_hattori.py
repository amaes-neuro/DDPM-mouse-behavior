### Example python code for fitting Rescorla-Wagner reinforcement learning models to behavioral data. 
### It fits data with the simplest model and the version used in Hattori et al., Cell, 2019
### Written by Ryoma Hattori

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

file_path = r"...\Codes for Amadeus\ReinforcementLearning.mat"
SessionData = loadmat(file_path)
R = SessionData['R'].squeeze()
a = SessionData['a'].squeeze()
n_trials = len(a)

# %% Define models
def RW_decays_bias(params, n_trials, a, R):
    alpha_rew = params[0]
    alpha_unr = params[1]
    decayun = params[2]
    betaq = params[3]
    beta0 = params[4]
    q0 = np.array([0.5, 0.5])
    QQ = np.zeros((n_trials, 2))
    PP = np.zeros((n_trials, 2))
    q = q0
    for t in range(n_trials):
        ev = np.exp(betaq * (q + [0, beta0]))
        sev = sum(ev)
        p = ev / sev
        QQ[t, :] = q
        PP[t, :] = p
        if a[t] == 1:
            if R[t] == 1:
                q[0] = q[0] + alpha_rew * (R[t] - q[0])
            elif R[t] == 0:
                q[0] = q[0] + alpha_unr * (R[t] - q[0])
            q[1] = (1 - decayun) * q[1]
        elif a[t] == 2:
            q[0] = (1 - decayun) * q[0]
            if R[t] == 1:
                q[1] = q[1] + alpha_rew * (R[t] - q[1])
            elif R[t] == 0:
                q[1] = q[1] + alpha_unr * (R[t] - q[1])
        elif a[t] == 3 or a[t] == 4:
            q[0] = (1 - decayun) * q[0]
            q[1] = (1 - decayun) * q[1]
    nloglik = -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 2, 1])))
    return nloglik, QQ, PP


def RW_decays_bias_nloglik(params, n_trials, a, R):
    alpha_rew = params[0]
    alpha_unr = params[1]
    decayun = params[2]
    betaq = params[3]
    beta0 = params[4]
    q0 = np.array([0.5, 0.5])
    QQ = np.zeros((n_trials, 2))
    PP = np.zeros((n_trials, 2))
    q = q0
    for t in range(n_trials):
        ev = np.exp(betaq * (q + [0, beta0]))
        sev = sum(ev)
        p = ev / sev
        QQ[t, :] = q
        PP[t, :] = p
        if a[t] == 1:
            if R[t] == 1:
                q[0] = q[0] + alpha_rew * (R[t] - q[0])
            elif R[t] == 0:
                q[0] = q[0] + alpha_unr * (R[t] - q[0])
            q[1] = (1 - decayun) * q[1]
        elif a[t] == 2:
            q[0] = (1 - decayun) * q[0]
            if R[t] == 1:
                q[1] = q[1] + alpha_rew * (R[t] - q[1])
            elif R[t] == 0:
                q[1] = q[1] + alpha_unr * (R[t] - q[1])
        elif a[t] == 3 or a[t] == 4:
            q[0] = (1 - decayun) * q[0]
            q[1] = (1 - decayun) * q[1]
    nloglik = -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 2, 1])))
    return nloglik


def RW_simplest(params, n_trials, a, R):
    alpha = params[0]
    betaq = params[1]
    q0 = np.array([0.5, 0.5])
    QQ = np.zeros((n_trials, 2))
    PP = np.zeros((n_trials, 2))
    q = q0
    for t in range(n_trials):
        ev = np.exp(betaq * q)
        sev = sum(ev)
        p = ev / sev
        QQ[t, :] = q
        PP[t, :] = p
        if a[t] == 1:
            q[0] = q[0] + alpha * (R[t] - q[0])
        elif a[t] == 2:
            q[1] = q[1] + alpha * (R[t] - q[1])
    nloglik = -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 2, 1])))
    return nloglik, QQ, PP


def RW_simplest_nloglik(params, n_trials, a, R):
    alpha = params[0]
    betaq = params[1]
    q0 = np.array([0.5, 0.5])
    QQ = np.zeros((n_trials, 2))
    PP = np.zeros((n_trials, 2))
    q = q0
    for t in range(n_trials):
        ev = np.exp(betaq * q)
        sev = sum(ev)
        p = ev / sev
        QQ[t, :] = q
        PP[t, :] = p
        if a[t] == 1:
            q[0] = q[0] + alpha * (R[t] - q[0])
        elif a[t] == 2:
            q[1] = q[1] + alpha * (R[t] - q[1])
    nloglik = -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 2, 1])))
    return nloglik
#%% Fit models with maximum likelihood estimation (This is actually minimization of negative log likelihood)

# Fit simplest RW model with 2 parameters
initial_params = np.array([0.2, 3])
bounds = [(0, 1), (0, 10)]
RW_model = minimize(RW_simplest_nloglik, initial_params, args=(n_trials, a, R), method='L-BFGS-B', bounds=bounds)
RW_model_nloglik, RW_model_QQ, RW_model_PP = RW_simplest(RW_model.x, n_trials, a, R)  # RW_model_PP has [right choice probability, left choice probability]

# Fit 5 parameter model used in Hattori et al., 2019
initial_params = np.array([0.2, 0.2, 0.2, 3, 0])
bounds = [(0, 1), (0, 1), (0, 1), (0, 10), (None, None)]
RW_decays_bias_model = minimize(RW_decays_bias_nloglik, initial_params, args=(n_trials, a, R), method='L-BFGS-B', bounds=bounds)
RW_decays_bias_model_nloglik, RW_decays_bias_model_QQ, RW_decays_bias_model_PP = RW_decays_bias(RW_decays_bias_model.x, n_trials, a, R)  # RW_decays_bias_model_PP has [right choice probability, left choice probability]

# Plot the best fit parameters
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
ax = axes[0, 0]
labels = [r'$\alpha$']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
ax.bar(x, RW_model.x[0], width, color='k')
ax.set_ylabel('Weights')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.title.set_text("RW model (Simplest)")
ax = axes[0, 1]
labels = [r'$\beta_{\Delta_Q}$']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
ax.bar(x, RW_model.x[1], width, color='k')
ax.set_ylabel('Weights')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.title.set_text("RW model (Simplest)")
ax = axes[1, 0]
labels = [r'$\alpha_{rew}$', r'$\alpha_{unr}$', r'$\delta_{f}$', 'bias']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
ax.bar(x, RW_decays_bias_model.x[np.r_[0:3, -1]], width, color='k')
ax.set_ylabel('Weights')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.title.set_text("RW model (Hattori et al., 2019)")
ax = axes[1, 1]
labels = [r'$\beta_{\Delta_Q}$']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
ax.bar(x, RW_decays_bias_model.x[3], width, color='k')
ax.set_ylabel('Weights')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.title.set_text("RW model (Hattori et al., 2019)")
plt.show()
