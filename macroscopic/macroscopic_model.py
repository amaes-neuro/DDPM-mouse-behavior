# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:35:38 2024

This code estimates parameters to model the cage dwell times using maximum likelihood.
We start with estimating parameters for phase A. After this we will consider phases B and C and potential changes to the model.
Work in progress.

Code from Ryomi Hattori was helpful as a starting point.

@author: ahm8208
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

#load data
dataset_path='data/'
# read from pickle files
file = open(dataset_path+'actions.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'states.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()


#organize data
concentrations = np.array([8,10,9,9,7])
phase_A_data = []
for i in range(5):
    for j in range(concentrations[i]):
        if i==0 and j==0:
            start_idx = 0
        if i!=0 or j>0:
            start_idx = episode_ends[np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations[0:i])+j] 
        phase_A_data.append(states[start_idx:end_idx,0] < 40)
n_trials_A = len(phase_A_data)


#compute negative log-likelihood and return also Q-values and probabilities for plotting
def RW_baseline(params, n_trials, data):
    betaq = params[0]
    beta0 = params[1]
    nloglik = 0
    QQ_ = []
    PP_ = []
    for i in range(n_trials):
        a = data[i]
        q0 = np.array([0.5, 0.5])
        QQ = np.zeros((len(a), 2))
        PP = np.zeros((len(a), 2))
        q = q0
        switch_time = 0
        prev_side = a[0]
        for t in range(len(a)):
            if prev_side != a[t]:
                switch_time = t
                prev_side = a[t]
            R = np.exp(-(t-switch_time)/60)
            ev = np.exp(betaq * (q + [0, beta0]))
            sev = sum(ev)
            p = ev / sev
            QQ[t, :] = q
            PP[t, :] = p
            if a[t] == 1:
                q[0] = q[0] + 0.1 * (R - q[0])
                q[1] = q[1] #+ alpha_R * (1-R - q[1])
            elif a[t] == 0:
                q[0] = q[0] #+ alpha_L * (1-R - q[0])
                q[1] = q[1] + 0.1 * (R - q[1])
        nloglik += -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 0, 1])))
        QQ_.append(QQ)
        PP_.append(PP)
    return nloglik, QQ_, PP_


#compute negative log-likelihood 
def RW_baseline_nll(params, n_trials, data):
    betaq = params[0]
    beta0 = params[1]
    nloglik = 0
    for i in range(n_trials):
        a = data[i]
        q0 = np.array([0.5, 0.5])
        QQ = np.zeros((len(a), 2))
        PP = np.zeros((len(a), 2))
        q = q0
        switch_time = 0
        prev_side = a[0]
        for t in range(len(a)):
            if prev_side != a[t]:
                switch_time = t
                prev_side = a[t]
            R = np.exp(-(t-switch_time)/60)
            ev = np.exp(betaq * (q + [0, beta0]))
            sev = sum(ev)
            p = ev / sev
            QQ[t, :] = q
            PP[t, :] = p
            if a[t] == 1:
                q[0] = q[0] + 0.1 * (R - q[0])
                q[1] = q[1] #+ alpha_R * (1-R - q[1])
            elif a[t] == 0:
                q[0] = q[0] #+ alpha_L * (1-R - q[0])
                q[1] = q[1] + 0.1 * (R - q[1])
        nloglik += -(sum(np.log(PP[a == 1, 0])) + sum(np.log(PP[a == 0, 1])))
    print(nloglik , params)
    return nloglik


# Fit 5 parameter model 
initial_params = np.array([ 3, 0])
bounds = [(0, 10), (None, None)]
RW_model = minimize(RW_baseline_nll, initial_params, args=(n_trials_A, phase_A_data), method='L-BFGS-B', bounds=bounds)
RW_model_nloglik, RW_model_QQ, RW_model_PP = RW_baseline(RW_model.x, n_trials_A, phase_A_data)  












