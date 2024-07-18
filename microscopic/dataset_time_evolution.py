# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:18:50 2024

The time evolution of the probability of being in the left side of the cage.
This is estimated from the data using a sliding window.

@author: ahm8208
"""


import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

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


#set variable
T = 2500 #this is almost 14 minutes at 3Hz, each trial has a variable length so we have to cut it off to the minimum
#there is actually only one trial below 2600 steps so it is a shame I have to cut to this extent
time_evol_A = np.zeros((5,T))
time_evol_B = np.zeros((5,T))
time_evol_C = np.zeros((5,T))
concentrations = np.array([8,10,9,9,7])

#get location data phase A
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        if i==0 and j==0:
            start_idx = 0
        if i!=0 or j>0:
            start_idx = episode_ends[np.sum(concentrations[0:i])+j-1] 
        end_idx = start_idx+T  
        print(start_idx)
        ev_[j,:] = states[start_idx:end_idx,0] < 40
    time_evol_A[i,:] = np.mean(ev_,axis=0)
    
#get location data phase B
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = start_idx+T  
        print(start_idx)

        ev_[j,:] = states[start_idx:end_idx,0] < 40
    time_evol_B[i,:] = np.mean(ev_,axis=0)
    
#get location data phase C
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = start_idx+T  
        print(start_idx)

        ev_[j,:] = states[start_idx:end_idx,0] < 40
    time_evol_C[i,:] = np.mean(ev_,axis=0)
    
   
#set params
window_length = 180 #in nb of time steps (3Hz)
window_step = 180 #less overlap in windows than if you would take convolution

#compute mean with running window
phase_A = np.zeros((5,(T-window_length)//window_step))
phase_B = np.zeros((5,(T-window_length)//window_step))
phase_C = np.zeros((5,(T-window_length)//window_step))
for i in range(5):
    for j in range((T-window_length)//window_step):
        phase_A[i,j] = np.mean(time_evol_A[i,j*window_step:j*window_step+window_length])
        phase_B[i,j] = np.mean(time_evol_B[i,j*window_step:j*window_step+window_length])
        phase_C[i,j] = np.mean(time_evol_C[i,j*window_step:j*window_step+window_length])

colors = ['rosybrown','lightcoral','indianred','brown','darkred']
plt.figure()
for i in range(5):
    plt.plot(window_step*np.arange(((T-window_length)//window_step))/180,phase_A[i,:],color=colors[i])
    plt.plot(window_step*np.arange(((T-window_length)//window_step),2*((T-window_length)//window_step))/180,phase_B[i,:],color=colors[i])
    plt.plot(window_step*np.arange(2*((T-window_length)//window_step),3*((T-window_length)//window_step))/180,phase_C[i,:],color=colors[i])
plt.vlines(window_step*((T-window_length)//window_step)/180,0,1,linestyles='--')
plt.vlines(window_step*(2*((T-window_length)//window_step))/180,0,1,linestyles='--')
plt.ylabel('fraction in left side of cage')    
plt.xlabel('Time (mins)')








