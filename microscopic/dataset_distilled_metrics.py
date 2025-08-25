# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:25:55 2024

Plots of time spent on left side of cage.

@author: ahm8208
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression



dataset_path='data/'
dataset_name = 'balanced7_x1'
# read from pickle files
file = open(dataset_path+'actions_'+dataset_name+'.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()


concentrations = np.array([8,10,9,9,7])

model = 't_701_2'
#phase A
nb_A = 7
locs_A = []
synth_locs_A = []
for i in range(5):
    locs = []
    for j in range(concentrations[i]):
        if i==4:
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            synth_locs_A.append(np.mean(states_synth[:,0]<40))
    
        start_idx = 0
        if i>0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        locs.append(np.mean(states[start_idx:end_idx,0]<40))

    locs_A.append(np.vstack(locs))
synth_locs_A = np.vstack(synth_locs_A)

#phase B
locs_B = []
synth_locs_B = []
for i in range(5):
    locs = []
    synth_locs = []
    for j in range(concentrations[i]):
        file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        loc = states_synth[:,0]
        synth_locs.append(np.mean(loc<40))
        
        start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
        end_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j]   
        loc = states[start_idx:end_idx,0]
        locs.append(np.mean(loc<40))

    locs_B.append(np.vstack(locs))
    synth_locs_B.append(np.vstack(synth_locs))


#phase C
locs_C = []
synth_locs_C = []
for i in range(5):
    locs = []
    synth_locs = []
    for j in range(concentrations[i]):
        file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        loc = states_synth[:,0]
        synth_locs.append(np.mean(loc<40))
        
        start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
        end_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j]   
        loc = states[start_idx:end_idx,0]
        locs.append(np.mean(loc<40))

    locs_C.append(np.vstack(locs))
    synth_locs_C.append(np.vstack(synth_locs))


colors = ['rosybrown','lightcoral','indianred','brown','darkred']
fig, ax = plt.subplots()
ax.scatter(locs_A[-1],synth_locs_A,color='r')
for i in range(5):
    ax.scatter(locs_B[i],synth_locs_B[i],color=colors[i],marker='*')
    ax.scatter(locs_C[i],synth_locs_C[i],color=colors[i],marker='X')
ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),linestyle='--',color='black')

ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Mice time spent on left side')
plt.ylabel('Model time spent on left side')
plt.show()

