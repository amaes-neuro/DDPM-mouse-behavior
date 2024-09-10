# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:46:25 2024

Plot some prob curves.

@author: amaes
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset_path='data/'
# read from pickle files
file = open(dataset_path+'actions_M_balanced10.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'states_M_balanced10.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends_M_balanced10.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

nb_A = 7
concentrations = np.array([8,10,9,9,7])
#phase B
sides_B = []
for i in range(5):
    side_left = []
    side_right = []
    side_synth_left = []
    side_synth_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
        end_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j]   
        locs = states[start_idx:end_idx,0]
        idx = np.where(locs[1:]!=locs[0:-1])[0]
        for k in range(len(idx)-1):
            if locs[idx[k]] == 1:
                side_left.append(idx[k+1]-idx[k])
            else:
                side_right.append(idx[k+1]-idx[k])
    sides_B.append(np.hstack(side_left))
    sides_B.append(np.hstack(side_right))

#example plot empirical prob curve
plt.figure()
for j in range(5):
    temp = sides_B[2*j]
    uniqs = np.unique(temp)
    probs = np.zeros((len(uniqs),))
    for i in range(len(uniqs)-1):
        idx = np.where(temp>=uniqs[i])[0]
        idx2 = np.where(temp > uniqs[i])[0]
        probs[i] = len(idx2)/len(idx)
    plt.plot(probs) #note that the x-axis is not cum time, but index for visualization
    



#example synthetic curves
model = 't_M_23' #six states model
plt.figure()
no_food = np.zeros((8,))
for i in range(8):
    file = open('data_model_curves_M/'+model+'/curve_M_'+str(i)+'.pickle', 'rb')
    temp = pickle.load(file) 
    file.close()
    plt.plot(np.mean(temp['actions'],axis=0))
    no_food[i] = np.mean(temp['actions'])

plt.figure()
food = np.zeros((8,))
for i in range(8,16):
    file = open('data_model_curves_M/'+model+'/curve_M_'+str(i)+'.pickle', 'rb')
    temp = pickle.load(file) 
    file.close()
    plt.plot(np.mean(temp['actions'],axis=0))
    food[i-8] = np.mean(temp['actions'])
    
plt.figure()
plt.plot(1-no_food[0:4],color='b')
plt.plot(1-food[0:4],color='b' , linestyle='dotted')
plt.plot(np.arange(4)+5,no_food[4:],color='r' )
plt.plot(np.arange(4)+5,food[4:],color='r' , linestyle='dotted')
plt.xticks(np.arange(9), ['0', '0.15', '0.30', '0.45', '', '0', '0.15', '0.30', '0.45'])
plt.xlabel('TMT concentration')
plt.ylabel('Probability of staying right (blue) or left (red)')