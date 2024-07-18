# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:36:40 2024

Wall distance as function of experimental phase and concentration.


@author: ahm8208
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


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


# phase A
distances_A = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    state_slice = states[start_idx:end_idx,2:20]
    dist_idx = np.argpartition(state_slice,6,axis=-1)
    distances_A.append(np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1))

plt.figure()
plt.hist(np.hstack(distances_A),50,(0,20),True)
plt.xlabel('Mean of smallest 6 distances to walls at 3Hz [cm]')
plt.ylabel('Density')
plt.title('Phase A')



# different concentrations, phase B
concentrations = np.array([8,10,9,9,7])
distances_B = []
for i in range(5):
    dist = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        state_slice = states[start_idx:end_idx,2:20]
        dist_idx = np.argpartition(state_slice,6,axis=-1)
        dist.append(np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1))
    distances_B.append(np.hstack(dist))
    
plt.figure()    
plt.violinplot([np.hstack(distances_A)],[1],showmeans=True)
plt.violinplot([distances_B[i] for i in range(5)],[2,3,4,5,6],showmeans=True)
plt.hlines(np.mean(np.hstack(distances_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Mean of smallest 6 distances to walls at 3Hz [cm]')
plt.ylim(0,20)
plt.title('Phase B')



# different concentrations, phase C
concentrations = np.array([8,10,9,9,7])
distances_C = []
for i in range(5):
    dist = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        state_slice = states[start_idx:end_idx,2:20]
        dist_idx = np.argpartition(state_slice,6,axis=-1)
        dist.append(np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1))
    distances_C.append(np.hstack(dist))
    
plt.figure()    
plt.violinplot([np.hstack(distances_A)],[1],showmeans=True)
plt.violinplot([distances_C[i] for i in range(5)],[2,3,4,5,6],showmeans=True)
plt.hlines(np.mean(np.hstack(distances_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Mean of smallest 6 distances to walls at 3Hz [cm]')
plt.ylim(0,20)
plt.title('Phase C')



"""
Break down by left/right side of cage
"""

# phase A
distances_A_left = []
distances_A_right = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    side = states[start_idx:end_idx,0]<40
    state_slice = states[start_idx:end_idx,2:20]
    dist_idx = np.argpartition(state_slice,6,axis=-1)
    mean_dists = np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1)
    distances_A_left.append(mean_dists[side])
    distances_A_right.append(mean_dists[~side])


# different concentrations, phase B
concentrations = np.array([8,10,9,9,7])
distances_B = []
for i in range(5):
    dist_left = []
    dist_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        side = states[start_idx:end_idx,0]<40
        state_slice = states[start_idx:end_idx,2:20]
        dist_idx = np.argpartition(state_slice,6,axis=-1)
        mean_dists = np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1)
        dist_left.append(mean_dists[side])
        dist_right.append(mean_dists[~side])
    distances_B.append(np.hstack(dist_left))
    distances_B.append(np.hstack(dist_right))
    
plt.figure()    
plt.violinplot([np.hstack(distances_A_left),np.hstack(distances_A_right)],[0.8,1.2],showmeans=True)
plt.violinplot([distances_B[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(distances_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Mean of smallest 6 distances to walls at 3Hz [cm]')
plt.ylim(0,20)
plt.title('Phase B')


# different concentrations, phase C
concentrations = np.array([8,10,9,9,7])
distances_C = []
for i in range(5):
    dist_left = []
    dist_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        side = states[start_idx:end_idx,0]<40
        state_slice = states[start_idx:end_idx,2:20]
        dist_idx = np.argpartition(state_slice,6,axis=-1)
        mean_dists = np.mean(np.take_along_axis(state_slice,dist_idx,axis=-1)[:,:6],1)
        dist_left.append(mean_dists[side])
        dist_right.append(mean_dists[~side])
    distances_C.append(np.hstack(dist_left))
    distances_C.append(np.hstack(dist_right))
    
plt.figure()    
plt.violinplot([np.hstack(distances_A_left),np.hstack(distances_A_right)],[0.8,1.2],showmeans=True)
plt.violinplot([distances_C[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(distances_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Mean of smallest 6 distances to walls at 3Hz [cm]')
plt.ylim(0,20)
plt.title('Phase C')

