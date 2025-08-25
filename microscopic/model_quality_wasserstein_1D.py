# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 02:49:53 2025

1D version

@author: amade
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import ot


def seconds_distance(model, centroid, dataset, nb_steps, side):

    file = open('data/states_'+dataset+'.pickle', 'rb')
    states = pickle.load(file)
    file.close()
    
    file = open('data/episode_ends_'+dataset+'.pickle', 'rb')
    epi_ends = pickle.load(file)
    file.close()
    
    trajectories = []
    avg_locs = []
    for i in range(43):
        if i==0:
            start_idx = 0
        else:
            start_idx = epi_ends[i-1]
        end_idx = epi_ends[i]
        series = states[start_idx:end_idx,0:2]
        binary = 1*(series[:,0]>=40)
        idx = np.where((binary[1:]-binary[0:-1]!=0) & (np.abs(series[1:,1]+2)<3))[0]
                
        trajectories.append(states[1+start_idx+idx[0]:1+start_idx+idx[0]+90,0:2])   
        avg_locs.append(states[start_idx+idx[0]+1,3:5])
    
        for j in range(1,np.size(idx)):
            if end_idx-start_idx-idx[j]>90 and idx[j]>idx[j-1]+90:
                trajectories.append(states[1+start_idx+idx[j]:1+start_idx+idx[j]+90,0:2])   
                avg_locs.append(states[start_idx+idx[j]+1,3:5])
    
    avg_locs = np.vstack(avg_locs)
    distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
    idx_ = np.where(distances<5)[0]
    
    data = np.zeros((len(idx_),90,2))
    for i in range(len(idx_)):
        data[i,:,:] = trajectories[idx_[i]]
      
    wasser = np.zeros((nb_steps,))
    data_files = os.listdir('./data_model_curves/'+model+'/samples')
    nb_samples = len([name for name in data_files])
    samples = np.zeros((nb_samples,2,90,2)) #this is for 16 different cases, 30 seconds and x/y axis
    for i in range(nb_samples):
        file = open('data_model_curves/'+model+'/samples/'+data_files[i], 'rb')
        temp = pickle.load(file)
        file.close()
        samples[i,:,:,:] = temp['actions']
        
    for p in range(nb_steps):    
        wasser[p] = ot.emd2_1d(data[:,p,0], samples[:,side,p,0], metric='minkowski')
    
    return wasser
    

model_list = ['t_406_','t_400_','t_401_', 't_402_']
centroid_list = [[16,0], [16,1], [21,0], [30,-1]]
dataset_list = ['40','30','20', '10']
nb_steps = 15
clrs = ['b','g','r','y']

fig, ax = plt.subplots()
for i in range(len(model_list)):
    print('Processing model '+model_list[i])
    dists = np.zeros((6,nb_steps))
    for j in range(6):
        dists[j,:] = seconds_distance(model_list[i]+str(j), centroid_list[i], dataset_list[i], nb_steps, 0)
    means = np.mean(dists,axis=0)
    stds = np.std(dists,axis=0)
    ax.plot(np.linspace(0,nb_steps//3,nb_steps), means, label=model_list[i], c=clrs[i])
    ax.fill_between(np.linspace(0,nb_steps//3,nb_steps), means-stds/np.sqrt(6), means+stds/np.sqrt(6) ,alpha=0.3, facecolor=clrs[i])
                    
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Time (s)')
plt.ylabel('1D Wasserstein distance')
plt.title('Comparing models of mice passing through the center from the left side')
plt.savefig('figures/model_quality/seconds_distances_1D_400L.pdf', format="pdf", bbox_inches="tight")
plt.legend()



centroid_list = [[64,-2], [64,-3], [57,1], [52,-2]]

fig, ax = plt.subplots()
for i in range(len(model_list)):
    print('Processing model '+model_list[i])
    dists = np.zeros((6,nb_steps))
    for j in range(6):
        dists[j,:] = seconds_distance(model_list[i]+str(j), centroid_list[i], dataset_list[i], nb_steps, 1)
    means = np.mean(dists,axis=0)
    stds = np.std(dists,axis=0)
    ax.plot(np.linspace(0,nb_steps//3,nb_steps), means, label=model_list[i], c=clrs[i])
    ax.fill_between(np.linspace(0,nb_steps//3,nb_steps), means-stds/np.sqrt(6), means+stds/np.sqrt(6) ,alpha=0.3, facecolor=clrs[i])
                    
ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Time (s)')
plt.ylabel('1D Wasserstein distance')
plt.title('Comparing models of mice passing through the center from the right side')
plt.savefig('figures/model_quality/seconds_distances_1D_400R.pdf', format="pdf", bbox_inches="tight")
plt.legend()



