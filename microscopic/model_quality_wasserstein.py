# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:35:54 2024

Establish model quality by comparing the samples from model with data.
Difficulty 1: limited number of samples in data -> pool samples from various tmt concentrations.
Difficulty 2: various points in cage should be samples (not just the central point).
Difficulty 3: 30 second trajectories might be too long (limits amount of samples further).

The MA is very dependent on phase. This is because the speed of mice is A<B<C.
Nb_samples for (40,0), coming from (20,5) is 66 in phase A, 35 in phase B and 21 for phase C.

@author: amaes
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
        
    a, b = np.ones((len(data),)) / len(data), np.ones((len(samples),)) / len(samples)
    for p in range(nb_steps):    
        wasser[p] = ot.sliced_wasserstein_distance(data[:,p,:], samples[:,side,p,:], a, b, 500,p=1)
    
    return wasser
    


model_list = ['t_406_','t_400_','t_401_', 't_402_','t_403_','t_404_','t_405_']
centroid_list = [[16,0], [16,1], [21,0], [30,-1], [16,1], [16,1], [16,1]]
dataset_list = ['40','30','20', '10', '30', '30', '30']
nb_steps = 15
clrs = ['b','g','r','y','black','grey','orange']

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
plt.ylabel('2D Wasserstein distance')
plt.title('Comparing models of mice passing through the center from the left side')
plt.savefig('figures/model_quality/seconds_distances_400L.pdf', format="pdf", bbox_inches="tight")
plt.legend()



centroid_list = [[64,-2], [64,-3], [57,1], [52,-2], [64,-3], [64,-3], [64,-3]]

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
plt.ylabel('2D Wasserstein distance')
plt.title('Comparing models of mice passing through the center from the right side')
plt.savefig('figures/model_quality/seconds_distances_400R.pdf', format="pdf", bbox_inches="tight")
plt.legend()



