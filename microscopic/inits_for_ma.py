# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:36:42 2024

Determine MA initial points.

@author: amaes
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as stats
import os

file = open('data/states_20.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open('data/episode_ends_20.pickle', 'rb')
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

plt.figure()
plt.scatter(avg_locs[:,0],avg_locs[:,1])

#how do i get the coordinate where i get most samples? except this dumb loop
centroid = [0,0]
distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
curr_max = len(np.where(distances<5)[0])
for i in range(0,40):
    for j in range(-10,10):
        centroid_ = [i,j]
        distances = np.sqrt( (avg_locs[:,0]-centroid_[0])**2 + (avg_locs[:,1]-centroid_[1])**2 )
        if len(np.where(distances<5)[0]) > curr_max:
            centroid = centroid_
            curr_max = len(np.where(distances<5)[0])

distances = np.sqrt( (avg_locs[:,0]-centroid[0])**2 + (avg_locs[:,1]-centroid[1])**2 )
idx_ =  np.where(distances<5)[0]


