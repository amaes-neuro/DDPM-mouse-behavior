# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:37:56 2025

Script analyzing inferred TMT dynamics.
Is it reliable? -> multiple runs give similar result?
Is there a relationship with movement?

@author: amade
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import shapely
import pandas as pd

dataset_name = 'balanced4_x1'
dataset_path='data_dynamic_tmt/'

iteration = 25 #it 84, it 56 good opposite examples

file = open(dataset_path+'t_401_2/naive/dynamic_tmt_'+str(iteration)+'.pickle', 'rb')
tmt_vals1 = pickle.load(file)
file.close()

idx = [1,7,16,25,34,43,50,59,68,77,86]

file = open(dataset_path+'t_401_heldout0/dynamic_tmt_'+str(iteration)+'.pickle', 'rb')
tmt_vals2 = pickle.load(file)
file.close()

dataset_path='data/'

file = open(dataset_path+'exploration_measure.pickle', 'rb')
exploration = pickle.load(file)
file.close()

file = open(dataset_path+'states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open(dataset_path+'actions_'+dataset_name+'.pickle', 'rb')
actions = pickle.load(file) # Marks one-past the last index for each episode
file.close()

file = open(dataset_path+'episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()
states = states[episode_ends[iteration-1]:episode_ends[iteration],:] 
actions = actions[episode_ends[iteration-1]:episode_ends[iteration],:] 

tmt_vals1 = tmt_vals1[0:-11:10]
tmt_vals2 = tmt_vals2[0:-11:10]

plt.figure()
plt.scatter(tmt_vals1,tmt_vals2)

#plt.figure()
#plt.plot((tmt_vals1-np.mean(tmt_vals1))/np.max(tmt_vals1-np.mean(tmt_vals1)))
#veloc = np.sign(np.gradient(states[:,0]))*np.sqrt(np.gradient(states[:,0])**2+np.gradient(states[:,1])**2)
#plt.plot((veloc[0:-11:10]-np.mean(veloc[0:-11:10]))/np.max(veloc[0:-11:10]-np.mean(veloc[0:-11:10])),'--')

temp = pd.DataFrame(tmt_vals1)
smooth = temp.ewm(alpha=1/5, adjust=False).mean()
tmt_vals1_smooth = smooth.to_numpy()
temp = pd.DataFrame(tmt_vals2)
smooth = temp.ewm(alpha=1/5, adjust=False).mean()
tmt_vals2_smooth = smooth.to_numpy()

plt.figure()    
plt.plot(tmt_vals1_smooth)
plt.plot(tmt_vals2_smooth)
plt.hlines(states[0,-1], 0, tmt_vals1_smooth.shape[0],linestyle='--',color='r')

#plt.figure()
#plt.plot(tmt_vals1)
#plt.plot(tmt_vals1_smooth)



fig, ax1 = plt.subplots()
t = np.arange(0, tmt_vals1_smooth.shape[0]/20, 1/20)
color = 'tab:red'
ax1.set_xlabel('time (min)')
ax1.set_ylabel('Inferred TMT', color=color)
ax1.plot(t, tmt_vals1_smooth, color=color)
ax1.tick_params(axis='y', labelcolor=color)
plt.hlines(states[0,-1], 0, np.max(t),linestyle='--',color='r')

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Horizontal location (cm)', color=color)  # we already handled the x-label with ax1
ax2.plot(t, states[0:-11:10,0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


"""
fig, ax1 = plt.subplots()
t = np.arange(0, tmt_vals1_smooth.shape[0]/20, 1/20)
color = 'tab:red'
ax1.set_xlabel('time (min)')
ax1.set_ylabel('Inferred TMT', color=color)
ax1.plot(t, tmt_vals1_smooth, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Exploration measure', color=color)  # we already handled the x-label with ax1
ax2.plot(t, exploration[iteration+36][0:-11:10], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
"""





