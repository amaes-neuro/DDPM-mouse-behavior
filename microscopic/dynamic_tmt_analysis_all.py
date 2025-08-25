# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 05:40:31 2025

Analysis of inferred TMT on multiple mice.

@author: amade
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import shapely
import os
import pandas as pd

model = 't_401_2'
dataset_name = 'balanced4_x1'
dataset_path='data_dynamic_tmt/'+model+'/naive'

data_files = os.listdir('./'+dataset_path)
nb_points = len([name for name in data_files])


dataset_path_='data/'

file = open(dataset_path_+'exploration_measure.pickle', 'rb')
exploration = pickle.load(file)
file.close()

file = open(dataset_path_+'exploration_random.pickle', 'rb')
exploration_rand = pickle.load(file)
file.close()

file = open(dataset_path_+'states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open(dataset_path_+'actions_'+dataset_name+'.pickle', 'rb')
actions = pickle.load(file) # Marks one-past the last index for each episode
file.close()

file = open(dataset_path_+'episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

loc_corrs = []
tmt_all = np.zeros((93,330))
tmts = np.zeros((len(episode_ends),))
for iteration in range(nb_points):
    file = open(dataset_path+'/'+data_files[iteration], 'rb')
    grid_id = data_files[iteration].removesuffix('.pickle')
    grid_id = grid_id.removeprefix('dynamic_tmt_')
    grid_id = int(grid_id)

    tmt_vals = pickle.load(file)
    file.close()
    tmt_vals = tmt_vals[0:-11:10]
    tmt_all[grid_id,0:tmt_vals.shape[0]] = tmt_vals

    expl = exploration[36+grid_id][0:-11:10]
    expl_rand = exploration_rand[36+grid_id][0:-11:10]
    idx_expl = np.where(expl == 0)[0]
    if grid_id == 0:
        states_ = states[0:episode_ends[grid_id],:] 
        actions_ = actions[0:episode_ends[grid_id],:] 
    else:        
        states_ = states[episode_ends[grid_id-1]:episode_ends[grid_id],:] 
        actions_ = actions[episode_ends[grid_id-1]:episode_ends[grid_id],:] 


    temp = pd.DataFrame(tmt_vals)  
    smooth = temp.ewm(alpha=1/5, adjust=False).mean()
    tmt_vals_smooth = smooth.to_numpy()
    temp1 = pd.DataFrame(states_[0:-11:10,0])
    smooth1 = temp1.ewm(alpha=1/5, adjust=False).mean()
    smooth1 = smooth1.to_numpy()
    #loc_corrs.append(np.array([grid_id,np.corrcoef(tmt_vals_smooth.T,np.sqrt(states_[0:-11:10,0]**2+states_[0:-11:10,1]**2))[0,1],states_[0,-1],np.std(tmt_vals_smooth)]))
    loc_corrs.append(np.array([grid_id,np.corrcoef(tmt_vals_smooth[idx_expl[-1]:].T,smooth1[idx_expl[-1]:].T)[0,1],states_[0,-1],np.std(tmt_vals_smooth),np.mean(tmt_vals_smooth),np.corrcoef(tmt_vals_smooth[idx_expl[-1]:].T,expl[idx_expl[-1]:])[0,1],np.corrcoef(smooth1[idx_expl[-1]:].T,expl[idx_expl[-1]:])[0,1],np.corrcoef(smooth1[idx_expl[-1]:].T,expl_rand[idx_expl[-1]:])[0,1]]))

    tmts[grid_id] = states_[0,-1]
    
idx_high = np.where(tmts>0.24)[0]
idx_low = np.where( (tmts<=0.24) & (tmts>0.) )[0]


fig, ax = plt.subplots()

tmt_all = tmt_all[:,0:250]

for i in range(93):
    if ( tmt_all[i,:] == np.zeros((1,250)) ).all():
        tmt_all[i,:] = np.nan  
        
means = np.nanmean(tmt_all[idx_high[0:idx_high.shape[0]//2],:],axis=0)
stds = np.nanstd(tmt_all[idx_high[0:idx_high.shape[0]//2],:],axis=0)
ax.plot(np.linspace(0,15,250), means, label='Phase B', c='olivedrab')
ax.fill_between(np.linspace(0,15,250), means-stds/np.sqrt(idx_high.shape[0]//2), means+stds/np.sqrt(idx_high.shape[0]//2) ,alpha=0.3, facecolor='olivedrab')

means = np.nanmean(tmt_all[idx_low[0:idx_low.shape[0]//2],:],axis=0)
stds = np.nanstd(tmt_all[idx_low[0:idx_low.shape[0]//2],:],axis=0)
ax.plot(np.linspace(0,15,250), means,'--', c='olivedrab')
ax.fill_between(np.linspace(0,15,250), means-stds/np.sqrt(idx_low.shape[0]//2), means+stds/np.sqrt(idx_low.shape[0]//2) ,alpha=0.3, facecolor='olivedrab')

means = np.nanmean(tmt_all[idx_high[idx_high.shape[0]//2:],:],axis=0)
stds = np.nanstd(tmt_all[idx_high[idx_high.shape[0]//2:],:],axis=0)
ax.plot(np.linspace(0,15,250), means, label='Phase C', c='yellowgreen')
ax.fill_between(np.linspace(0,15,250), means-stds/np.sqrt(idx_high.shape[0]//2), means+stds/np.sqrt(idx_high.shape[0]//2) ,alpha=0.3, facecolor='yellowgreen')

means = np.nanmean(tmt_all[idx_low[idx_low.shape[0]//2:],:],axis=0)
stds = np.nanstd(tmt_all[idx_low[idx_low.shape[0]//2:],:],axis=0)
ax.plot(np.linspace(0,15,250), means, '--', c='yellowgreen')
ax.fill_between(np.linspace(0,15,250), means-stds/np.sqrt(idx_low.shape[0]//2), means+stds/np.sqrt(idx_low.shape[0]//2) ,alpha=0.3, facecolor='yellowgreen')

ax.spines[['right', 'top']].set_visible(False)
plt.xlabel('Time (min)')
plt.ylabel('Inferred TMT')
plt.legend()



temp = np.vstack(loc_corrs)
idx = np.where(temp[:,0]>=50)[0]
idx_ = np.where((temp[:,0]>=7) & (temp[:,0]<50))[0]
idx_0 = np.where(temp[:,0]<7)[0]
plt.figure()
plt.scatter(temp[idx,1],temp[idx,2])
plt.scatter(temp[idx_,1],temp[idx_,2])
plt.scatter(temp[idx_0,1],temp[idx_0,2])

fig, ax = plt.subplots()
ax.scatter(temp[idx,4],temp[idx,2],color='yellowgreen',label='Phase C')
ax.scatter(temp[idx_,4],temp[idx_,2],color='olivedrab',label='Phase B')
ax.scatter(temp[idx_0,4],temp[idx_0,2],color='olive',label='Phase A')
ax.plot(np.linspace(0,0.8,10),np.linspace(0,0.8,10),linestyle='--',color='black')
plt.ylabel('Correct TMT value')
plt.xlabel('Avg inferred TMT value')
ax.spines[['right', 'top']].set_visible(False)
plt.legend()


corrs = []
corrs.append(temp[idx_0,1])
corrs.append(temp[idx_,1])
corrs.append(temp[idx,1])
w = 0.8
x = [i for i in range(3)]
fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in corrs],
       yerr=[np.std(yi) for yi in corrs],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=['Phase A','Phase B','Phase C'],
       color=(0,0,0,0),  # face color transparent
       edgecolor=['olive','olivedrab','yellowgreen'],
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + 0.9*np.random.random(corrs[i].size) * w - w / 2, corrs[i], color='grey')
ax.spines[['right', 'top']].set_visible(False)

plt.ylabel('Correlation x-position and inferred TMT')

#inferring the correct tmt level seems to work for phase B and C but not for phase A 
#the presence of food seems to reduce the estimate for tmt for low tmt values and vice versa for high tmt values (this is contrary to my earlier idea that the presence of food would INCREASE the risk assesment)
#unclear if this would still be like this when more discretization and reducing noise

#can i also infer food? i can actually try to turn food into a non-binary variable
#we know how much each mouse has eaten! so we can turn this from 1 to 1-a where a is amount eaten (a<1)
#actually that is tricky because there is 1) the PRESENCE of food, 2) the hunger level, it would require an additional state right

#i can infer food presence actually very easy, the naive way, however this might double the time required still


