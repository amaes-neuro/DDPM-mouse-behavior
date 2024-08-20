# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:47:09 2024

Compare synthetic data with real data from the macroscopic model.
    1. Distributions of dwell times
    2. Evolution of probability in time
    
This is for the case where there are only four states in the model.

@author: ahm8208
"""



import numpy as np
import pickle
import matplotlib.pyplot as plt


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

#the model
model = 't_M_25'

#phase A
nb_A = 7
sides_A_left = []
sides_A_right = []
sides_synth_A_left = []
sides_synth_A_right = []
for i in range(nb_A):
    file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(i)+'.pickle', 'rb')
    states_synth = pickle.load(file) 
    file.close()
    locs = states_synth[:,0]
    idx = np.where(locs[1:]!=locs[0:-1])[0]
    for j in range(len(idx)-1):
        if locs[idx[j]] == 1:
            sides_synth_A_left.append(idx[j+1]-idx[j])
        else:
            sides_synth_A_right.append(idx[j+1]-idx[j])

    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    locs = states[start_idx:end_idx,0]
    idx = np.where(locs[1:]!=locs[0:-1])[0]
    for j in range(len(idx)-1):
        if  locs[idx[j]] == 1:
            sides_A_left.append(idx[j+1]-idx[j])
        else:
            sides_A_right.append(idx[j+1]-idx[j])

plt.figure()    
plt.violinplot([np.hstack(sides_A_left),np.hstack(sides_A_right)],[0.8,1.2],widths=0.3,showmeans=True)
plt.violinplot([np.hstack(sides_synth_A_left),np.hstack(sides_synth_A_right)],[1.8,2.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(sides_A_left)), 0.8, 1.8,colors='b' , linestyles='dotted')
plt.hlines(np.mean(np.hstack(sides_A_right)), 1.2, 2.2,colors='b' , linestyles='dotted')
plt.xticks(np.arange(2)+1,['Data','Model'])
plt.xlabel('Phase A')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,50)
plt.savefig('figures_M/'+model+'/comparison_phase_A.pdf', format="pdf", bbox_inches="tight")


concentrations = np.array([8,10,9,9,7])
#phase B
sides_B = []
sides_synth_B = []
for i in range(5):
    side_left = []
    side_right = []
    side_synth_left = []
    side_synth_right = []
    for j in range(concentrations[i]):
        file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        locs = states_synth[:,0]
        idx = np.where(locs[1:]!=locs[0:-1])[0]
        for k in range(len(idx)-1):
            if locs[idx[k]] == 1:
                side_synth_left.append(idx[k+1]-idx[k])
            else:
                side_synth_right.append(idx[k+1]-idx[k])
        
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
    sides_synth_B.append(np.hstack(side_synth_left))
    sides_synth_B.append(np.hstack(side_synth_right))
    
plt.figure()    
plt.violinplot([sides_B[i] for i in range(len(sides_B))],[0.8,1.2,2.8,3.2,4.8,5.2,6.8,7.2,8.8,9.2],widths=0.3,showmeans=True)
plt.violinplot([sides_synth_B[i] for i in range(len(sides_synth_B))],[1.8,2.2,3.8,4.2,5.8,6.2,7.8,8.2,9.8,10.2],widths=0.3,showmeans=True)
plt.xticks(np.arange(10)+1,['Data','Model','Data','Model','Data','Model','Data','Model','Data','Model'])
plt.xlabel('Phase B')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,50)
plt.savefig('figures_M/'+model+'/comparison_phase_B.pdf', format="pdf", bbox_inches="tight")


#phase C
sides_C = []
sides_synth_C = []
for i in range(5):
    side_left = []
    side_right = []
    side_synth_left = []
    side_synth_right = []
    for j in range(concentrations[i]):
        file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        locs = states_synth[:,0]
        idx = np.where(locs[1:]!=locs[0:-1])[0]
        for k in range(len(idx)-1):
            if locs[idx[k]] == 1:
                side_synth_left.append(idx[k+1]-idx[k])
            else:
                side_synth_right.append(idx[k+1]-idx[k])
        
        start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
        end_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j]   
        locs = states[start_idx:end_idx,0]
        idx = np.where(locs[1:]!=locs[0:-1])[0]
        for k in range(len(idx)-1):
            if locs[idx[k]] == 1:
                side_left.append(idx[k+1]-idx[k])
            else:
                side_right.append(idx[k+1]-idx[k])
    sides_C.append(np.hstack(side_left))
    sides_C.append(np.hstack(side_right))
    sides_synth_C.append(np.hstack(side_synth_left))
    sides_synth_C.append(np.hstack(side_synth_right))
    
plt.figure()    
plt.violinplot([sides_C[i] for i in range(10)],[0.8,1.2,2.8,3.2,4.8,5.2,6.8,7.2,8.8,9.2],widths=0.3,showmeans=True)
plt.violinplot([sides_synth_C[i] for i in range(10)],[1.8,2.2,3.8,4.2,5.8,6.2,7.8,8.2,9.8,10.2],widths=0.3,showmeans=True)
plt.xticks(np.arange(10)+1,['Data','Model','Data','Model','Data','Model','Data','Model','Data','Model'])
plt.xlabel('Phase C')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,70)
plt.savefig('figures_M/'+model+'/comparison_phase_C.pdf', format="pdf", bbox_inches="tight")


#time evolution
T = 830 #this is 14 minutes at 1Hz, each trial has a variable length so we have to cut it off to the minimum
time_evol_A = np.zeros((5,T))
time_evol_B = np.zeros((5,T))
time_evol_C = np.zeros((5,T))
concentrations = np.array([8,10,9,9,7])

#get location data phase A
for i in range(5):
    ev_ = np.zeros((concentrations[4],T))
    for j in range(concentrations[4]):
        file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        ev_[j,:] = states_synth[0:T,0] 
    time_evol_A[i,:] = np.mean(ev_,axis=0)
    
#get location data phase B
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()

        ev_[j,:] = states_synth[0:T,0] 
    time_evol_B[i,:] = np.mean(ev_,axis=0)
    
#get location data phase C
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        file = open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()

        ev_[j,:] = states_synth[0:T,0] 
    time_evol_C[i,:] = np.mean(ev_,axis=0)
    
   
#set params
window_length = 60 #in nb of time steps (1Hz)
window_step = 60 #less overlap in windows than if you would take convolution

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
    plt.plot(window_step*np.arange(((T-window_length)//window_step))/window_length,phase_A[i,:],color=colors[i])
    plt.plot(window_step*np.arange(((T-window_length)//window_step),2*((T-window_length)//window_step))/window_length,phase_B[i,:],color=colors[i])
    plt.plot(window_step*np.arange(2*((T-window_length)//window_step),3*((T-window_length)//window_step))/window_length,phase_C[i,:],color=colors[i])
plt.vlines(window_step*((T-window_length)//window_step)/window_length,0,1,linestyles='--')
plt.vlines(window_step*(2*((T-window_length)//window_step))/window_length,0,1,linestyles='--')
plt.ylabel('fraction in left side of cage')    
plt.xlabel('Time (mins)')
plt.savefig('figures_M/'+model+'/time_evolution_synth.pdf', format="pdf", bbox_inches="tight")


