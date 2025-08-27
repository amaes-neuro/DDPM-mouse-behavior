# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:07:22 2024

Compare experimental data with synthetic data.

@author: amade
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import ot
from scipy.stats import kstest


def trial_distance(model, dataset_name):
    
    dataset_path='data/'
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

    #phase A
    nb_A = 7
    locs_A = []
    synth_locs_A = []
    for i in range(nb_A):
        file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        locs = states_synth[:,0:2]
        synth_locs_A.append(locs)
        
        start_idx = 0
        if i>0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        locs = states[start_idx:end_idx,0:2]
        locs_A.append(locs)
    
    locs_A = np.vstack(locs_A)
    synth_locs_A = np.vstack(synth_locs_A)
    a, b = np.ones((len(locs_A),)) / len(locs_A), np.ones((len(synth_locs_A),)) / len(synth_locs_A)   
    distance_A = ot.sliced_wasserstein_distance(locs_A, synth_locs_A, a, b, 300,p=1)

    concentrations = np.array([8,10,9,9,7])
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
            loc = states_synth[:,0:2]
            synth_locs.append(loc)
            
            start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j]   
            loc = states[start_idx:end_idx,0:2]
            locs.append(loc)
    
        locs_B.append(np.vstack(locs))
        synth_locs_B.append(np.vstack(synth_locs))
    
    distances_B = np.zeros((5,))
    for i in range(5):
        a, b = np.ones((len(locs_B[i]),)) / len(locs_B[i]), np.ones((len(synth_locs_B[i]),)) / len(synth_locs_B[i])   
        distances_B[i] = ot.sliced_wasserstein_distance(locs_B[i], synth_locs_B[i], a, b, 300,p=1)
    
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
            loc = states_synth[:,0:2]
            synth_locs.append(loc)
            
            start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j]   
            loc = states[start_idx:end_idx,0:2]
            locs.append(loc)
    
        locs_C.append(np.vstack(locs))
        synth_locs_C.append(np.vstack(synth_locs))
    
    distances_C = np.zeros((5,))
    for i in range(5):
        a, b = np.ones((len(locs_C[i]),)) / len(locs_C[i]), np.ones((len(synth_locs_C[i]),)) / len(synth_locs_C[i])   
        distances_C[i] = ot.sliced_wasserstein_distance(locs_C[i], synth_locs_C[i], a, b, 300,p=1)

    return distance_A, distances_B, distances_C


def switching_distance(model, dataset_name):
    
    dataset_path='data/'
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

    #phase A
    nb_A = 7
    locs_A_left = []
    synth_locs_A_left = []
    locs_A_right = []
    synth_locs_A_right = []
    for i in range(nb_A):
        file = open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'rb')
        states_synth = pickle.load(file) 
        file.close()
        locs = states_synth[:,0]<40
        idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
        if locs[0]:
            if idx[0]!=0:
                synth_locs_A_left.append(idx[0])
            for k in range(1,len(idx)-1,2):
                synth_locs_A_left.append(idx[k+1]-idx[k])
                synth_locs_A_right.append(idx[k]-idx[k-1])
        else:
            if idx[0]!=0:
                synth_locs_A_right.append(idx[0])
            for k in range(1,len(idx)-1,2):
                synth_locs_A_right.append(idx[k+1]-idx[k])
                synth_locs_A_left.append(idx[k]-idx[k-1])

        
        start_idx = 0
        if i>0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        locs = states[start_idx:end_idx,0]<40
        idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
        if locs[0]:
            if idx[0]!=0:
                locs_A_left.append(idx[0])
            for k in range(1,len(idx)-1,2):
                locs_A_left.append(idx[k+1]-idx[k])
                locs_A_right.append(idx[k]-idx[k-1])
        else:
            if idx[0]!=0:
                locs_A_right.append(idx[0])
            for k in range(1,len(idx)-1,2):
                locs_A_right.append(idx[k+1]-idx[k])
                locs_A_left.append(idx[k]-idx[k-1])
    
    
    distance_A_left = ot.emd2_1d(np.log(locs_A_left), np.log(synth_locs_A_left), metric='minkowski')#kstest(locs_A_left,synth_locs_A_left).statistic#
    distance_A_right = ot.emd2_1d(np.log(locs_A_right), np.log(synth_locs_A_right), metric='minkowski') #kstest(locs_A_right,synth_locs_A_right).statistic#
    distance_A = np.hstack((distance_A_left,distance_A_right))


    concentrations = np.array([8,10,9,9,7])
    #phase B
    locs_B_left = []
    locs_B_right = []
    synth_locs_B_left = []
    synth_locs_B_right = []
    for i in range(5):
        left = []
        right = []
        synth_left = []
        synth_right = []
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            locs = states_synth[:,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0] and len(idx)>0:
                if idx[0]!=0:
                    synth_left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_left.append(idx[k+1]-idx[k])
                    synth_right.append(idx[k]-idx[k-1])
            if locs[0] == 0 and len(idx)>0:
                if idx[0]!=0:
                    synth_right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_right.append(idx[k+1]-idx[k])
                    synth_left.append(idx[k]-idx[k-1])
                
            start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j]   
            locs = states[start_idx:end_idx,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0]:
                if idx[0]!=0:
                    left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    left.append(idx[k+1]-idx[k])
                    right.append(idx[k]-idx[k-1])
            else:
                if idx[0]!=0:
                    right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    right.append(idx[k+1]-idx[k])
                    left.append(idx[k]-idx[k-1])
    
        locs_B_left.append(np.vstack(left))
        locs_B_right.append(np.vstack(right))
        synth_locs_B_left.append(np.vstack(synth_left))
        synth_locs_B_right.append(np.vstack(synth_right))
    
    distances_B = np.zeros((10,))
    for i in range(5):
        distances_B[i] = ot.emd2_1d(np.log(locs_B_left[i]), np.log(synth_locs_B_left[i]), metric='minkowski') #kstest(locs_B_left[i].flatten(),synth_locs_B_left[i].flatten()).statistic#
        distances_B[i+5] = ot.emd2_1d(np.log(locs_B_right[i]), np.log(synth_locs_B_right[i]), metric='minkowski') #kstest(locs_B_right[i].flatten(),synth_locs_B_right[i].flatten()).statistic#
    
    
    #phase C
    locs_C_left = []
    locs_C_right = []
    synth_locs_C_left = []
    synth_locs_C_right = []
    for i in range(5):
        left = []
        right = []
        synth_left = []
        synth_right = []
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            locs = states_synth[:,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0] and len(idx)>0:
                if idx[0]!=0:
                    synth_left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_left.append(idx[k+1]-idx[k])
                    synth_right.append(idx[k]-idx[k-1])
            if locs[0] == 0 and len(idx)>0:
                if idx[0]!=0:
                    synth_right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    synth_right.append(idx[k+1]-idx[k])
                    synth_left.append(idx[k]-idx[k-1])
            
            start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
            end_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j]   
            locs = states[start_idx:end_idx,0]<40
            idx = np.where(locs[1:]!=locs[0:-1] )[0]/3
            if locs[0]:
                if idx[0]!=0:
                    left.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    left.append(idx[k+1]-idx[k])
                    right.append(idx[k]-idx[k-1])
            else:
                if idx[0]!=0:
                    right.append(idx[0])
                for k in range(1,len(idx)-1,2):
                    right.append(idx[k+1]-idx[k])
                    left.append(idx[k]-idx[k-1])
    
        locs_C_left.append(np.vstack(left))
        locs_C_right.append(np.vstack(right))
        if len(synth_left) > 0: #behavioral cloning in phase C does not switch
            synth_locs_C_left.append(np.vstack(synth_left))
        else:
            synth_locs_C_left.append(synth_left)
        if len(synth_right) > 0:
            synth_locs_C_right.append(np.vstack(synth_right))
        else:
            synth_locs_C_right.append(synth_right)
    
    distances_C = np.nan*np.zeros((10,))
    for i in range(5):
        if len(synth_locs_C_left[i])>0 and len(synth_locs_C_right[i])>0 :
            distances_C[i] = ot.emd2_1d(np.log(locs_C_left[i]), np.log(synth_locs_C_left[i]) , metric='minkowski') #kstest(locs_C_left[i].flatten(),synth_locs_C_left[i].flatten()).statistic#
            distances_C[i+5] = ot.emd2_1d(np.log(locs_C_right[i]), np.log(synth_locs_C_right[i]), metric='minkowski')#kstest(locs_C_right[i].flatten(),synth_locs_C_right[i].flatten()).statistic#

    return distance_A, distances_B, distances_C


def minutes_distance(model, dataset_name):

    dataset_path='data/'
    # read from pickle files
    file = open(dataset_path+'data_time_evolution.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    
    #time evolution
    T = 830*3 #this is 14 minutes at 1Hz, each trial has a variable length so we have to cut it off to the minimum
    time_evol_A = np.zeros((5,T))
    time_evol_B = np.zeros((5,T))
    time_evol_C = np.zeros((5,T))
    concentrations = np.array([8,10,9,9,7])
    nb_A = 7
    #get location data phase A
    for i in range(5):
        ev_ = np.zeros((concentrations[4],T))
        for j in range(concentrations[4]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
            ev_[j,:] = states_synth[0:T,0] < 40
        time_evol_A[i,:] = np.mean(ev_,axis=0)
        
    #get location data phase B
    for i in range(5):
        ev_ = np.zeros((concentrations[i],T))
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
    
            ev_[j,:] = states_synth[0:T,0] < 40
        time_evol_B[i,:] = np.mean(ev_,axis=0)
        
    #get location data phase C
    for i in range(5):
        ev_ = np.zeros((concentrations[i],T))
        for j in range(concentrations[i]):
            file = open('data_synthetic/'+model+'/states_synthetic_'+str(nb_A+43+np.sum(concentrations[0:i])+j)+'.pickle', 'rb')
            states_synth = pickle.load(file) 
            file.close()
    
            ev_[j,:] = states_synth[0:T,0] < 40
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
    
    distance_A = np.mean((phase_A[4,:]-data[0][4,:])**2)
    distance_B = np.mean((phase_B-data[1])**2,axis=1)
    distance_C = np.mean((phase_C-data[2])**2,axis=1)
    
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
    
    plt.savefig('figures/time_evolution/micro_'+model+'.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    return distance_A, distance_B, distance_C


model_list_1 = ['t_0','t_1_1','t_2_1','t_7','t_8','t_6']
model_list_2 = ['t_0_1','t_1_2','t_2_2','t_7_1','t_8_1','t_6_1']
dataset_list = ['balanced0','balanced1','balanced2','balanced7','balanced8','balanced4']

model_list_8 = ['t_10_0','t_11_0','t_12_0','t_13_0','t_14_0','t_15_0']
model_list_12 = ['t_20_0','t_21_0','t_22_0','t_23_0','t_24_0','t_25_0']
dataset_list = ['balanced0','balanced1','balanced2','balanced7','balanced8','balanced4']

model_list_TMT = ['t_100_','t_200_','t_300_','t_400_']
dataset_list_TMT = ['balanced4_s','balanced4_b','balanced4_a','balanced4_x']

model_list_400 = ['t_406_','t_400_','t_401_','t_402_','t_403_','t_404_','t_405_','t_500_']
dataset_list_400 = ['balanced4_x6','balanced4_x','balanced4_x1','balanced4_x2','balanced4_x3','balanced4_x4','balanced4_x5','balanced4_x']

model_list_800 = ['t_806_','t_800_','t_801_','t_802_','t_807_']
dataset_list_800 = ['balanced7_x6','balanced7_x0','balanced7_x1','balanced7_x2','balanced7_x1']


save = '800'
model_list = model_list_800
nb_ = len(model_list)
dataset_list = dataset_list_800
trial_distances = []
minutes_distances = []
switch_distances = []
for i in range(nb_):
    print('Processing model '+model_list[i])
    dist1,dist2,dist3 = np.zeros((6,)),np.zeros((6,)),np.zeros((6,))
    for j in range(6):
        a,b,c = trial_distance(model_list[i]+str(j), dataset_list[i])
        dist1[j] =  np.mean(np.hstack((a,b,c)))
        
        a,b,c = minutes_distance(model_list[i]+str(j), dataset_list[i])
        dist2[j] =  np.mean(np.hstack((a,b,c)))
    
        a,b,c = switching_distance(model_list[i]+str(j), dataset_list[i])
        dist3[j] =  np.nanmean(np.hstack((a,b,c)))
        
    trial_distances.append(dist1)
    minutes_distances.append(dist2)
    switch_distances.append(dist3)
    
    
w = 0.8
x = [i for i in range(len(model_list))]
fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in trial_distances],
       yerr=[np.std(yi) for yi in trial_distances],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=model_list,
       color=(0,0,0,0),  # face color transparent
       edgecolor='black',
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + 0.9*np.random.random(trial_distances[i].size) * w - w / 2, trial_distances[i], color='black')
ax.spines[['right', 'top']].set_visible(False)

plt.ylabel('2D Wasserstein distance')
plt.title('Comparing trajectories of entire trials')
plt.savefig('figures/model_quality/trial_distances_'+save+'.pdf', format="pdf", bbox_inches="tight")
plt.show()

    
fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in minutes_distances],
       yerr=[np.std(yi) for yi in minutes_distances],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=model_list,
       color=(0,0,0,0),  # face color transparent
       edgecolor='black',
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + 0.9*np.random.random(minutes_distances[i].size) * w - w / 2, minutes_distances[i], color='black')
ax.spines[['right', 'top']].set_visible(False)

plt.ylabel('Mean squared error')
plt.title('Comparing the minute-averaged time-series of cage side location')
plt.savefig('figures/model_quality/minutes_distances_'+save+'.pdf', format="pdf", bbox_inches="tight")
plt.show() 
    
    
fig, ax = plt.subplots()
ax.bar(x,
       height=[np.mean(yi) for yi in switch_distances],
       yerr=[np.std(yi) for yi in switch_distances],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=model_list,
       color=(0,0,0,0),  # face color transparent
       edgecolor='black',
       )

for i in range(len(x)):
    # distribute scatter randomly across whole width of bar
    ax.scatter(x[i] + 0.9*np.random.random(switch_distances[i].size) * w - w / 2, switch_distances[i], color='black')
ax.spines[['right', 'top']].set_visible(False)

plt.ylabel('1D Wasserstein distance')
plt.title('Comparing distributions of left and right side cage dwell times')
plt.savefig('figures/model_quality/switch_distances_'+save+'.pdf', format="pdf", bbox_inches="tight")
plt.show()  
    
    
