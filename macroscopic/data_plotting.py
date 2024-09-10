# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:08:08 2024

Some plotting of the raw data.

@author: amaes
"""

#I need to decide how the switching mechanism is implemented. 
#It is not logical that most switching happens within one second?

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import scipy.stats
import scipy.optimize
from matplotlib.lines import Line2D


dataset_path='data/'
# read from pickle files
file = open(dataset_path+'actions_M.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'states_M.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends_M.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()


#phase A
concentrations = np.array([8,10,9,9,7])
nb_A = 43
sides_A = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):        
        if i == 0:
            start_idx = 0
        else:
            start_idx = episode_ends[np.sum(concentrations[0:i])+j-1]
        end_idx = episode_ends[np.sum(concentrations[0:i])+j]   
        locs = states[start_idx:end_idx,0]
        idx = np.where(locs[1:]!=locs[0:-1])[0]
        for k in range(len(idx)-1):
            if locs[idx[k]] == 1:
                side_left.append(idx[k+1]-idx[k])
            else:
                side_right.append(idx[k+1]-idx[k])
    sides_A.append(np.hstack(side_left))
    sides_A.append(np.hstack(side_right))

#phase B
sides_B = []
for i in range(5):
    side_left = []
    side_right = []
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

#phase C
sides_C = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):
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
    
#PLOTS THE DWELL TIMES
colors = ['olive','olivedrab','yellowgreen']
fig, axs = plt.subplots(3, 5,sharex=True,sharey=True,)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time spent in side of cage")
plt.ylabel("ECDF")
for i in range(5):
    sns.ecdfplot(ax=axs[0,i],data=sides_A[2*i],color=colors[0],linestyle='--',label='R')
    sns.ecdfplot(ax=axs[0,i],data=sides_A[2*i+1],color=colors[0],label='L')
    axs[0,i].set_xlim(0,100)
    axs[0,i].set_ylabel('')

    sns.ecdfplot(ax=axs[1,i],data=sides_B[2*i],color=colors[1],linestyle='--',label='R')
    sns.ecdfplot(ax=axs[1,i],data=sides_B[2*i+1],color=colors[1],label='L')
    axs[1,i].set_xlim(0,100)
    axs[1,i].set_ylabel('')

    sns.ecdfplot(ax=axs[2,i],data=sides_C[2*i],color=colors[2],linestyle='--',label='R')
    sns.ecdfplot(ax=axs[2,i],data=sides_C[2*i+1],color=colors[2],label='L')
    axs[2,i].set_xlim(0,100)
    axs[2,i].set_ylabel('')
plt.savefig('data_figures/dwell_times_data.pdf', format="pdf", bbox_inches="tight")


#PLOTS THE PARAMETERS OF EXPONENTIAL DISTRIBUTIONS FIT TO THE SIDE DWELL TIMES
def func(x, a):
    return 1 - np.exp(-a * x)

exp_params = np.zeros((15,5))
for j in range(2):
    for i in range(5):
        temp = scipy.stats.ecdf(sides_A[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i,j] = popt[0]
        
        temp = scipy.stats.ecdf(sides_B[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i+5,j] = popt[0]

        temp = scipy.stats.ecdf(sides_C[2*i+j])
        xdata = temp.cdf.quantiles
        ydata = temp.cdf.probabilities
        popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
        exp_params[i+10,j] = popt[0]

custom_lines = [Line2D([0], [0], color=colors[0]),
                Line2D([0], [0],color=colors[1]),
                Line2D([0], [0],color=colors[2])]

fig, ax = plt.subplots()
ax.plot(np.arange(5)+6,1./exp_params[0:5,0],color=colors[0],linestyle='--')
ax.plot(np.arange(5)+6,1./exp_params[5:10,0],color=colors[1],linestyle='--')
ax.plot(np.arange(5)+6,1./exp_params[10:15,0],color=colors[2],linestyle='--')
ax.plot(1./exp_params[0:5,1],color=colors[0])
ax.plot(1./exp_params[5:10,1],color=colors[1])
ax.plot(1./exp_params[10:15,1],color=colors[2])
ax.set_xticks(np.arange(11), ['1', '3', '10','30', '90', '', '1', '3', '10','30', '90'])
ax.set_xlabel('TMT concentration')
ax.set_ylabel('Exponential distribution parameter fit [s]')
ax.legend(custom_lines, ['Phase A', 'Phase B', 'Phase C'])
plt.savefig('data_figures/dwell_times_data_fit.pdf', format="pdf", bbox_inches="tight")


#PLOTS THE FRACTION OF TIME THE MICE ARE ON THE LEFT SIDE OF THE CAGE
T = 830 #this is 14 minutes at 1Hz, each trial has a variable length so we have to cut it off to the minimum
time_evol_A = np.zeros((5,T))
time_evol_B = np.zeros((5,T))
time_evol_C = np.zeros((5,T))
concentrations = np.array([8,10,9,9,7])

#get location data phase A
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        if i == 0:
            start_idx = 0
        else:
            start_idx = episode_ends[np.sum(concentrations[0:i])+j-1]
        ev_[j,:] = states[start_idx:start_idx+T,0]

    time_evol_A[i,:] = np.mean(ev_,axis=0)
    
#get location data phase B
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
        ev_[j,:] = states[start_idx:start_idx+T,0]

    time_evol_B[i,:] = np.mean(ev_,axis=0)
    
#get location data phase C
for i in range(5):
    ev_ = np.zeros((concentrations[i],T))
    for j in range(concentrations[i]):
        start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
        ev_[j,:] = states[start_idx:start_idx+T,0]

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
#plt.vlines(window_step*((T-window_length)//window_step)/window_length,0,1,linestyles='--')
#plt.vlines(window_step*(2*((T-window_length)//window_step))/window_length,0,1,linestyles='--')
plt.ylabel('Fraction of time spent in left side of cage')    
plt.xlabel('Time (mins)')
plt.savefig('data_figures/time_data.pdf', format="pdf", bbox_inches="tight")




#PLOTS THE MARKOV TRANSITION MATRICES
steps = 4
window = int(T/steps)
markov_A = np.zeros((5,2,2,steps))
stationary_A = np.zeros((5,steps))
for i in range(5):
    for h in range(steps):
        for j in range(concentrations[i]):
            if i == 0:
                start_idx = 0
            else:
                start_idx = episode_ends[np.sum(concentrations[0:i])+j-1]
        
            state_sequence = states[start_idx+window*h:start_idx+window*(h+1),0]
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                markov_A[i,int(k),int(l),h]+=1    

        markov_A[i,:,:,h] = (markov_A[i,:,:,h].T/np.sum(markov_A[i,:,:,h],axis=1)).T
        a,b = np.linalg.eig(markov_A[i,:,:,h].T)
        idx = np.where(a==1)[0][0]
        stationary_A[i,h] = (b[:,idx]/np.sum(b[:,idx]))[1]

markov_B = np.zeros((5,2,2,steps))
stationary_B = np.zeros((5,steps))
for i in range(5):
    for h in range(steps):
        for j in range(concentrations[i]):
            start_idx = episode_ends[nb_A+np.sum(concentrations[0:i])+j-1]
            state_sequence = states[start_idx+window*h:start_idx+window*(h+1),0]
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                markov_B[i,int(k),int(l),h]+=1    

        markov_B[i,:,:,h] = (markov_B[i,:,:,h].T/np.sum(markov_B[i,:,:,h],axis=1)).T
        a,b = np.linalg.eig(markov_B[i,:,:,h].T)
        idx = np.where(a==1)[0][0]
        stationary_B[i,h] = (b[:,idx]/np.sum(b[:,idx]))[1]


markov_C = np.zeros((5,2,2,steps))
stationary_C = np.zeros((5,steps))
for i in range(5):
    for h in range(steps):
        for j in range(concentrations[i]):
            start_idx = episode_ends[nb_A+43+np.sum(concentrations[0:i])+j-1]
            state_sequence = states[start_idx+window*h:start_idx+window*(h+1),0]
            for (k,l) in zip(state_sequence,state_sequence[1:]):
                markov_C[i,int(k),int(l),h]+=1    

        markov_C[i,:,:,h] = (markov_C[i,:,:,h].T/np.sum(markov_C[i,:,:,h],axis=1)).T
        a,b = np.linalg.eig(markov_C[i,:,:,h].T)
        idx = np.where(a==1)[0][0]
        stationary_C[i,h] = (b[:,idx]/np.sum(b[:,idx]))[1]

"""
plt.figure()
for i in range(5):
    plt.plot(np.arange(steps),markov_A[i,1,1,:],color=colors[i])
    plt.plot(np.arange(steps)+steps,markov_B[i,1,1,:],color=colors[i])
    plt.plot(np.arange(steps)+2*steps,markov_C[i,1,1,:],color=colors[i])
    
    plt.plot(np.arange(steps),markov_A[i,0,1,:],color=colors[i],linestyle='--')
    plt.plot(np.arange(steps)+steps,markov_B[i,0,1,:],color=colors[i],linestyle='--')
    plt.plot(np.arange(steps)+2*steps,markov_C[i,0,1,:],color=colors[i],linestyle='--')
plt.vlines(steps,0,1,linestyles='--')
plt.vlines(2*steps,0,1,linestyles='--')
plt.ylabel('Markov transition probability')    
plt.xlabel('Time (mins)')

plt.figure()
for i in range(5):
    plt.plot(np.arange(steps),markov_A[i,0,0,:],color=colors[i])
    plt.plot(np.arange(steps)+steps,markov_B[i,0,0,:],color=colors[i])
    plt.plot(np.arange(steps)+2*steps,markov_C[i,0,0,:],color=colors[i])
    
    plt.plot(np.arange(steps),markov_A[i,1,0,:],color=colors[i],linestyle='--')
    plt.plot(np.arange(steps)+steps,markov_B[i,1,0,:],color=colors[i],linestyle='--')
    plt.plot(np.arange(steps)+2*steps,markov_C[i,1,0,:],color=colors[i],linestyle='--')
plt.vlines(steps,0,1,linestyles='--')
plt.vlines(2*steps,0,1,linestyles='--')
plt.ylabel('Markov transition probability')    
plt.xlabel('Time (mins)')
"""

fig, ax = plt.subplots(2,1,sharex=True,sharey=True)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel('Markov self-transition probability')    
plt.xlabel('Time (mins)')
for i in range(5):
    ax[0].plot(window/60*np.arange(steps),markov_A[i,1,1,:],color=colors[i])
    ax[0].plot(window/60*(np.arange(steps)+steps),markov_B[i,1,1,:],color=colors[i])
    ax[0].plot(window/60*(np.arange(steps)+2*steps),markov_C[i,1,1,:],color=colors[i])

    ax[1].plot(window/60*np.arange(steps),markov_A[i,0,0,:],color=colors[i],linestyle='--')
    ax[1].plot(window/60*(np.arange(steps)+steps),markov_B[i,0,0,:],color=colors[i],linestyle='--')
    ax[1].plot(window/60*(np.arange(steps)+2*steps),markov_C[i,0,0,:],color=colors[i],linestyle='--')
plt.savefig('data_figures/time_data_markov.pdf', format="pdf", bbox_inches="tight")

"""
#plot stationary prob eigs instead of transition probs
plt.figure()
for i in range(5):
    plt.plot(window/60*np.arange(steps),stationary_A[i,:],color=colors[i])
    plt.plot(window/60*(np.arange(steps)+steps),stationary_B[i,:],color=colors[i])
    plt.plot(window/60*(np.arange(steps)+2*steps),stationary_C[i,:],color=colors[i])
    
plt.vlines(window/60*steps,0,1,linestyles='--')
plt.vlines(window/60*2*steps,0,1,linestyles='--')
plt.ylabel('Markov stationary chance in left side')    
plt.xlabel('Time (mins)')
"""



