# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:46:05 2024

Visualize distribution of direction change and cage dwell times.

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


"""
Distributions of time spent going in the same horizontal direction.
"""

#phase A
direction_A = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    dir_states_added = states[start_idx:end_idx,20] +  states[start_idx:end_idx,21]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    direction_A.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(direction_A),50,(0,10),True)
plt.xlabel('Time before changing horizontal direction [s]')
plt.ylabel('Density')
plt.title('Phase A')

#phase B
direction_B = []
for i in range(int(len(episode_ends)/3)):
    start_idx = episode_ends[int(len(episode_ends)/3)-1]
    if i>0:
        start_idx = episode_ends[int(len(episode_ends)/3)+i-1]
    end_idx = episode_ends[int(len(episode_ends)/3)+i]
    dir_states_added = states[start_idx:end_idx,20] +  states[start_idx:end_idx,21]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    direction_B.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(direction_B),50,(0,10),True)
plt.xlabel('Time before changing horizontal direction [s]')
plt.ylabel('Density')
plt.title('Phase B')

#phase C
direction_C = []
for i in range(int(len(episode_ends)/3)):
    start_idx = episode_ends[int(2*len(episode_ends)/3)-1]
    if i>0:
        start_idx = episode_ends[int(2*len(episode_ends)/3)+i-1]
    end_idx = episode_ends[int(2*len(episode_ends)/3)+i]
    dir_states_added = states[start_idx:end_idx,20] +  states[start_idx:end_idx,21]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    direction_C.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(direction_C),50,(0,10),True)
plt.xlabel('Time before changing horizontal direction [s]')
plt.ylabel('Density')
plt.title('Phase C')


# different concentrations, phase B
concentrations = np.array([8,10,9,9,7])
direction_B = []
for i in range(5):
    side = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,20] +  states[start_idx:end_idx,21]
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
        side.append(dir_states_added[dir_change_idx]/3)
    direction_B.append(np.hstack(side))
    
plt.figure()    
plt.violinplot([direction_B[0],direction_B[1],direction_B[2],direction_B[3],direction_B[4]],showmeans=True)
plt.hlines(np.mean(np.hstack(direction_A)),0.8,5.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(5)+1,['1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of horizontal direction times [s]')
plt.ylim(0,2)
plt.title('Phase B')


#different concentrations, phase C
direction_C = []
for i in range(5):
    side = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,20] +  states[start_idx:end_idx,21]
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
        side.append(dir_states_added[dir_change_idx]/3)
    direction_C.append(np.hstack(side))
    
plt.figure()    
plt.violinplot([direction_C[0],direction_C[1],direction_C[2],direction_C[3],direction_C[4]],showmeans=True)
plt.hlines(np.mean(np.hstack(direction_A)),0.8,5.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(5)+1,['1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of horizontal direction times [s]')
plt.ylim(0,2)
plt.title('Phase C')


#phase A
direction_A_left = []
direction_A_right = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    dir_states_added = states[start_idx:end_idx,20] 
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
    direction_A_left.append(dir_states_added[dir_change_idx]/3)
    dir_states_added = states[start_idx:end_idx,21] 
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
    direction_A_right.append(dir_states_added[dir_change_idx]/3)


#split left/right direction concentrations, phase B
direction_B = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,20] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_left.append(dir_states_added[dir_change_idx]/3)
        dir_states_added = states[start_idx:end_idx,21] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_right.append(dir_states_added[dir_change_idx]/3)
    direction_B.append(np.hstack(side_left))
    direction_B.append(np.hstack(side_right))
    
plt.figure()    
plt.violinplot([np.hstack(direction_A_left),np.hstack(direction_A_right)],[0.8,1.2],widths=0.3,showmeans=True)
plt.violinplot([direction_B[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(direction_A)),0.8,6.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of direction times [s]')
plt.ylim(0,2)
plt.title('Phase B')


#split left/right direction concentrations, phase C
direction_C = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,20] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_left.append(dir_states_added[dir_change_idx]/3)
        dir_states_added = states[start_idx:end_idx,21] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_right.append(dir_states_added[dir_change_idx]/3)
    direction_C.append(np.hstack(side_left))
    direction_C.append(np.hstack(side_right))
    
plt.figure()    
plt.violinplot([np.hstack(direction_A_left),np.hstack(direction_A_right)],[0.8,1.2],widths=0.3,showmeans=True)
plt.violinplot([direction_C[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(direction_A)),0.8,6.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of direction times [s]')
plt.ylim(0,2)
plt.title('Phase C')



"""

Same analysis for the sides of the cage, longer time scale

"""

#phase A
side_A = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    dir_states_added = states[start_idx:end_idx,22] +  states[start_idx:end_idx,23]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    side_A.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(side_A),50,(0,50),True)
plt.xlabel('Time before changing cage side [s]')
plt.ylabel('Density')
plt.title('Phase A')

#phase B
side_B = []
for i in range(int(len(episode_ends)/3)):
    start_idx = episode_ends[int(len(episode_ends)/3)-1]
    if i>0:
        start_idx = episode_ends[int(len(episode_ends)/3)+i-1]
    end_idx = episode_ends[int(len(episode_ends)/3)+i]
    dir_states_added = states[start_idx:end_idx,23] +  states[start_idx:end_idx,22]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    side_B.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(side_B),50,(0,50),True)
plt.xlabel('Time before changing cage side [s]')
plt.ylabel('Density')
plt.title('Phase B')

#phase C
side_C = []
for i in range(int(len(episode_ends)/3)):
    start_idx = episode_ends[int(2*len(episode_ends)/3)-1]
    if i>0:
        start_idx = episode_ends[int(2*len(episode_ends)/3)+i-1]
    end_idx = episode_ends[int(2*len(episode_ends)/3)+i]
    dir_states_added = states[start_idx:end_idx,22] +  states[start_idx:end_idx,23]
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
    side_C.append(dir_states_added[dir_change_idx]/3)

plt.figure()
plt.hist(np.hstack(side_C),50,(0,50),True)
plt.xlabel('Time before changing cage side [s]')
plt.ylabel('Density')
plt.title('Phase C')



# different concentrations, phase B
concentrations = np.array([8,10,9,9,7])
sides_B = []
for i in range(5):
    side = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,22] +  states[start_idx:end_idx,23]
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
        side.append(dir_states_added[dir_change_idx]/3)
    sides_B.append(np.hstack(side))
    
plt.figure()    
plt.violinplot([sides_B[0],sides_B[1],sides_B[2],sides_B[3],sides_B[4]],showmeans=True)
plt.hlines(np.mean(np.hstack(side_A)),0.8,5.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(5)+1,['1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,50)
plt.title('Phase B')


#different concentrations, phase C
sides_C = []
for i in range(5):
    side = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,22] +  states[start_idx:end_idx,23]
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<=0)
        side.append(dir_states_added[dir_change_idx]/3)
    sides_C.append(np.hstack(side))
    
plt.figure()    
plt.violinplot([sides_C[0],sides_C[1],sides_C[2],sides_C[3],sides_C[4]],showmeans=True)
plt.hlines(np.mean(np.hstack(side_A)),0.8,5.2,colors='b',linestyles='dotted')
plt.xticks(np.arange(5)+1,['1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,50)
plt.title('Phase C')


#phase A
sides_A_left = []
sides_A_right = []
for i in range(int(len(episode_ends)/3)):
    start_idx = 0
    if i>0:
        start_idx = episode_ends[i-1]
    end_idx = episode_ends[i]
    dir_states_added = states[start_idx:end_idx,22] 
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
    sides_A_left.append(dir_states_added[dir_change_idx]/3)
    dir_states_added = states[start_idx:end_idx,23] 
    dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
    sides_A_right.append(dir_states_added[dir_change_idx]/3)
    

#split left/right side concentrations, phase B
sides_B = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,22] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_left.append(dir_states_added[dir_change_idx]/3)
        dir_states_added = states[start_idx:end_idx,23] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_right.append(dir_states_added[dir_change_idx]/3)
    sides_B.append(np.hstack(side_left))
    sides_B.append(np.hstack(side_right))
    
plt.figure()    
plt.violinplot([np.hstack(sides_A_left),np.hstack(sides_A_right)],[0.8,1.2],widths=0.3,showmeans=True)
plt.violinplot([sides_B[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(side_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,50)
plt.title('Phase B')


#split left/right side concentrations, phase C
sides_C = []
for i in range(5):
    side_left = []
    side_right = []
    for j in range(concentrations[i]):
        start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])-1]
        if j>0:
            start_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j-1] 
        end_idx = episode_ends[2*np.sum(concentrations)+np.sum(concentrations[0:i])+j]   
        dir_states_added = states[start_idx:end_idx,22] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_left.append(dir_states_added[dir_change_idx]/3)
        dir_states_added = states[start_idx:end_idx,23] 
        dir_change_idx = np.where(dir_states_added[1:]-dir_states_added[0:-1]<0)
        side_right.append(dir_states_added[dir_change_idx]/3)
    sides_C.append(np.hstack(side_left))
    sides_C.append(np.hstack(side_right))
    
plt.figure()  
plt.violinplot([np.hstack(sides_A_left),np.hstack(sides_A_right)],[0.8,1.2],widths=0.3,showmeans=True)
plt.violinplot( [sides_C[i] for i in range(10)],[1.8,2.2,2.8,3.2,3.8,4.2,4.8,5.2,5.8,6.2],widths=0.3,showmeans=True)
plt.hlines(np.mean(np.hstack(side_A)),0.8,6.2,colors='b',linestyle='dotted')
plt.xticks(np.arange(6)+1,['0','1','3','10','30','90'])
plt.xlabel('TMT concentrations levels [$\mu$L]')
plt.ylabel('Distribution of side dwell times [s]')
plt.ylim(0,75)
plt.title('Phase C')

