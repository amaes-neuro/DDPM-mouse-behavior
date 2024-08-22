# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:16:15 2024\
    
Script to organize the data and save in pickle files. 
Organization files: 
    one long array of all states for all mice
    one long array of all actions for all mice
    array which indicates the last index of each episode 

The files are then read by the dataset class later on.

The action is the side of cage where mouse will be in the next time step.
States (by increasing temporal scale):
    1. Location at current time step (0 or 1)
    2. Time in current side of cage (>0)
    3. Total time elapsed (>0)
    4. Food present (constant, 0 during phase A and B, 1 in phase C)
    5. Concentration (constant, 0 during phase A or >0 in phase B and C)

@author: ahm8208
"""


import pickle
import numpy as np
from tqdm import tqdm



def compute_cum_side_time(location):
    #this function computes the cumulative time in the same side
    sides = np.zeros((len(location),2))
    for i in range(len(location)):
        
        if i>0 and location[i]!=location[i-1]:
            sides[i,0] = 0
            sides[i,1] = sides[i-1,0]
        elif i>0:
            sides[i,0] = sides[i-1,0] + 1
            sides[i,1] = sides[i-1,1]

    return sides

      

def compute_threat(mouse_id, concentration, path, phase, subsample, walls):
    file = open(path+'/points_nose_A'+'_'+concentration+'.pickle', 'rb')    
    points_A = pickle.load(file)
    file.close()
    if phase == 'B':
        file = open(path+'/points_nose_B'+'_'+concentration+'.pickle', 'rb')    
        points = pickle.load(file)
        file.close()
    elif phase == 'C':
        file = open(path+'/points_nose_C'+'_'+concentration+'.pickle', 'rb')   
        points = pickle.load(file)
        file.close()

    points_A = points_A[mouse_id][0::subsample,0]
    points = points[mouse_id][0::subsample,0]
    ratio = np.sum(get_location(points, walls))/len(points)
    ratio_A = np.sum(get_location(points_A, walls))/len(points_A)
    return ratio_A-ratio #does it make more sense to divide those quantities?



def get_location(points, walls):
    locs = np.zeros(np.shape(points))
    if points[0]<(walls[3,0]+walls[2,0])/2:
        locs[0] = 1
    for i in range(1,len(points)):
        if points[i]<(walls[3,0]+walls[2,0])/2:
            locs[i] = 1
        """
        if points[i]>walls[3,0] and locs[i-1]==1:
            locs[i] = 0
        elif points[i]<walls[2,0] and locs[i-1]==0:
            locs[i] = 1
        else:
            locs[i] = locs[i-1]
        """
    return locs


    
def generate_space(concentration, phase, subsample):
    states = []
    actions = []
    idx_e = []
    
    #load data files
    path = 'C:\\Users\\ahm8208\\OneDrive - Northwestern University\\Documents\\behavior_diffusion_project\\data_rotated_shifted'
    file = open(path+'/points_nose_'+phase+'_'+concentration+'.pickle', 'rb')
    points = pickle.load(file)
    file.close()
    file = open(path+'/walls_'+phase+'_'+concentration+'.pickle', 'rb')
    walls = pickle.load(file)
    file.close()

    for i in tqdm(range(len(points))):
        agent_location = get_location(points[i][0::subsample,0], walls[i])#1*(points[i][0::subsample,0] < 40)
        agent_action = agent_location[1:]
        agent_location = agent_location[0:-1]
        agent_sides = compute_cum_side_time(agent_location)
        if phase=='C':
            food_present = np.ones((len(agent_location),1))
        else:
            food_present = np.zeros((len(agent_location),1))
        total_time = np.zeros((len(agent_location),1))
        total_time[:,0] = np.linspace(0,len(agent_location)-1,len(agent_location))
        if phase=='A':
            agent_threat = np.zeros((len(agent_location),1))
        else:
            agent_threat = compute_threat(i, concentration, path, phase, subsample, walls[i])*np.ones((len(agent_location),1))#int(concentration)*np.ones((len(agent_location),1))

        #states_ = np.hstack((np.reshape(agent_location,(np.size(agent_location),1)),np.reshape(agent_sides[:,1],(np.size(agent_location),1)),
        #                     total_time,food_present,agent_threat))
        states_ = np.hstack((np.reshape(agent_location,(np.size(agent_location),1)),
                             total_time,food_present,agent_threat))
        states.append(states_)
        actions.append(np.reshape(agent_action,(np.size(agent_action),1)))
        idx_e.append(len(agent_location))

    print('Training data processed for phase '+phase+' and concentration '+concentration+'...')
    return np.vstack(states), np.vstack(actions), np.array(idx_e)
   
    
  
def main():
    concentrations = ['1','3','10','30','90']
    phases = ['A','B','C']
    states_list = []
    actions_list = []
    idx_ends = []
    for i in range(len(phases)):
        for j in range(len(concentrations)):
            if i == 0 and j<4:
                continue
            else:
                states_add,actions_add,idx_add = generate_space(concentrations[j], phases[i], 30)
                states_list.append(states_add)
                actions_list.append(actions_add)
                idx_ends.append(idx_add)
                
    #try to add one C-90 to make it match? if not i have to think about adding a state  
    #states_add,actions_add,idx_add = generate_space('90', 'C', 30)
    #states_list.append(states_add)
    #actions_list.append(actions_add)
    #idx_ends.append(idx_add)

    #write to files  
    states = np.vstack(states_list)
    actions = np.vstack(actions_list)
    idx_ends = np.hstack(idx_ends)
    with open('data/states_M_balanced10.pickle', 'wb') as file:
        pickle.dump(np.float32(states), file)
    with open('data/actions_M_balanced10.pickle', 'wb') as file:
        pickle.dump(np.float32(actions), file)
    with open('data/episode_ends_M_balanced10.pickle', 'wb') as file:
        pickle.dump(np.cumsum(idx_ends), file)
    print('Preprocessing done... Data saved.')



if __name__ == "__main__":
    main() 
    
#balanced is cutting out all but one group of phase A mice
#balanced2 is with an additional 90C copied into the dataset
#balanced3 is without additional 90C mice, but with threat state computed separete for B and C
#balanced4 is like 3, but with additional state recording last dwell time 
#balanced5 is with threat state 1-3-10-30-90, we have to run this naive test (otherwise also reduced baseline)

#I will next try to see if the way switching is computed affects the distributions strongly.
#My intuition is that in some experimental groups the mouse is often on the border between cages leading to more noisy switching.
#I have to first look at the data alone, then if it makes a big difference create synthetic data using naive and behavioral threat encoding.
#Finally, check if the additional state is necessary or if we can go back to five states.
#balanced6 is with behavioral encoded threat state, different switching calculation.
    
    
#SOME CONCLUSIONS:
    #1. Balance is important
    #2. Additional state with previous dwell time helps
    #3. Naive threat encoding gives good performance at distributions but not over time.
    #4. Different threat encoding for phase B and C gives reasonable performance over time.
    #5. Changing the switching criterion does not help. 

#NEXT balanced7
#1. go back to simple switching criterion.
#2. Different normalization for time-related states

#Different normalization does not help performance..... Actually, it gives good results on the t-test and mannwhitneyu for phase C!

#balanced8 
#simple switching, 6 states but the same threat for B and C -> does not work for 90C   

#when we generate probability curves, it looks like the second state is not important!
#balanced 9: no second state
#somehow it looks like the previous state is also not important
#balanced 10: only four states

