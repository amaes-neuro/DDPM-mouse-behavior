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
    sides = np.zeros((len(location),1))
    for i in range(len(location)):
        
        if i>0 and location[i]!=location[i-1]:
            sides[i] = 0
        elif i>0:
            sides[i] = sides[i-1] + 1

    return sides

      

def compute_threat(mouse_id, concentration, path, phase):
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

    points_A = points_A[mouse_id]
    points = points[mouse_id]
    ratio = np.sum(points[:,0]<40)/len(points)
    ratio_A = np.sum(points_A[:,0]<40)/len(points_A)
    return ratio_A-ratio #does it make more sense to divide those quantities?


    
def generate_space(concentration, phase, subsample):
    states = []
    actions = []
    idx_e = []
    
    #load data files
    path = 'C:\\Users\\ahm8208\\OneDrive - Northwestern University\\Documents\\behavior_diffusion_project\\data_rotated_shifted'
    file = open(path+'/points_nose_'+phase+'_'+concentration+'.pickle', 'rb')
    points = pickle.load(file)
    file.close()
    
    for i in tqdm(range(len(points))):
        agent_location = 1*(points[i][0::subsample,0] < 40)
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
            agent_threat = compute_threat(i, concentration, path, phase)*np.ones((len(agent_location),1))

        states_ = np.hstack((np.reshape(agent_location,(np.size(agent_location),1)),agent_sides,total_time,food_present,agent_threat))
        
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
    with open('data/states_M_balanced3.pickle', 'wb') as file:
        pickle.dump(np.float32(states), file)
    with open('data/actions_M_balanced3.pickle', 'wb') as file:
        pickle.dump(np.float32(actions), file)
    with open('data/episode_ends_M_balanced3.pickle', 'wb') as file:
        pickle.dump(np.cumsum(idx_ends), file)
    print('Preprocessing done... Data saved.')



if __name__ == "__main__":
    main() 
    
#balanced is cutting out all but one group of phase A mice
#balanced2 is with an additional 90C copied into the dataset
#balanced3 is without additional 90C mice, but with threat state computed separete for B and C



    
    
    
    