# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:58:29 2024

Script to organize the data and save in pickle files. 
Organization files: 
    one long array of all states for all mice
    one long array of all actions for all mice
    array which indicates the last index of each episode 

The files are then read by the dataset class later on.

Actions are distance travelled in 2D in one time step.
States (by increasing temporal scale):
    1. Location at current time step (2D)
    2. Wall distances at current time step (18D, 0 or >0)
    3. Nb of timesteps towards food (0 or >0)
    4. Nb of timesteps away from food (0 or >0)
    5. Time in left side of cage (0 or >0)
    6. Time in right side of cage (0 or >0)
    7. Total time elapsed (>0)
    8. Food present (constant, 0 during phase A and B, 1 in phase C)
    9. Concentration (constant, 0 during phase A or >0 in phase B and C)

@author: ahm8208
"""

import pickle
import numpy as np
from matplotlib.path import Path
import shapely
import shapely.ops
from tqdm import tqdm
import pandas as pd


def compute_multiple_states(location, points, action):
    #this function computes the distances to walls for each location,
    #the direction state and the side of cage state (states 2-6 in description)
    nb_angles = 18
    walls = list(map(tuple,points))
    walls.append( (points[0,0],points[0,1]) ) #make sure the box is closed
    walls = shapely.LineString( walls )
    sensory_field = np.zeros((len(location),nb_angles))
    direction = np.zeros((len(location),2))
    sides = np.zeros((len(location),2))
    direction_bool = action[:,0]<0
    sides_bool = location[:,0]<40
    for i in range(len(location)):
        
        if i>0 and direction_bool[i]==1:
            direction[i,0] = direction[i-1,0] + 1
            direction[i,1] = 0
        elif i>0 and direction_bool[i]==0:
            direction[i,0] =  0
            direction[i,1] = direction[i-1,1] + 1

        if i>0 and sides_bool[i]==1:
            sides[i,0] = sides[i-1,0] + 1
            sides[i,1] = 0
        elif i>0 and sides_bool[i]==0:
            sides[i,0] =  0
            sides[i,1] = sides[i-1,1] + 1

        for j in range(nb_angles):
            line = shapely.LineString([ (location[i,0],location[i,1]),
                                       (location[i,0]+1e3*np.cos(j*2*np.pi/nb_angles),location[i,1]+1e3*np.sin(j*2*np.pi/nb_angles)) ])
            inters = shapely.intersection(line, walls)
            if inters.geom_type == 'MultiPoint': #multiple intersections
                temps = np.zeros((len(inters.geoms),))        
                for h in range(len(inters.geoms)):
                    temps[h] = np.sqrt( (inters.geoms[h].x-location[i,0])**2 + (inters.geoms[h].y-location[i,1])**2 )
                sensory_field[i,j] = np.min(temps) 
            elif inters.geom_type == 'LineString': #this happens when the location is outside the walls
                sensory_field[i,j] = 0
            else: #a single intersection
                sensory_field[i,j] = np.sqrt( (inters.x-location[i,0])**2 + (inters.y-location[i,1])**2 )
    return sensory_field, direction, sides



def clip_to_box(point, box):
    walls = list(map(tuple,box))
    walls.append( (box[0,0],box[0,1]) ) #make sure the box is closed
    walls = shapely.LineString( walls )
    box_path = Path( box )
    for i in range(len(point)):
        if box_path.contains_point( point[i] ) == False :
            projection = shapely.ops.nearest_points(walls,shapely.Point((point[i,0],point[i,1])))
            point[i] = np.array([projection[0].x,projection[0].y])
    return point
      

def compute_threat(mouse_id, concentration, path, walls, phase):
    file = open(path+'/points_nose_A'+'_'+concentration+'.pickle', 'rb')    
    points_A = pickle.load(file)
    file.close()
    #if phase == 'B':
    file = open(path+'/points_nose_B'+'_'+concentration+'.pickle', 'rb')    
    points_B = pickle.load(file)
    file.close()
    #elif phase == 'C':
    file = open(path+'/points_nose_C'+'_'+concentration+'.pickle', 'rb')   
    points_C = pickle.load(file)
    file.close()

    points_A = points_A[mouse_id][:,0]
    ratio_A = np.sum(get_location(points_A, walls))/len(points_A)
    
    points_B = points_B[mouse_id][:,0]
    ratio_B = np.sum(get_location(points_B, walls))/len(points_B)
    
    points_C = points_C[mouse_id][:,0]
    ratio_C = np.sum(get_location(points_C, walls))/len(points_C)

    return (2*ratio_A-ratio_B-ratio_C)/2 #does it make more sense to divide those quantities?


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
    path = 'C:\\Users\\amaes\\OneDrive - The Scripps Research Institute\\Documents\\Diffusion_project\\DDPM-mouse-behavior\\data_rotated_shifted'
    file = open(path+'\\points_nose_'+phase+'_'+concentration+'.pickle', 'rb')
    points = pickle.load(file)
    file.close()
    file = open(path+'\\walls_'+phase+'_'+concentration+'.pickle', 'rb')
    walls = pickle.load(file)
    file.close()
    
    for i in tqdm(range(len(points))):
        box = walls[i]
        agent_location = points[i][0::subsample]
        agent_location = clip_to_box(agent_location, box)
        tempx = pd.DataFrame(agent_location[:,0])
        tempy = pd.DataFrame(agent_location[:,1])
        agent_emax = tempx.ewm(alpha=1/10, adjust=False).mean()
        agent_emay = tempy.ewm(alpha=1/10, adjust=False).mean()
        agent_action = agent_location[1:]-agent_location[0:-1]
        agent_location = agent_location[0:-1]
        agent_sensory_field, agent_directions, agent_sides = compute_multiple_states(agent_location, box, agent_action)
        if phase=='C':
            food_present = np.ones((len(agent_location),1))
        else:
            food_present = np.zeros((len(agent_location),1))
        total_time = np.zeros((len(agent_location),1))
        total_time[:,0] = np.linspace(0,len(agent_location)-1,len(agent_location))
        if phase=='A':
            agent_threat = np.zeros((len(agent_location),1))
        else:
            agent_threat = compute_threat(i, concentration, path, box, phase)*np.ones((len(agent_location),1)) #int(concentration)*np.ones((len(agent_location),1))
        
        dist_idx = np.argpartition(agent_sensory_field,6,axis=-1)
        walls_distance = np.mean(np.take_along_axis(agent_sensory_field,dist_idx,axis=-1)[:,:6],1)
        walls_distance = np.reshape(walls_distance,(len(walls_distance),1))
        #states_ = np.hstack((agent_location,agent_sensory_field,agent_directions,
        #                     agent_sides,total_time,food_present,agent_threat))
        cum_sid = np.reshape(agent_sides[:,0]+agent_sides[:,1],(len(agent_sides),1))
        agent_avg = np.hstack((agent_emax.to_numpy(),agent_emay.to_numpy()))[0:-1]
        #states_ = np.hstack((agent_location,food_present,agent_threat))
        states_ = np.hstack((agent_location,walls_distance, agent_avg, total_time, food_present,agent_threat))

        
        states.append(states_)
        actions.append(agent_action)
        idx_e.append(len(agent_location))

    print('Training data processed for phase '+phase+' and concentration '+concentration+'...')
    return np.vstack(states), np.vstack(actions), np.vstack(idx_e)
   
    
  
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
                states_add,actions_add,idx_add = generate_space(concentrations[j], phases[i], 10)
                states_list.append(states_add)
                actions_list.append(actions_add)
                idx_ends.append(idx_add)
                
            
    #write to files  
    states = np.vstack(states_list)
    actions = np.vstack(actions_list)
    idx_ends = np.vstack(idx_ends)
    #remove heldout mice 
    """
    idx_heldout = np.array([1,7,16,25,34,43,50,59,68,77,86])
    temp = np.cumsum(idx_ends)
    for k in idx_heldout[-1::-1]:
        states = np.delete(states,np.arange(temp[k-1],temp[k]),axis=0)
        actions = np.delete(actions,np.arange(temp[k-1],temp[k]),axis=0)
    idx_ends = np.delete(idx_ends,idx_heldout)
    idx_ends = np.cumsum(idx_ends)
    states[idx_ends[5]:,7] += 0.05 - np.min(states[idx_ends[5]:,7]) #when holding out a baseline
    """
    idx_ends = np.cumsum(idx_ends)
    print(idx_ends)
    states[idx_ends[6]:,7] += 0.2 - np.min(states[idx_ends[6]:,7]) #for x only, to make tmt state nonnegative earlier i did 0.05
    with open('data/states_balanced7_x2.pickle', 'wb') as file:
        pickle.dump(np.float32(states), file)
    with open('data/actions_balanced7_x2.pickle', 'wb') as file:
        pickle.dump(np.float32(actions), file)
    with open('data/episode_ends_balanced7_x2.pickle', 'wb') as file:
        pickle.dump(idx_ends, file)
    print('Preprocessing done... Data saved.')


if __name__ == "__main__":
    main() 
    

#balanced indicates that we have tossed away most of the baseline in the training data
#we will try many types of data

#balanced0: 4 states (current location, food presence, TMT)
#balanced1: 5 states (current location, time, food presence, TMT)
#balanced2: 6 states (current location, wall distance, time, food presence, TMT)
#balanced3: 7 states (current location, wall distance, cum time in side, time, food presence, TMT)
#balanced4: 8 states (current location, wall distance, moving avg of location 30, time, food presence, TMT))
#balanced7: 8 states (same as balanced4 but with 10 time constant instead of 30)   
#balanced8: 8 states (same as balanced4 but with 20 time constant instead of 30)   
#balanced9: 8 states (same as 4 but with 5 time constant instead of 30)
    
#TODO:
#Run synthetic data for t_1_1 (V), t_1_2 (V), t_2_1 (V), t_2_2 (V), t_7 (V), t_7_1 (V), t_8 (V), t_8_1 (V)
#Train t_0 (V), t_0_1 (V) removing the time feature (4 dims), and run synthetic data t_0 (V), t_0_1 (V)
#Run paralell loop for central point for t_0 (V), t_0_1, t_1_1 (V), t_1_2, t_2_1 (V), t_2_2
#Determine best ma points for balanced 6, 7, 8 datasets, instead of TMT sweep use TMT values from dataset!
#Run paralell loop for central point for t_6 (V), t_6_1, t_7 (V), t_7_1, t_8 (V), t_8_1

#should i run the model predicting more future steps?

#t_7_2 is ran with 8 predicting steps instead of 4, on balanced4 (by accident balanced4 instead of balanced7)

"""
set of models:
    1. t_10_0 - t_15_0 (balanced0,balanced1,balanced2,balanced7,balanced8,balanced4) pred_hor = 8, action_hor = 1 - 200 epochs
    2. t_20_0 - t_25_0 (balanced0,balanced1,balanced2,balanced7,balanced8,balanced4) pred_hor = 12, action_hor = 1 - 200 epochs
    3. t_13_3, t_14_3, t_15_3 (balanced7,balanced8,balanced4) pred_hor = 8, action_hor = 3 - 200 epochs 
    4. t_13_6, t_14_6, t_15_6 (balanced7,balanced8,balanced4) pred_hor = 8, action_hor = 6 - 200 epochs
"""

#balanced4_s: 8 states like balanced 4, but TMT is not encoded it is just the actual concentration
#balanced4_b: 8 states like balanced 4, but TMT is encoded A-B
#balanced4_a: 8 states like balanced 4, but TMT is encoded (2A-B_C)/2
#balanced4_x: 8 states like balanced 4, but TMT is encoded (2A-B_C)/2 + shift to make it nonnegative
#in theory I can add a binary state indicating presence of TMT

#balanced7_x1 is the same as balanced4_x1 except more separation in TMT encoding

# x : 30
# x1 : 20
# x2 : 10
# x3 : 30, 7D
# x4 : 30, 6D
# x5 : 30, 4D
# x6 : 40, 8D

# x1_heldout : the 7th one of baseline heldout and then from phase B and C symmetric: 0,9,18,27,36 (+7 for phase B or +43 for phase C)
