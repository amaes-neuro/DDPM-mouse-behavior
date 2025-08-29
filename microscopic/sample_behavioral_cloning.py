# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 11:26:54 2025

Samples through the center from the behavioral cloning model.

@author: amaes
"""

from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
import torch
from mouse_2D import Mouse2DEnv
from tqdm.auto import tqdm
import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shapely
from matplotlib.path import Path
import shapely.ops


# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


"""
Load dataset
"""

#model
model = 't_807_0'#sys.argv[1]
dataset_name = 'balanced7_x1'#sys.argv[2]

path = 'checkpoints/'+model+'.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')
if not os.path.exists('data_model_curves/'+model+'/samples'):
    os.makedirs('data_model_curves/'+model+'/samples')

#parameters
pred_horizon = 8
obs_horizon = 1
action_horizon = 1
obs_dim = 8
action_dim = 2


# create dataset from file
dataset = MouseTrajectoryDataset(
    dataset_path='data/',
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    name=dataset_name
)
# save training data statistics (min, max) for each dim
stats = dataset.stats


# create network object
class FF_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(FF_net, self).__init__()
        # hidden layer 
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_neurons),
            torch.nn.BatchNorm1d(hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, hidden_neurons),
            torch.nn.BatchNorm1d(hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, hidden_neurons),
            torch.nn.BatchNorm1d(hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, hidden_neurons),
            torch.nn.BatchNorm1d(hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_neurons, output_size),
            )
    # prediction function
    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred

model_net = FF_net(obs_dim,50,action_dim*pred_horizon)
model_ = model_net
model_.load_state_dict(state_dict)
# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
_ = model_.to(device)
model_.eval() 


walls = [(-1.169,14.47),(35.33,16.47),(33.915,3.4086),(50.7458,3.1977),(49.80,16.520),(85.61,13.88),
              (85.61,-21.118),( 49.36,-22.87),(50.07,-9.905),(34.75,-9.34),(35.04,-22.576),(-0.87,-19.694)] #this is static, the box does not change over time

closed_box = walls
closed_box.append( (-1.169,14.47) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )

box_path = Path( np.array(walls) )

def compute_sensory_field(location):
    nb_angles = 18
    sensory_field = np.zeros((nb_angles,))
    for j in range(nb_angles):
        line = shapely.LineString([ (location[0],location[1]),
                                   (location[0]+1e3*np.cos(j*2*np.pi/nb_angles),location[1]+1e3*np.sin(j*2*np.pi/nb_angles)) ])
        inters = shapely.intersection(line, closed_box)
        if inters.geom_type == 'MultiPoint': #multiple intersections
            temps = np.zeros((len(inters.geoms),))    
            for i in range(len(inters.geoms)):
                temps[i] = np.sqrt( (inters.geoms[i].x-location[0])**2 + (inters.geoms[i].y-location[1])**2 )
            sensory_field[j] = np.min(temps)
        elif inters.geom_type == 'LineString': #this happens when the location is outside the walls
            sensory_field[j] = 0
        else: #a single intersection
            sensory_field[j] = np.sqrt( (inters.x-location[0])**2 + (inters.y-location[1])**2 )
    return sensory_field


env = Mouse2DEnv(render_mode='rgb_array')
#set up loop
nb_samples = 500
for sample in range(nb_samples):
    print('Generating sample '+str(sample))
    
    threat_values = [0]
    time_steps = 90
    time_shift = 0
    curves = np.zeros((2*len(threat_values),time_steps,2))
    for i in range(2*len(threat_values)):    
        
        threat = threat_values[i%len(threat_values)]
        location = np.array([40,-2])
        food = 0
        if i<len(threat_values):
            loc_avg = np.array([18,-1]) #coming from left side #40: (16,0) 30: (18,-1)  20: (18,-1) 10: (30,-1)
        else:
            loc_avg = np.array([54,-1]) #coming from right side #40: (64,-2) 30: (57,-3) 20: (54,-1) 10: (52,-2)

        trajectory = []
        obs, info = env.reset(location = location, loc_avg=loc_avg,  time = time_shift, food = food, threat = threat)
        state = obs
        for j in range(time_steps):
            #normalize
            nobs = normalize_data(state.reshape((1,obs_dim)), stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            with torch.no_grad():
                act = model_(nobs)
            action = act.detach().to('cpu').numpy()
            new_loc = state[0:2] + unnormalize_data(action[0][0:2], stats=stats['action'])
            
            if box_path.contains_point( new_loc ):
                state[0:2] = new_loc
            else:     
                projection = shapely.ops.nearest_points(closed_box,shapely.Point((new_loc[0],new_loc[1])))
                state[0:2] = np.array([projection[0].x,projection[0].y])
    
            
            agent_sensory_field = compute_sensory_field(state[0:2])
            dist_idx = np.sort(agent_sensory_field)
            state[2] = np.mean(dist_idx[0:6])
            state[3:5] = (1-1/20)*state[3:5] + state[0:2]/20
            state[5] += 1        
            
            trajectory.append(state)
        #save trajectory
        curves[i,:,:] = np.vstack(trajectory)[:,0:2]
        
    #save curve    
    data_dict = {'state': np.array([location[0], location[1]]), 'actions': curves}    
    print('Save sampled actions in state:',data_dict['state'])
    with open('data_model_curves/'+model+'/samples/curve_'+str(sample)+'.pickle', 'wb') as file:
        pickle.dump(data_dict, file)

