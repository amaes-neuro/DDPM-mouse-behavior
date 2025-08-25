# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:25:13 2025

Imitation learning with LSTM - we use this as a baseline model.

@author: amaes
"""

from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
import torch
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
model = 't_500_5'#sys.argv[1]
dataset_name = 'balanced4_x'#sys.argv[2]

file = open('data/episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

# set path
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')


# parameters
pred_horizon = 8
obs_horizon = 1
action_horizon = 1


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

# create dataloader
obs_dim = 8
action_dim = 2

train = []
train_y = []
test = []
test_y = []
for i in tqdm(range(dataset.indices.shape[0])):
    begin, end = dataset.indices[i][0], dataset.indices[i][1]
    bools = episode_ends[np.array([1,7,16,25,34,43,50,59,68,77,86])-1] - begin
    bools2 = episode_ends[np.array([1,7,16,25,34,43,50,59,68,77,86])] - begin
    if -1 in (np.sign(bools)*np.sign(bools2)):
        test.append(dataset[i]['obs'])
        test_y.append(dataset[i]['action'].reshape((1,action_dim*pred_horizon)))
    else:
        train.append(dataset[i]['obs'])
        train_y.append(dataset[i]['action'].reshape((1,action_dim*pred_horizon)))

test = np.vstack(test)
train = np.vstack(train)
test_y = np.vstack(test_y)
train_y = np.vstack(train_y)



batch_size = 256
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# visualize data in batch
batch = next(iter(train_dataloader))
print(batch)
print('Training and test datasets prepared')


"""
Setup model
"""

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


# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model_net = FF_net(obs_dim,50,action_dim*pred_horizon)
_ = model_net.to(device)



"""
Train model
"""

num_epochs = 100
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    cost = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        cost.append(loss.item())
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    
    return np.mean(cost)


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    test_loss = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = torch.nn.functional.mse_loss(pred, y)
            test_loss.append( loss.item() )

    return np.mean(test_loss)


optimizer = torch.optim.AdamW(
    params=model_net.parameters(),
    lr=1e-4, weight_decay=1e-6)


costs = []
test_cost = []
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    cost = train_loop(train_dataloader, model_net, optimizer)
    costs.append(cost)
    test_c = test_loop(test_dataloader,model_net)
    test_cost.append(test_c) 
    print(f"test loss: {test_cost[-1]:>7f} ")  
print("Done!")

plt.figure()
plt.plot(costs[5:])
plt.plot(test_cost[5:])


#TODO  generate synthetic data (how bad will it be...)
path = 'checkpoints/'+model+'.pt'
torch.save(model_net.state_dict(), path)
print('Model saved to'+path)


"""
Test model
"""

file = open('data/states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()

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

if not os.path.exists('data_synthetic/'+model):
    os.makedirs('data_synthetic/'+model)

for i in range(len(episode_ends)):
    print('Generating synthetic data for mouse nb '+str(i))
    if i == 0:
        state = states[0,:]
        T = episode_ends[0]
    else:
        state = states[episode_ends[i-1],:]
        T = episode_ends[i] - episode_ends[i-1]
    trajectory = np.zeros((T,8))
    print(state)
    for j in range(T):
        trajectory[j,:] = state
        #normalize
        nobs = normalize_data(state.reshape((1,obs_dim)), stats=stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = model_net(nobs)
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
        state[3:5] = (1-1/30)*state[3:5] + state[0:2]/30
        state[5] += 1        
        
    with open('data_synthetic/'+model+'/states_synthetic_'+str(i)+'.pickle', 'wb') as file:
        pickle.dump(trajectory, file)


