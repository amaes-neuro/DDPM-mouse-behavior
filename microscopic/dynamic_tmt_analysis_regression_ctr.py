# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 11:36:08 2025

We train a feedforward network to predict TMT, as opposed to using the diffusion model.
We want to compare this to the inferred TMT as predicted by diffusion model, is this worse? Different prediction?

@author: amaes
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_selection import mutual_info_regression


dataset_name = 'balanced4_x1'

dataset_path_='data/'

file = open(dataset_path_+'states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()

file = open(dataset_path_+'actions_'+dataset_name+'.pickle', 'rb')
actions = pickle.load(file) # Marks one-past the last index for each episode
file.close()

file = open(dataset_path_+'episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

tmt_all = np.array([])
states_all = np.empty([0, 9])
episodes = np.zeros([len(episode_ends)])
for grid_id in range(len(episode_ends)):

    if grid_id == 0:
        states_ = states[0:episode_ends[grid_id],:] 
        actions_ = actions[0:episode_ends[grid_id],:] 
    else:        
        states_ = states[episode_ends[grid_id-1]:episode_ends[grid_id],:] 
        actions_ = actions[episode_ends[grid_id-1]:episode_ends[grid_id],:] 

    states_all = np.vstack((states_all,np.hstack((states_[0:-11:10,:-1],actions_[0:-11:10,:])))) #not taking tmt value as predictor
    episodes[grid_id] = states_[0:-11:10,:].shape[0]
    
    tmt_vals = states_[0:-11:10,-1]
    tmt_all = np.hstack((tmt_all,tmt_vals))


episodes = np.cumsum(np.int64(episodes)) #careful, you need all mice to be done
states_all = 2*( (states_all-np.min(states_all,axis=0))/(np.max(states_all,axis=0)-np.min(states_all,axis=0)) - 0.5)


# Define the model
class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.BatchNorm1d(hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_size),
            )
    # prediction function
    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred


batch_size = 128
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    cost = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        cost += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    
    return cost


def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += torch.nn.functional.mse_loss(pred, y).item()

    return test_loss

in_dim = states_all.shape[1]
model = one_layer_net(in_dim, 50, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
X = torch.from_numpy(states_all).type(torch.FloatTensor)
Y = torch.from_numpy(np.reshape(tmt_all,shape=(tmt_all.shape[0],1))).type(torch.FloatTensor)

train = np.empty([0,in_dim])
train_y = np.array([])
test = np.empty([0,in_dim])
test_y = np.array([])
burn_in = 0 #first minute maybe there is an effect of the exponential smoothing that is not yet accurate
for i in range(len(episodes)):
    if i in [1,7,16,25,34,43,50,59,68,77,86]:
        test = np.vstack((test,states_all[episodes[i-1]+burn_in:episodes[i],:]))
        test_y = np.hstack((test_y,tmt_all[episodes[i-1]+burn_in:episodes[i]]))
    else:
        if i==0:
            train = np.vstack((train,states_all[0+burn_in:episodes[i],:]))
            train_y = np.hstack((train_y,tmt_all[0+burn_in:episodes[i]]))
        else:
            train = np.vstack((train,states_all[episodes[i-1]+burn_in:episodes[i],:]))
            train_y = np.hstack((train_y,tmt_all[episodes[i-1]+burn_in:episodes[i]]))
 
test_y = np.reshape(test_y,shape=(test_y.shape[0],1))
train_y = np.reshape(train_y,shape=(train_y.shape[0],1))
train_dataset = TensorDataset(torch.from_numpy(train).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
test_dataset = TensorDataset(torch.from_numpy(test).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


epochs= 350
costs = []
test_cost = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    cost = train_loop(train_dataloader, model, optimizer)
    costs.append(cost)
    test_c = test_loop(test_dataloader,model)
    test_cost.append(test_c*len(train_dataloader)/len(test_dataloader)) 
    print(f"test loss: {test_cost[-1]:>7f} ")  
print("Done!")

plt.figure()    
plt.plot(costs[5:])
plt.plot(test_cost[5:])
    
model.eval()
with torch.no_grad():
    pred = model(X[episodes[39]:episodes[40],:]).numpy()
    y = Y[episodes[39]:episodes[40]].numpy()
        
plt.figure()
plt.plot(pred)
plt.plot(y)


with torch.no_grad():
    pred = model(X[episodes[79]:episodes[80],:]).numpy()
    y = Y[episodes[79]:episodes[80]].numpy()
        
plt.figure()
plt.plot(pred)
plt.plot(y)


with torch.no_grad():
    pred = model(X[episodes[11]:episodes[12],:]).numpy()
    y = Y[episodes[11]:episodes[12]].numpy()
        
plt.figure()
plt.plot(pred)
plt.plot(y)


errs = []
corrs = []
avgs = []
for i in range(93):
    with torch.no_grad():
        if i == 0:
            pred = model(X[0:episodes[i],:]).numpy()
            y = Y[0:episodes[i]].numpy()
            corrs.append(np.corrcoef(X[0:episodes[i],0].numpy(),pred[:,0])[0,1])
            avgs.append(np.mean(pred))
        else:
            pred = model(X[episodes[i-1]:episodes[i],:]).numpy()
            y = Y[episodes[i-1]:episodes[i]].numpy()
            corrs.append(np.corrcoef(X[episodes[i-1]:episodes[i],0].numpy(),pred[:,0])[0,1])
            avgs.append(np.mean(pred))
    errs.append(np.sum((pred-y)**2))
    
    
plt.figure()    
plt.hist(errs,40)

plt.figure()
plt.scatter(errs[0:7],corrs[0:7])
plt.scatter(errs[7:50],corrs[7:50])
plt.scatter(errs[50:],corrs[50:])

fig, ax = plt.subplots()
ax.scatter(avgs[50:],tmt_all[episodes[50:]-1],color='yellowgreen',label='Phase C')
ax.scatter(avgs[7:50],tmt_all[episodes[7:50]-1],color='olivedrab',label='Phase B')
ax.scatter(avgs[:7],tmt_all[episodes[:7]-1],color='olive',label='Phase A')
for i in [1,7,16,25,34,43,50,59,68,77,86]:
    ax.scatter(avgs[i],tmt_all[episodes[i]-1],marker='X',edgecolor='black',facecolors='none')
    
ax.plot(np.linspace(0,0.8,10),np.linspace(0,0.8,10),linestyle='--',color='black')
plt.ylabel('Correct TMT value')
plt.xlabel('Avg inferred TMT value')
ax.spines[['right', 'top']].set_visible(False)
plt.legend()

w = 0.8
x = [i for i in range(3)]
corrs = [corrs[:7],corrs[7:50],corrs[50:]]
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
    ax.scatter(x[i] + 0.9*np.random.random(np.array(corrs[i]).size) * w - w / 2, corrs[i], color='grey')
ax.spines[['right', 'top']].set_visible(False)

plt.ylabel('Correlation x-position and inferred TMT')



print(mutual_info_regression(states_all[episodes[7]:episodes[50],:], tmt_all[episodes[7]:episodes[50]]))
print(mutual_info_regression(states_all[episodes[50]:,:], tmt_all[episodes[50]:]))

#total model params
print( sum(p.numel() for p in model.parameters() if p.requires_grad) )








