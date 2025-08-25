# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:35:01 2025

Regression of all state variables to predict the inferred TMT fluctuations.
The question is whether the inferred TMT is 'real'. If we manage to make a prediction it helps to argue that it is real.


@author: amaes
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_regression

#t_401_1 is initializing using a random number between 0 and 0.6, if I do that for 0.075 LR and 120 opt steps it seems 
#to remain on average around 0.3, so it is very sensitive to initialization
#t_401_2 takes additional opt steps (250 in total)

model = 't_401_2'
dataset_name = 'balanced4_x1'
dataset_path='data_dynamic_tmt/'+model+'/naive'

data_files = os.listdir('./'+dataset_path)
nb_points = len([name for name in data_files])

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
tmt_all_fluct = np.array([])
states_all = np.empty([0, 9])
episodes = np.zeros([len(episode_ends)])
for iteration in range(nb_points):
    file = open('data_dynamic_tmt/'+model+'/naive/dynamic_tmt_'+str(iteration)+'.pickle', 'rb')
    grid_id = iteration

    if grid_id == 0:
        states_ = states[0:episode_ends[grid_id],:] 
        actions_ = actions[0:episode_ends[grid_id],:] 
    else:        
        states_ = states[episode_ends[grid_id-1]:episode_ends[grid_id],:] 
        actions_ = actions[episode_ends[grid_id-1]:episode_ends[grid_id],:] 

    states_all = np.vstack((states_all,np.hstack((states_[0:-11:10,:-1],actions_[0:-11:10,:])))) #not taking tmt value as predictor
    episodes[grid_id] = states_[0:-11:10,:].shape[0]
    
    tmt_vals = pickle.load(file)
    file.close()
    tmt_vals = tmt_vals[0:-11:10]

    temp = pd.DataFrame(tmt_vals)  
    smooth = temp.ewm(alpha=1/5, adjust=False).mean()
    tmt_vals_smooth = np.reshape(smooth.to_numpy(),shape=(len(tmt_vals),))

    tmt_fluct = tmt_vals_smooth #- np.mean(tmt_vals_smooth) #this is tricky, if the tmt inferred is initialized randomly it will give underestimate high tmt and overestimate low tmt
    tmt_all = np.hstack((tmt_all,tmt_vals_smooth))
    tmt_all_fluct = np.hstack((tmt_all_fluct, 2*( (tmt_fluct-np.min(tmt_fluct))/( np.max(tmt_fluct)-np.min(tmt_fluct) ) -0.5) ))


episodes = np.cumsum(np.int64(episodes)) #careful, you need all mice to be done

"""
def func(X, *args):
    a, b, c, d, e, f, g, h, i = args
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = X
    return a*x1 + b*x2 + c*x3 + d*x4 + e*x5 + f*x6 + g*x7 + h*x8 + i*x9 

def funcvec(x, y):
    return np.dot(x,y)

xdata = tuple(map(tuple, states_all.T))
ydata = tmt_all_fluct
popt, pcov = curve_fit( func, xdata, ydata, p0=[0]*9)
"""

#linear regression clearly fails, i could build in some nonlinearities but in that case 
#it is easier to go straight into fitting a neural network with one hidden layer

#try to embed (state,action) pairs into 2d space using t-sne, is there a gradient for tmt fluctuations?
states_all = 2*( (states_all-np.min(states_all,axis=0))/(np.max(states_all,axis=0)-np.min(states_all,axis=0)) - 0.5)
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=30).fit_transform(states_all)


fig, ax = plt.subplots()
sc = ax.scatter(X_embedded[:,0], X_embedded[:,1], c=tmt_all_fluct, cmap=plt.cm.RdYlGn, alpha=.5);
plt.colorbar(sc)
ax.axis('off')



# Define the class for single layer NN
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
X = torch.from_numpy(states_all).type(torch.FloatTensor)
Y = torch.from_numpy(np.reshape(tmt_all_fluct,shape=(tmt_all_fluct.shape[0],1))).type(torch.FloatTensor)

train = np.empty([0,in_dim])
train_y = np.array([])
test = np.empty([0,in_dim])
test_y = np.array([])
burn_in = 0 #first minute maybe there is an effect of the exponential smoothing that is not yet accurate
for i in range(len(episodes)):
    if i in [1,7,16,25,34,43,50,59,68,77,86]:
        test = np.vstack((test,states_all[episodes[i-1]+burn_in:episodes[i],:]))
        test_y = np.hstack((test_y,tmt_all_fluct[episodes[i-1]+burn_in:episodes[i]]))
    else:
        if i==0:
            train = np.vstack((train,states_all[0+burn_in:episodes[i],:]))
            train_y = np.hstack((train_y,tmt_all_fluct[0+burn_in:episodes[i]]))
        else:
            train = np.vstack((train,states_all[episodes[i-1]+burn_in:episodes[i],:]))
            train_y = np.hstack((train_y,tmt_all_fluct[episodes[i-1]+burn_in:episodes[i]]))
 
test_y = np.reshape(test_y,shape=(test_y.shape[0],1))
train_y = np.reshape(train_y,shape=(train_y.shape[0],1))
train_dataset = TensorDataset(torch.from_numpy(train).type(torch.FloatTensor), torch.from_numpy(train_y).type(torch.FloatTensor))
test_dataset = TensorDataset(torch.from_numpy(test).type(torch.FloatTensor), torch.from_numpy(test_y).type(torch.FloatTensor))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


epochs= 150
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
for i in range(93):
    with torch.no_grad():
        if i == 0:
            pred = model(X[0:episodes[i],:]).numpy()
            y = Y[0:episodes[i]].numpy()
            corrs.append(np.corrcoef(X[0:episodes[i],0].numpy(),y[:,0])[0,1])
        else:
            pred = model(X[episodes[i-1]:episodes[i],:]).numpy()
            y = Y[episodes[i-1]:episodes[i]].numpy()
            corrs.append(np.corrcoef(X[episodes[i-1]:episodes[i],0].numpy(),y[:,0])[0,1])
    errs.append(np.sum((pred-y)**2))
    
    
plt.figure()    
plt.hist(errs,40)

plt.figure()
plt.scatter(errs[0:7],corrs[0:7])
plt.scatter(errs[7:50],corrs[7:50])
plt.scatter(errs[50:],corrs[50:])


print(mutual_info_regression(states_all[episodes[7]:episodes[50],:], tmt_all_fluct[episodes[7]:episodes[50]]))
print(mutual_info_regression(states_all[episodes[50]:,:], tmt_all_fluct[episodes[50]:]))
#if you shuffle inferred tmt the MI is more or less zero, i.e. independent variables
#food presence should give 0 MI, it is a constant if you split the dataset in two
#the ACTION is very uninformative as a feature but this is understandable because we are only predicting the fluctuations, not the amplitude/average!
#there is more MI for food present: so those fluctuations are more predictable, the presence of food reduces the entropy as seen in FIG4!
#two different networks to predict tmt flucts? one for food presence and for no food?

