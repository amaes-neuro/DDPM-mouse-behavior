# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:50:04 2025

This script will aim to invert the trained model.
For a given trained model, we freeze the parameters and take the gradient towards the TMT input.
We will try to get a dynamic and optimal TMT value for a series of state,action pairs.

Sampling one 15 minute trajectory from a model takes about 15 minutes on our gpu. At every step we find the
optimal TMT value using gradient descent, this takes T steps. This means that getting a dynamic TMT value will for
each individual mouse takes T/4 hours. We can do much of this in parallel however because we do not need to forward simulate.

Downside: outputs are sampled, this means we have variation. The true optimal value might not be reached.
Other approach: because we have only one dimension, it might be okay to discretize possible TMT values and compute the optimal one.
This is much more computationaly expensive however.

STEPS:
    1. Import mouse data
    2. Import trained model
    3. Define optimizer
    4. Define model and loss
    5. Outer loop runs through time steps in data
    6. Inner loop optimizes TMT value at each time step
    7. Save dynamics

@author: amaes
"""

import pickle
import torch
from ConditionalDiffuser import ConditionalUnet1D
import collections
from mouse_2D import Mouse2DEnv
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
import sys
import matplotlib.pyplot as plt

#load trained model
model = 't_400_0'#sys.argv[1]
dataset_name = 'balanced4_x'#sys.argv[2]
iteration = 56#sys.argv[3]

path = 'checkpoints/'+model+'.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')
if not os.path.exists('data_dynamic_tmt/'+model):
    os.makedirs('data_dynamic_tmt/'+model)
    
#parameters
pred_horizon = 8
obs_horizon = 1
action_horizon = 1
obs_dim = 8
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict)
print('Weights loaded')

# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
_ = noise_pred_net.to(device)

#get stats
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

#noise scheduler
num_diffusion_iters = 50
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)


#load data of all mice
dataset_path='data/'
file = open(dataset_path+'states_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'actions_'+dataset_name+'.pickle', 'rb')
actions = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

#select the individual mouse using system input to script
if iteration == 0:
    states = states[0:episode_ends[iteration],:] 
    actions = actions[0:episode_ends[iteration],:] 
else:
    states = states[episode_ends[iteration-1]:episode_ends[iteration],:] 
    actions = actions[episode_ends[iteration-1]:episode_ends[iteration],:] 



def get_input_optimizer(input_val):
    # this line to show that input is a parameter that requires a gradient
    optimizer = torch.optim.Adam([input_val],lr=0.1,weight_decay=1e-6)
    return optimizer


def run_tmt_optmizer(model, target_output, state, num_steps=40, M=5):
    """Run the optimizer."""
    #make sure the state and target output are tensors and normalized
    target_output = torch.from_numpy(target_output).to(device,dtype=torch.float32)
    state = torch.from_numpy(state).to(device, dtype=torch.float32)

    # We want to optimize part of the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    A = state[0][-1]
    A.requires_grad_(True)
    B = state[0][0:-1]
    B.requires_grad_(False)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(A)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print('Optimizing..')
    for i in range(num_steps):
        
        #do i need a closure def? -> yes if using LBFGS optimizer, otherwise no
        with torch.no_grad():
                A.clamp_(-1, 1)
                
        C = torch.reshape(A,(1,1))
        B = torch.reshape(B,(1,7))
        nobs = torch.cat((B, C), 1)

        # infer action
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (1, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = model(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        for k in range(M):
            # predict noise
            noise_pred = model(
                sample=naction,
                timestep=0,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=0,
                sample=naction
            ).prev_sample
        
        #network predicts multiple steps in future, mse on this whole trajectory
        naction = naction[0]
        loss = torch.nn.functional.mse_loss(naction, target_output)

        print('loss: '+str(loss.item()),'tmt: '+str(A.item()))

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()
        
    with torch.no_grad():
        A.clamp_(-1, 1)    
        
    return A.detach().to('cpu').numpy(), loss.item()


tmt_vals = np.zeros((states.shape[0],))
print('Encoded TMT value:'+str(states[0,7]))
#for i in range(200,states.shape[0],10):
nb_its, nb_tmts = 10, 50
losses = np.zeros((nb_its,nb_tmts))
for j in range(nb_its):
    for k in range(nb_tmts):
        i=280
        env = Mouse2DEnv(render_mode='rgb_array')
        obs, _ = env.reset(location = states[i,0:2], loc_avg = states[i,3:5], time=i, food = states[i,6], threat = k*0.7/nb_tmts)#np.random.rand()*0.6)#states[i,7])
    
        # keep a queue of last steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        
        # stack the last obs_horizon number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        # device transfer
    
        #get target action
        target = actions[i:i+pred_horizon,:]
        target = normalize_data(target, stats=stats['action'])
        
        #optimize loop
        result, loss = run_tmt_optmizer(ema_noise_pred_net,target,nobs,num_steps=1,M=0)
        temp = np.zeros((8,))
        temp[-1] = result
        tmt_vals[i] = unnormalize_data(temp,stats=stats['obs'])[-1]
        print('Time step '+str(i), 'TMT value:'+str(tmt_vals[i]))
        losses[j,k] = loss
        


fig, ax = plt.subplots()
means = np.mean(losses,axis=0)
stds = np.std(losses,axis=0)
ax.plot(np.linspace(0,0.7,nb_tmts), means)
ax.fill_between(np.linspace(0,0.7,nb_tmts), means-stds/np.sqrt(nb_its), means+stds/np.sqrt(nb_its) ,alpha=0.3)
plt.vlines(states[0,7], 0, 0.01, linestyle='--')
print(np.where(means==np.min(means))[0]*0.7/nb_tmts)

#with open('data_dynamic_tmt/'+model+'/dynamic_tmt_X1_'+str(iteration)+'.pickle', 'wb') as file:
#    pickle.dump(tmt_vals, file)


# QUESTION
# To what degree is this estimate of the optimal TMT correct?
# Do we need a lower learning rate, or more iteration steps?
# How does the inherent sample diversity affect our estimate?

# We can test whether it gives something meaningful by looking at TMT dynamics as a function of location,
# maybe approach/retreat modulates it strongly. So is there a structure? It should be in line with our 
# understanding of how tmt affects trajectories so far (distance to walls, food, left vs right movement, time in box!).

# IN THEORY we can extend this to also the food state, however this is a binary state, slightly weird

