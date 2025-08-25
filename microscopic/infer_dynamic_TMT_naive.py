# -*- coding: utf-8 -*-
"""
Created on Thu May 29 17:06:06 2025

A different way to infer the TMT. Not using gradient descent, but naively discretizing the TMT variable.
This is more costly potentially, and would not work when inferring more than one variable.
The reason I am doing this is that it seems that the TMT loss landscape can have multiple minima.

Training for more epochs significantly reduces the noise in the loss landscape. Does this mean there is overfitting?
Check the training and test loss curves. -> YES overfitting on held out mice
Holding out mice leads to a situation where a different minima is picked. Small problem.

I try to use all models to predict TMT now.

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
model = 't_701_'#sys.argv[1]
dataset_name = 'balanced7_x1'#sys.argv[2]
iteration = 0#int(sys.argv[3])

if not os.path.exists('data_dynamic_tmt/'+model):
    os.makedirs('data_dynamic_tmt/'+model)
    
#parameters
pred_horizon = 8
obs_horizon = 1
action_horizon = 1
obs_dim = 8
action_dim = 2

# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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




def run_tmt_optmizer(model, target_output, state, steps=1):
    """Run the optimizer."""
    
    obs_deque = collections.deque(
        [state] * obs_horizon, maxlen=obs_horizon)

    done = False
    step_idx = 0
    trajectory = []
    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_noise_pred_net(
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

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # stepping env
        obs, reward, done, _, info = env.step(action[0])
        # save observations
        obs_deque.append(obs)
        trajectory.append(obs)

        # update progress bar
        step_idx += 1
        if step_idx >= steps:
            done = True
            env.close()
        if done:
            break
        
    loss = np.mean( (np.vstack(trajectory)[:,0:2] - target )**2) 
    #print(loss, np.vstack(trajectory)[:,0:2], target)     
    return loss

tmt_vals = np.zeros((states.shape[0],))
nb_tmts = 200
num_steps = 8
losses_all = []
all_vals = []  
all_ls = []              
env = Mouse2DEnv(render_mode='rgb_array')
print('Encoded TMT value:'+str(states[0,7]))
for n in range(1):
    
    path = 'checkpoints/'+model+str(n+1)+'.pt'
    if torch.cuda.is_available():
        state_dict = torch.load(path, map_location='cuda')
    else:
        state_dict = torch.load(path, map_location='cpu')
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    _ = noise_pred_net.to(device)

    print('Weights loaded model '+str(n))
    losses = np.zeros((states.shape[0]-10,nb_tmts))
    for i in range(0,states.shape[0]-10,10):
        for k in range(nb_tmts):
            obs, _ = env.reset(location = states[i,0:2], loc_avg = states[i,3:5], time=i, food = states[i,6], threat = k*1/nb_tmts)#np.random.rand()*0.6)#states[i,7])
                
            #get target action
            target = states[i+1:i+1+num_steps,0:2]
            
            #optimize loop
            loss = run_tmt_optmizer(ema_noise_pred_net, target, obs, steps=num_steps)
            losses[i,k] = loss
            
        test = np.convolve(losses[i,:], np.ones(1)/1, mode='valid')
        all_vals.append(np.where(test==np.min(test))[0][0]*1/nb_tmts)
        all_ls.append(np.min(test))
        print('Time step '+str(i),'...', 'Estimated TMT value:'+f"{np.where(test==np.min(test))[0][0]*1/nb_tmts :.3f}",'...', 'Real TMT value:'+f"{states[0,7]:.3f}",'...', 'Average estimate:'+f"{np.average(all_vals,weights=1/np.array(all_ls)):.3f}")
    losses_all.append(losses)
    
losses = np.zeros((states.shape[0]-10,nb_tmts))
for n in range(6):
    losses += losses_all[n]/6
for i in range(0,states.shape[0]-10,10): 
    tmt_vals[i] = np.where(losses[i,:]==np.min(losses[i,:]))[0][0]*1/nb_tmts 

        

"""
fig, ax = plt.subplots()
means = np.mean(losses,axis=0)
stds = np.std(losses,axis=0)
ax.plot(np.linspace(0,0.7,nb_tmts), means)
ax.fill_between(np.linspace(0,0.7,nb_tmts), means-stds/np.sqrt(nb_its), means+stds/np.sqrt(nb_its) ,alpha=0.3)
plt.vlines(states[0,7], 0, 0.01, linestyle='--')
plt.xlabel('TMT')
plt.ylabel('Loss')
print(np.where(means==np.min(means))[0]*0.77/nb_tmts)
"""

with open('data_dynamic_tmt/'+model+'/dynamic_tmt_'+str(iteration)+'.pickle', 'wb') as file:
    pickle.dump(tmt_vals, file)


# QUESTION
# To what degree is this estimate of the optimal TMT correct?
# Do we need a lower learning rate, or more iteration steps?
# How does the inherent sample diversity affect our estimate?

# We can test whether it gives something meaningful by looking at TMT dynamics as a function of location,
# maybe approach/retreat modulates it strongly. So is there a structure? It should be in line with our 
# understanding of how tmt affects trajectories so far (distance to walls, food, left vs right movement, time in box!).

# IN THEORY we can extend this to also the food state, however this is a binary state, slightly weird

