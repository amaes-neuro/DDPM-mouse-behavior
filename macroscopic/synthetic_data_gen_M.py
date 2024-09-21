# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:34:13 2024

Create synthetic data from macroscopic model. 
Run it 129 times to generate the same amount of synthetic data and save all resulting trajectories.

@author: ahm8208
"""

import pickle
import torch
from ConditionalDiffuser import ConditionalUnet1D
from mouse_M import MouseMEnv
import collections
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset_M import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import os
import sys

#load trained model
model = sys.argv[1]
dataset_name = sys.argv[2]
path = 'checkpoints/'+model+'.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')
if not os.path.exists('data_synthetic_M/'+model):
    os.makedirs('data_synthetic_M/'+model)
    
#parameters
pred_horizon = 4
obs_horizon = 1
action_horizon = 1
obs_dim = 5
action_dim = 1

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

env = MouseMEnv(render_mode='rgb_array')
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(10000)

#load data to get threat state values
dataset_path='data/'
file = open(dataset_path+'states_M_'+dataset_name+'.pickle', 'rb')
states = pickle.load(file)
file.close()
file = open(dataset_path+'episode_ends_M_'+dataset_name+'.pickle', 'rb')
episode_ends = pickle.load(file) # Marks one-past the last index for each episode
file.close()

for j in range(0,93):
    
    if j<7:
        food = 0
        threat = 0
    elif j>=7 and j<50:
        food = 0
        threat = states[episode_ends[j]-1,-1] 
    elif j>= 50:
        food = 1
        threat = states[episode_ends[j]-1,-1] 
    
    if j == 0:
        max_steps = episode_ends[0]
        location = states[0,0]
        time_vector = states[0,1:4]
    else:
        max_steps = episode_ends[j]-episode_ends[j-1]
        location = states[episode_ends[j-1],0]
        time_vector = states[episode_ends[j]-1,1:4]
        
    # get first observation, conditioned on threat level (TMT concentration)
    obs, info = env.reset(location = location, time_vec=time_vector, food = food, threat = threat)
    
    # keep a queue of last steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render()]
    trajectory = [obs]
    rewards = list()
    done = False
    step_idx = 0
    
    with tqdm(total=max_steps, desc="Eval MouseMEnv") as pbar:
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
    
            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                trajectory.append(obs)
                rewards.append(reward)
                imgs.append(env.render())
    
                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                    env.close()
                if done:
                    break
                
    #save trajectory
    with open('data_synthetic_M/'+model+'/states_synthetic_M_'+str(j)+'.pickle', 'wb') as file:
        pickle.dump(np.vstack(trajectory), file)

    
    