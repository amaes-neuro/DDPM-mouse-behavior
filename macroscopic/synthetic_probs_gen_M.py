# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:40:28 2024

Sweep over state space to get the probability of taking an action.
To get an estimate of the probability, we sample from the distribution.

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

#load trained model
model = 't_M_24'
path = 'checkpoints/'+model+'.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')

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
    action_horizon=action_horizon
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

#set up loop
threat_values = [0,0.15,0.3,0.45]
samples = 100
time_steps = 80
time_shift = 100
side_prev = 10
env = MouseMEnv(render_mode='rgb_array')

for i in range(4*len(threat_values)):    
    
    threat = threat_values[i%len(threat_values)]
    if i>=2*len(threat_values):
        food = 1
    else:
        food = 0
    if i<len(threat_values):
        location = 0
    elif i>=2*len(threat_values) and i<3*len(threat_values):
        location = 0
    else:
        location = 1
    
    curves = np.zeros((samples,time_steps))
    for p in tqdm(range(time_steps)):

        for j in range(samples):
            #set up state
            obs, info = env.reset(location = location, side = np.random.randint(time_steps), side_prev=p, time = time_shift, food = food, threat = threat)
        
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            B = 1
            # stack the last obs_horizon number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1) 
            
            # infer action
            with torch.no_grad():
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
            
            #save action
            curves[j,p] = action[0]
            
    #save curve    
    data_dict = {'state': np.array([location, side_prev, time_shift, food, threat]), 'actions': curves}    
    print('Save sampled actions in state:',data_dict['state'])
    with open('data_model_curves_M/'+model+'/curve_M_'+str(i)+'.pickle', 'wb') as file:
        pickle.dump(data_dict, file)

#do not forget that the actions generated here are not binarized
#in the subsequent plotting I need to binarize the actions


