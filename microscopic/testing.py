# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:50:03 2024

Test a trained model, it runs once, plots the trajectory and saves an animation.

OUT OF DATE --- CHECK 

@author: ahm8208
"""

import torch
from ConditionalDiffuser import ConditionalUnet1D
from mouse_2D import Mouse2DEnv
import collections
from tqdm.auto import tqdm
import numpy as np
from MouseTrajectoryDataset import normalize_data, unnormalize_data, MouseTrajectoryDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
#import cv2
import matplotlib.pyplot as plt
import os
import shapely

#load trained model
path = 'checkpoints/t_800_0.pt'
if torch.cuda.is_available():
    state_dict = torch.load(path, map_location='cuda')
else:
    state_dict = torch.load(path, map_location='cpu')

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

# limit enviornment interaction before termination
subsample = 10
max_steps = 27000/subsample #this is for fifteen minutes of simulates as the videos were recorded at 30Hz
env = Mouse2DEnv(render_mode='rgb_array')
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(10000)

# get first observation, conditioned on threat level (TMT concentration)
obs, info = env.reset(food = 1, threat=0.2)

# keep a queue of last steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render()]
trajectory = [obs]
rewards = list()
done = False
step_idx = 0

with tqdm(total=max_steps, desc="Eval Mouse2DEnv") as pbar:
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


#plot 2D trajectory
walls = [(-0.90,13.0),(35.53,14.88),(34.11,1.88),(51.00,1.68),(50.03,14.94),(85.83,12.31),
         (85.83,-22.69),(49.54,-24.40),(50.23,-11.48),(35.01,-10.94),(35.23,-24.07),(-0.66,-21.24)] 
closed_box = walls
closed_box.append( (-0.90,13.0) ) #make sure the box is closed
closed_box = shapely.LineString( closed_box )


plt.figure()
points = np.vstack(trajectory)
plt.plot(points[:,0],points[:,1])
plt.plot(*closed_box.xy,color='black')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.show()
#plt.savefig("figures/synthetic_t1_A_0.pdf", format="pdf", bbox_inches="tight")


#plot hunger
#plt.figure()
#plt.plot(points[:,2])

#animate result
"""
video = cv2.VideoWriter('animations/t1_0_0.avi', 0, 10, (110,60))
for image in imgs:
    video.write(image)
cv2.destroyAllWindows()
video.release()
"""






