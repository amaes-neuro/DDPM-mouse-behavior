# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:52:40 2025

Training script that also tracks test loss.

This code is adapted from https://github.com/real-stanford/diffusion_policy/tree/main

@author: amaes
"""

from MouseTrajectoryDataset import MouseTrajectoryDataset
from ConditionalDiffuser import ConditionalUnet1D
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import pickle
import sys
import os

"""
Load dataset
"""
#model
model = 'test' #sys.argv[1]
dataset_name = 'balanced7_x1' #sys.argv[2]

# set path
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')


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
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

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

# device transfer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
_ = noise_pred_net.to(device)



"""
Train model
"""

num_epochs = 1000

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    use_ema_warmup=True, #try this out
    power=0.8)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs
)

losses = []
losses_test = []
with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch[0].to(device)
                B = nobs.shape[0]
                naction = nbatch[1].reshape((B,pred_horizon,action_dim)).to(device)

                # (B, obs_horizon * obs_dim)
                obs_cond = nobs

                # I want to see if adding noise to obs_cond can help generalization
                obs_cond[:,-1] = obs_cond[:,-1] + torch.randn(obs_cond[:,-1].shape, device=device)/40

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        losses.append(np.mean(epoch_loss))
        if epoch_idx%20 == 0: #save checkpoints every 20 epochs
            ema_noise_pred_net = noise_pred_net
            path = 'checkpoints/'+model+'.pt'
            torch.save(ema_noise_pred_net.state_dict(), path)
        
        with torch.no_grad():
            epoch_loss_test = list()
            for batch, (X, y) in enumerate(test_dataloader):
                nobs = X.to(device)
                B = nobs.shape[0]
                naction = y.reshape((B,pred_horizon,action_dim)).to(device)
                
                # (B, obs_horizon * obs_dim)
                obs_cond = nobs

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss_cpu = loss.item()
                epoch_loss_test.append(loss_cpu)
            losses_test.append(np.mean(epoch_loss_test))
            print('TEST LOSS: '+str(np.mean(epoch_loss_test)))

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

#save weights
path = 'checkpoints/'+model+'.pt'
torch.save(ema_noise_pred_net.state_dict(), path)
print('Model saved to'+path)

#save loss
with open('checkpoints/'+model+'_loss_train.pickle','wb') as file:
	pickle.dump(losses,file)
with open('checkpoints/'+model+'_loss_test.pickle','wb') as file:
	pickle.dump(losses_test,file)

