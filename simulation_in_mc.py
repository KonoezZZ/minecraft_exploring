# Need to download all your pre-trained models, and run locally through anaconda

from . import s2s
import numpy as np
import torch
import minerl
import gym
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


##### S2A case #####

def encode_action(a):
    x = OrderedDict()
    x['attack'] = np.array(0)
    x['camera'] = np.array([0, 0], dtype='float32')
    x['place'] = 'none'
    x['sneak'] = np.array(0)
    x['sprint'] = np.array(0)
    if a in [0,1,2,3,4,5]:
        x['forward'] = np.array(0)
        x['back'] = np.array(0)
    elif a in [6,7,8,9,10,11]:
        x['forward'] = np.array(1)
        x['back'] = np.array(0)
    else:
        x['forward'] = np.array(0)
        x['back'] = np.array(1)
    if a in [0,1,6,7,12,13]:
        x['right'] = np.array(0)
        x['left'] = np.array(0)
    elif a in [2,3,8,9,14,15]:
        x['right'] = np.array(1)
        x['left'] = np.array(0)
    else:
        x['right'] = np.array(0)
        x['left'] = np.array(1)
    if a in [0,2,4,6,8,10,12,14,16]:
        x['jump'] = np.array(0)
    else:
        x['jump'] = np.array(1)
    return x

def get_action(model, obs):
    state = torch.from_numpy(obs[0]['pov']).float()
    inputs = torch.stack(tuple([state for _ in range(8)]), dim=0)
    inputs = inputs.permute(0,3,1,2)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        bias = [0.35, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8, 0.6, 0.5, 0.8, 1, 10, 10, 10, 10, 10]
        #bias = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        scores = F.softmax(outputs, dim=-1).data * torch.tensor(bias)
        __, prediction = torch.max(scores, 1)
        prediction = prediction[0]
    return prediction

def run(n_steps=100, interval=5):
    rec = []
    obs = env.step(encode_action(0))
    for _ in range(n_steps):
        action = get_action(model, obs)
        while action == 0:
            action = get_action(model, obs)
        if random.random() <= 0.1:
            action = random.randint(0,17)
        rec.append(int(action))
        action = encode_action(action)
        for __ in range(interval):
            obs = env.step(action)
    return rec
    
env = gym.make("MineRLNavigateExtreme-v0")
env.reset()
model = build_network_s2a()
model.load_state_dict(torch.load('YOUR_MODEL.pt'))
run()




##### S2S + SA2S case #####

def load_sa2s(action_class):
    model = UNet_sa2s(3,
        depth=3,
        start_filts=8,
        merge_mode='concat')
    model.load_state_dict(torch.load('YOUR_SA2S_MODEL'+str(action_class)+'.pt'))
    return model

def load_all_sa2s():
    models = []
    for i in range(len(available_actions)):
      models.append(load_sa2s(i))
    return models

def load_s2s():
    model = UNet_s2s(3,
        depth=4,
        start_filts=8,
        merge_mode='concat')
    model.load_state_dict(torch.load('YOUR_S2S_MODEL.pt'))
    return model

def get_action2(s2s, sa2s, obs, multipliers):
    state = torch.from_numpy(obs[0]['pov']).float()
    X = torch.stack(tuple([state for _ in range(8)]), dim=0)
    with torch.set_grad_enabled(False):
        X = X.permute(0,3,1,2)
        y = s2s(X)
        best_a = 0
        losses = np.ones([len(y), len(sa2s)]) * 9999
        for a, model in enumerate(sa2s):
            with torch.set_grad_enabled(False):
                pred_y = model(X)
                for i in range(len(y)):
                    losses[i, a] = F.mse_loss(pred_y[i]*255, y[i]*255).item() * 1000
        losses = np.einsum('ij,j->ij', losses, multipliers)[0,1:]
        best_action = random.choice(np.where(losses == losses.min())[0])+1
    return best_action

available_actions = [[0, 0, 0],   # no action
            [0, 0, 1],   # jump
            [0, 1, 0],   # right
            [0, 1, 1],   # right+jump
            [0, -1, 0],  # left
            [0, -1, 1],  # left+jump
            [1, 0, 0],   # forward
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, -1, 0],
            [1, -1, 1],
            [-1, 0, 0],  # backward
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, 1, 1],
            [-1, -1, 0],
            [-1, -1, 1]]

def run2(n_steps=100,
         interval=20,
         multipliers=np.array([1, 1, 1, 1, 1, 1, 1.1, 1.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])):
    rec = []
    obs = env.step(encode_action(0))
    for _ in range(n_steps):
        action = get_action2(s2s, sa2s, obs, multipliers)
        rec.append(int(action))
        action = encode_action(action)
        for __ in range(interval):
            obs = env.step(action)
    return rec

env = gym.make("MineRLNavigateExtreme-v0")
env.reset()
s2s = load_s2s()
sa2s = load_all_sa2s()
run2()
