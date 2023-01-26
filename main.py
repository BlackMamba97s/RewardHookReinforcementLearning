import json
import time

import cv2
import requests
from PIL import Image

from Rewards.compute_rewards import ComputeRewards
from constants import num_episodes, num_steps, stuck_threshold, health_threshold
from models.PPO import PolicyNetwork, ValueNetwork
from models.my_trasformer import my_transformer
from util.data_functions import get_input_data
from util.screen import screen_pos, get_screen
import numpy as np
import torch

from util.utils import slow_down_training
from util.vehicle_action import back_to_start, vehicle_control_movement, random_action
import torchvision.transforms as transforms

device = torch.device("cuda:0")
print(device)
dtype = torch.float32


# def preprocess_screen(x):
#     x = np.array(x)
#     x = cv2.resize(x, (256, 256))
#     x = my_transformer(x)
#     x = x.unsqueeze(0)
#     x = x.cuda(device=device)
#     return x

def preprocess_screen(x):
    x = np.array(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (256, 256))
    x = transforms.ToTensor()(x)
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(x)
    x = x.unsqueeze(0)
    x = x.cuda(device=device)
    return x

def compute_returns(data,
                    discount_factor=0.45):  # discount factor value between 0 and 1, 0 --> prefer immediate rewards || 1 --> future rewards
    rewards = ComputeRewards(dtype=dtype, device=device)
    returns = torch.zeros(num_steps + 1, dtype=dtype, device=device)
    for t in reversed(range(num_steps)):
        print(returns.shape)

        returns[t] = torch.reshape(rewards.get_reward(data), returns[t].shape) + discount_factor * returns[t + 1]
    return returns


def compute_ppo_loss(action_log_probs, values, advantages, returns, old_action_log_probs, eps):  # eps hyperparameter
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - values).pow(2).mean()
    return policy_loss, value_loss


def save_model(model, optimizer):
    torch.save({
        'model': model,
        'optimizer': optimizer.state_dict(),
    }, 'checkpoint.pth')


def train():
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    optimizer = torch.optim.Adam(policy_network.parameters())
    eps = 0.1
    for episode in range(num_episodes):
        first_time = True
        print("episode number: " + str(episode))
        stuck_counter = 0
        for step in range(num_steps):
            if not first_time:
                print("step number: " + str(step))
                data, info = get_input_data(dtype, device)
                x = get_screen(screen_pos)
                print("questa e' x ", x)
                x = preprocess_screen(x)
                x = x.to(device)
                policy_network.to(device)
                value_network.to(device)
                actions, action_log_probs = policy_network.sample(x)
                vehicle_control_movement(actions[0], duration=0.2)
                time.sleep(0.1)
                old_action_log_probs = action_log_probs.detach()
                values = value_network(x)
                returns = compute_returns(data, info)
                advantages = returns - values
                if data.car.Speed < 0.1:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if stuck_counter > stuck_threshold:
                    back_to_start()
                    stuck_counter = 0
                    continue

                if data.car.Health < health_threshold:
                    back_to_start()
                    continue
            else:
                first_time = False
                actions = random_action()
                vehicle_control_movement(actions[0], duration=0.2)
                time.sleep(0.1)
                x = get_screen(screen_pos)
                print("questa e' x ", x)
                x = preprocess_screen(x)
                x = x.to(device)
                policy_network.to(device)
                value_network.to(device)
                data, info = get_input_data(dtype, device)
                time.sleep(0.1)
                actions, action_log_probs = policy_network.sample(x)
                old_action_log_probs = action_log_probs.detach()
                values = value_network(x)
                returns = compute_returns(data, info)
                advantages = returns - values

            policy_loss, value_loss = compute_ppo_loss(action_log_probs, values, advantages, returns,
                                                       old_action_log_probs, eps)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # mechanism for adjusting the learning rate of the optimizer over time,
            # which can help the agent converge faster and avoid getting stuck in local optima.
            if episode % 10 == 0:  # Adjust learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.99

            slow_down_training(sleep_time=0.1)

            if episode % 50 == 0:  # Save model periodically
                save_model(policy_network, optimizer)
                save_model(value_network, optimizer)

            eps *= 0.99  # Decreasing epsilon over time
            slow_down_training(sleep_time=0.1)

    save_model(policy_network,optimizer)
    save_model(value_network,optimizer)

train()
