import json
import os
import random
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
dtype = torch.float32


# SEMANTIC SEGMENTATION --> BIRD HIGH VIEW


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
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    x = x.unsqueeze(0)
    x = x.cuda(device=device)
    return x


def compute_returns(data, discount_factor=0.45):
    """
    Computes the discounted returns given the rewards.

    Parameters:
    - data (any type): data needed to compute the reward
    - discount_factor (float, optional): discount factor, default is 0.45

    Returns:
    - returns (Tensor): tensor of shape (183, 1) representing the discounted returns
    """
    rewards = ComputeRewards(dtype=dtype, device=device).get_reward(data)
    num_steps = rewards.shape[0]
    returns = torch.zeros(num_steps, dtype=rewards.dtype, device=rewards.device)
    returns[-1] = rewards[-1].reshape(1)
    for t in range(num_steps - 2, -1, -1):
        returns[t] = rewards[t].reshape(1) + discount_factor * returns[t + 1]

    return returns, rewards


# def compute_returns(data,
#                     discount_factor=0.45):  # discount factor value between 0 and 1, 0 --> prefer immediate rewards || 1 --> future rewards
#
#     rewards = ComputeRewards(dtype=dtype, device=device)
#     print((rewards.get_reward(data)).shape)
#     returns = torch.zeros(num_steps + 1, dtype=dtype, device=device)
#     for t in reversed(range(num_steps)):
#         returns = returns.unsqueeze(-1)
#         returns[t] = rewards.get_reward(data).reshape(returns[t + 1].shape) + discount_factor * returns[t + 1]
#
#     return returns


def compute_ppo_loss(action_log_probs, values, advantages, returns, old_action_log_probs, eps):  # eps hyperparameter
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - values).pow(2).mean()
    return policy_loss, value_loss


def save_model(epoch, model, name, optimizer, loss, loss_list, reward_list, return_list, total_step, good_episodes,
               bad_episodes_for_stuck, bad_episodes_for_car_damage,
               bad_episodes_for_big_penalties, tot_real_step):
    torch.save({
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'loss': loss,
        'losslist': loss_list,
        'rewardlist': reward_list,
        'returnlist': return_list,
        'total_step': total_step,
        'good_episodes': good_episodes,
        'bad_episodes_for_stuck': bad_episodes_for_stuck,
        'bad_episodes_for_car_damage': bad_episodes_for_car_damage,
        'bad_episodes_for_big_penalties': bad_episodes_for_big_penalties,
        'tot_real_step': tot_real_step
    }, 'checkpoint' + str(name) + '.pth')


def save_model_if_on_going(episode, policy_network, value_network, optimizer, loss, loss_list, reward_list, return_list,
                           total_step, good_episodes, bad_episodes_for_stuck, bad_episodes_for_car_damage,
                           bad_episodes_for_big_penalties, tot_real_step):
    if episode > 300 and episode % 300 == 0:  # Save model periodically

        save_model(episode, policy_network, '_policy_network', optimizer, loss, loss_list, reward_list, return_list,
                   total_step, good_episodes, bad_episodes_for_stuck, bad_episodes_for_car_damage,
                   bad_episodes_for_big_penalties, tot_real_step)
        save_model(episode, value_network, '_value_network', optimizer, loss, loss_list, reward_list, return_list,
                   total_step, good_episodes, bad_episodes_for_stuck, bad_episodes_for_car_damage,
                   bad_episodes_for_big_penalties, tot_real_step)


def episode_evaluation(episode, reward_list, episode_rewards, return_list, loss_list, good_ep, bad_episode_stuck,
                       bad_episode_car, bad_episode_penalties, step_per_episode, tot_real_step):
    # print("this is the sum of reward list: " + str(sum(reward_list)) + "list is : " + str(
    #    reward_list) + "episode is: " + str(episode) + " steps of the episode is: " + str(step_per_episode))
    avg_rewards = sum(reward_list) / tot_real_step
    # print("avg_rew " + str(avg_rewards))
    avg_eps_reward = sum(episode_rewards) / step_per_episode
    avg_loss = sum(loss_list) / step_per_episode
    avg_return = sum(return_list) / step_per_episode
    tot_bad = bad_episode_car + bad_episode_stuck + bad_episode_penalties
    with open(os.path.join(os.path.dirname(__file__), "evaluation_data.txt"), 'a') as f:
        # print("salvo valutazioni")
        f.write(f'Episode {episode}: --> Total reward: {sum(episode_rewards)} || AVG Reward tot {avg_rewards} || AVG '
                f'Reward for the episode {avg_eps_reward} ||  AVG Loss {avg_loss} || AVG Return {avg_return}\n')
        f.write(f'Reward list for that episode: {str(episode_rewards)}\n')
        f.write(f'Total Episodes till now --> Good: ' + str(good_ep) + " Total Bad: " + str(tot_bad) + " of which --> "
                                                                                                       " for car "
                                                                                                       "stuck: " +
                str(bad_episode_stuck) + " for car health: " + str(bad_episode_car) + " for big penalties: " +
                str(bad_episode_penalties))
        f.write(f'\n')

        f.close()


def train():
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()
    # Add weight decay parameter to optimizer
    optimizer = torch.optim.Adam(list(policy_network.parameters()) + list(value_network.parameters()), lr=0.001,
                                 weight_decay=1e-5)
    # normal optimizer
    # optimizer = torch.optim.Adam(policy_network.parameters())
    loss = 0.0

    if os.path.exists("checkpoint_policy_network.pth") and os.path.exists("checkpoint_value_network.pth"):
        print("Existing models loading .....")
        policy_checkpoint = torch.load("checkpoint_policy_network.pth")
        policy_network.load_state_dict((policy_checkpoint['model']))
        value_checkpoint = torch.load("checkpoint_value_network.pth")
        value_network.load_state_dict((value_checkpoint['model']))
        optimizer.load_state_dict(policy_checkpoint['optimizer'])
        # optimizer.load_state_dict(value_checkpoint['optimizer'])

        # Move model parameters to CPU
        for param in policy_network.parameters():
            param.to(device)
        for param in value_network.parameters():
            param.to(device)

        # Move optimizer state to CPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        epoch = policy_checkpoint['epoch']
        loss = policy_checkpoint['loss']
        reward_list = policy_checkpoint['rewardlist']
        return_list = policy_checkpoint['returnlist']
        loss_list = policy_checkpoint['losslist']
        total_step = policy_checkpoint['total_step']
        good_episodes = policy_checkpoint['good_episodes']
        bad_episodes_for_stuck = policy_checkpoint['bad_episodes_for_stuck']
        bad_episodes_for_car_damage = policy_checkpoint['bad_episodes_for_car_damage']
        bad_episodes_for_big_penalties = policy_checkpoint['bad_episodes_for_big_penalties']
        tot_real_step = policy_checkpoint['tot_real_step']

        print("epoch:", epoch)
        print("loss:", loss)
        print("reward_list:", reward_list)
        print("return_list:", return_list)
        print("loss_list:", loss_list)
        print("total_step:", total_step)
        print("good_episodes:", good_episodes)
        print("bad_episodes_for_stuck:", bad_episodes_for_stuck)
        print("bad_episodes_for_car_damage:", bad_episodes_for_car_damage)
        print("bad_episodes_for_big_penalties:", bad_episodes_for_big_penalties)
        print("tot_real_step:", tot_real_step)


        print("Models loaded...")
        print("model restarted at episode --> " + str(epoch))
    else:
        print("no checkpoint found, starting from scratch")
        epoch = 0
        reward_list = []
        return_list = []
        loss_list = []
        total_step = 0
        good_episodes = 0
        bad_episodes_for_stuck = 0
        bad_episodes_for_car_damage = 0
        bad_episodes_for_big_penalties = 0  # sometimes the car crash, but it bounce back in the track and so it does not
        tot_real_step = 0

    time.sleep(3)  # give time to put the cursor back to the game

    eps = 0.1

    # stop, but we need to penalize that actions
    first_time = True
    for episode in range(epoch, num_episodes):
        to_stop = False
        big_penalties_counter = 0
        back_to_start()
        time.sleep(1.5)  # give time to put the cursor back to the game
        stuck_counter = 0
        episode_rewards = []

        x = random.randint(0, 9)
        if x == 0:  # random action again, 10%
            first_time = True
        # if episode > 600 and episode % 600 == 0:
        #     back_to_start()
        #     slow_down_training(sleep_time=30)
        step_per_episode = 0
        for step in range(num_steps):
            step_per_episode += 1
            tot_possible_step = episode * num_steps

            total_step += 1
            if not first_time:
                print("episode number: " + str(episode + 1) + " with step number: " + str(step))
                data, info = get_input_data(dtype, device)
                x = get_screen(screen_pos)
                x = preprocess_screen(x)
                x = x.to(device)
                policy_network.to(device)
                value_network.to(device)
                actions, action_log_probs = policy_network.sample(x)
                vehicle_control_movement(actions[0].cpu().numpy(), duration=0.2)
                time.sleep(0.1)
                old_action_log_probs = action_log_probs.detach()
                values = value_network(x)
                returns, rewards = compute_returns(data, info)
                advantages = returns - values
                if data.car.Speed < 0.5:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if stuck_counter > stuck_threshold:
                    bad_episodes_for_stuck += 1
                    to_stop = True
                # print(" is on road? " + str(data.onRoad))
                # if data.car.Health < health_threshold:
                #     bad_episodes_for_car_damage += 1
                #     break
                if rewards <= - 27.00 and big_penalties_counter < 3:  # I noticed the datas in the episode in which the reward reach a certain point,
                    big_penalties_counter += 1
                    # start to become useless or even break a few moment after, I think the threshold might be around
                    # 23.5-23.6, at a reward of -23.4 or below, it starts to become random, so it might be still a
                    # valid episode, not necessarily good, but useful
                    # -23.39 might be the minimum threshold from the data i collected
                else:
                    big_penalties_counter = 0  # reset

                if big_penalties_counter >= 3:
                    # print("reward: " + str(rewards.item()) + " e' >= di -23.39")
                    bad_episodes_for_big_penalties += 1
                    print("Too bad episode, might cause by a crash or a bad start, restart")
                    to_stop = True

                if rewards >= 10:
                    to_stop = True
                    good_episodes += 1
                    print("Finally a good episode, with the reward: " + str(rewards.item()) + " this is the " + str(
                        good_episodes) + " good episode.")
            else:
                first_time = False
                actions = random_action()
                vehicle_control_movement(actions[0], duration=0.2)
                time.sleep(0.1)
                x = get_screen(screen_pos)
                x = preprocess_screen(x)
                x = x.to(device)
                policy_network.to(device)
                value_network.to(device)
                data, info = get_input_data(dtype, device)
                actions, action_log_probs = policy_network.sample(x)
                old_action_log_probs = action_log_probs.detach()
                values = value_network(x)
                returns, rewards = compute_returns(data, info)
                advantages = returns - values

            policy_loss, value_loss = compute_ppo_loss(action_log_probs, values, advantages, returns, old_action_log_probs, eps)

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Inner Episode Evaluation part, evaluation part after the end of episode
            episode_rewards.append(rewards)
            return_list.append(returns)
            loss_list.append(loss)
            # mechanism for adjusting the learning rate of the optimizer over time,
            # which can help the agent converge faster and avoid getting stuck in local optima.
            if episode > 300 and episode % 300 == 0:  # Adjust learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.99
                print('aggiusto learning rate')

            # slow_down_training(sleep_time=0.2)

            eps *= 0.99  # Decreasing epsilon over time
            # slow_down_training(sleep_time=0.2)

            if to_stop:
                break

        reward_list.extend(episode_rewards)
        tot_real_step += step_per_episode

        save_model_if_on_going(episode, policy_network.state_dict(), value_network.state_dict(), optimizer.state_dict(),
                               loss, loss_list, reward_list, return_list, total_step, good_episodes,
                               bad_episodes_for_stuck, bad_episodes_for_car_damage,
                               bad_episodes_for_big_penalties, tot_real_step)

        # Evaluation Outer Episode part, real evaluation
        episode_evaluation(episode + 1, [rl.item() for rl in reward_list], [t.item() for t in episode_rewards],
                           [r.item() for r in return_list], [ls.item() for ls in loss_list], good_episodes,
                           bad_episodes_for_stuck, bad_episodes_for_car_damage, bad_episodes_for_big_penalties,
                           step_per_episode, tot_real_step)
        return_list.clear()
        loss_list.clear()

    save_model(num_episodes, policy_network.state_dict(), '_policy_network', optimizer.state_dict(), loss,
               loss_list, reward_list, return_list, total_step, good_episodes, bad_episodes_for_stuck,
               bad_episodes_for_car_damage,
               bad_episodes_for_big_penalties, tot_real_step)
    save_model(num_episodes, value_network.state_dict(), '_value_network', optimizer.state_dict(), loss,
               loss_list, reward_list, return_list, total_step, good_episodes, bad_episodes_for_stuck,
               bad_episodes_for_car_damage,
               bad_episodes_for_big_penalties, tot_real_step)


train()
