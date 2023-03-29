# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="REINFORCE Algorithm")
parser.add_argument('--lr', type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--hs', type=int, default=128,
                    help="hidden size")
parser.add_argument('--gamma', type=float, default=0.98,
                    help="GAE gamma")
parser.add_argument('--lam', type=float, default=0.96,
                    help="GAE lambda")

args = parser.parse_args()

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter(f'./tb_record_3/lr_{args.lr}_hidden_{args.hs}_gamma_{args.gamma}_lambda_{args.lam}')
        
class Policy(nn.Module):
   
    def __init__(self):
        super(Policy, self).__init__()
        
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = args.hs
        self.double()
                
        self.share_layer = nn.Linear(self.observation_dim, self.hidden_size)
        
        self.value_layer = nn.Linear(self.hidden_size, 1)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)

        self.gae = GAE(args.gamma, args.lam, 300)

        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.dones = []

    def forward(self, state):
        tmp = F.relu(self.share_layer(state))

        state_value = self.value_layer(tmp)
        action_prob = F.softmax(self.action_layer(tmp), dim=-1)


        return action_prob, state_value


    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()


        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = []
        values = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / returns.std()

        for (prob, value) in saved_actions:
        	values.append(value)

        advantages = self.gae(self.rewards, values, self.dones)

        for (log_prob, state_value), A, R in zip(saved_actions, advantages, returns):
            policy_losses.append(-log_prob * A)
            value_losses.append(F.smooth_l1_loss(state_value, torch.tensor([R])))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.dones[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        
        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * self.lambda_
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages)

        return advantages

def train(lr=0.01):
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    ewma_reward = 0
    
    for i_episode in range(3000):
        state = env.reset()
        ep_reward = 0
        t = 0

        scheduler.step()
        for i in range(9999):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)

            model.dones.append(done)
            model.rewards.append(reward)

            ep_reward += reward

            if done:
                t = i+1
                break

        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))


        writer.add_scalar('lr', scheduler.optimizer.param_groups[0]['lr'], i_episode)
        writer.add_scalar('ewma_reward', ewma_reward, i_episode)
        writer.add_scalar('length', t, i_episode)


        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_GAE_{}_{}_{}.pth'.format(lr, args.lam, args.gamma))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    random_seed = 10  
    lr = args.lr
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'LunarLander_GAE_{args.lr}_{args.lam}_{args.gamma}.pth')
