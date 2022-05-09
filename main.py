from game import Game
from test import Test
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

def main():

    episode_durations = []
    # def plot_durations():
    #     plt.figure(2)
    #     plt.clf()
    #     durations_t = torch.FloatTensor(episode_durations)
    #     plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Duration')
    #     plt.plot(durations_t.numpy())
    #     # Take 100 episode averages and plot them too
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())

    #     plt.pause(0.001)  # pause a bit so that plots are updated

    # Parameters
    num_episode = 1000
    count = 100
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.99

    # env = gym.make('CartPole-v0')
    policy_net = PolicyNet()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0


    for e in range(num_episode):
        game = Game()
        game.generate_number()
        game.generate_number()
        state = game.field.copy().reshape([1,-1])
        state = torch.from_numpy(state).float()

        for t in range(count):
            action = policy_net(state)[0]
            with torch.no_grad():
                next_state, reward, done = game.step(action.numpy())
                print('r', reward)
            # env.render(mode='rgb_array')

            # To mark boundarys between episodes
            if done:
                reward = 0

            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)

            state = next_state.reshape([1,-1])
            state = torch.from_numpy(state).float()
            # state = Variable(state)

            steps += 1

            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                # print(steps)
                action = action_pool[i].reshape(1,-1)
                reward = reward_pool[i]

                probs = policy_net(state)
                print('reward', reward)
                loss = (-torch.log(action) * reward).sum()  # Negtive score function x reward
                print('loss:', loss)
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0


if __name__ == '__main__':
    main()