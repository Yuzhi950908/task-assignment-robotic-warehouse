"""
please note that! this file just about algorithum, more details from warehouse environment pls see warehouse.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, output_size)
        self.fc1 = nn.Linear(21 * 983, 128)  # 21Agent(14agvs + 7 people)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)  # num_actions 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def pad_state(self, state, target_length=983):
        padded_state = []
        for arr in state:
            # check length
            current_length = len(arr)
            if current_length < target_length:
                padded_arr = np.pad(arr, (0, target_length - current_length), mode='constant', constant_values=0)
            else:
                padded_arr = arr[:target_length]
            padded_state.append(padded_arr)

        return np.array(padded_state)




    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if len(state) > 1:
            state = self.pad_state(state)
        state = self.pad_and_stack(state)
        state = torch.FloatTensor(state)
        
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())  # return Q max


    

    def pad_and_stack(self, next_state):
        max_length = max(arr.shape[0] for arr in next_state)
        padded_arrays = []
        for arr in next_state:
            current_length = arr.shape[0]
            padding_length = max_length - current_length
            padded_array = np.pad(arr, (0, padding_length), mode='constant', constant_values=0)
            padded_arrays.append(padded_array)

        padded_state_array = np.stack(padded_arrays)
        return padded_state_array.flatten()  

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            next_state_array = self.pad_and_stack(next_state)
            
            state_array = self.pad_and_stack(next_state)  

            if not done:
                q_values = self.model(torch.FloatTensor(next_state_array)).detach().numpy()  
                target += self.gamma * np.amax(q_values)  

            # update target_f
            target_f = self.model(torch.FloatTensor(state_array)).detach()  # current Q
            target_tensor = torch.tensor(target, dtype=torch.float32)
            # target_f[0][action] = target_tensor  
            target_f[action] = target_tensor  # new action

            # update loss and para
            self.optimizer.zero_grad()
            loss = self.loss_function(target_f, self.model(torch.FloatTensor(state_array)))
            loss.backward()
            self.optimizer.step()

        # decay epsilon 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay









def heuristic_episode(env, render=False, seed=None):
    # update new state
    state_size = sum([space.shape[0] for space in env.observation_space])  
    action_size = len(env.action_id_to_coords_map)  
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    # updata new env
    initial_state = env.reset(seed=seed)  
    state = initial_state[0]  
    done = False
    all_infos = []
    timestep = 0
    global_episode_return = 0
    episode_returns = np.zeros(env.num_agents)

    while not done:
        actions = [agent.act(state) for _ in range(env.num_agents)]  

        step_result = env.step(actions)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
        else:
            print(f"Unexpected step result: {step_result}")
            reward = step_result
            next_state = None 
            terminated = False
            truncated = False
            info = {}

        done = all(terminated) or all(truncated)
        agent.remember(state, actions, reward, next_state, done)
        agent.replay(batch_size)  

        # update reward
        state = next_state
        global_episode_return += sum(reward)  
        episode_returns += reward  
        all_infos.append(info)
        timestep += 1

        if render:
            env.render(mode="human")

    return all_infos, global_episode_return, episode_returns



