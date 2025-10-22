import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# DQN模型定义
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # self.fc1 = nn.Linear(input_size, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, output_size)
        self.fc1 = nn.Linear(21 * 983, 128)  # 21个Agent(14车加7人)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)  # num_actions 是您的输出大小

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
        # 假设 state 是一个元组，每个元素都是一个 NumPy 数组
        padded_state = []
        for arr in state:
            # 检查当前数组的长度
            current_length = len(arr)
            if current_length < target_length:
                # 如果当前长度小于目标长度，用 0 填充
                padded_arr = np.pad(arr, (0, target_length - current_length), mode='constant', constant_values=0)
            else:
                # 如果当前长度大于目标长度，截断
                padded_arr = arr[:target_length]
            padded_state.append(padded_arr)

        # 将填充后的状态转换为 NumPy 数组
        return np.array(padded_state)

    # def act(self, state):
    #     # 使用填充函数处理 state
    #     state = self.pad_state(state)
    #     state = torch.FloatTensor(state)  # 转换为 PyTorch 张量
    #     with torch.no_grad():
    #         q_values = self.model(state)  # 假设模型期望的输入是一个张量
    #     action = torch.argmax(q_values).item()  # 选择最大 Q 值对应的动作
    #     return action


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if len(state) > 1:
            state = self.pad_state(state)
        state = self.pad_and_stack(state)
        state = torch.FloatTensor(state)
        
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())  # 返回 Q 最大的动作

    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target += self.gamma * np.amax(self.model(torch.FloatTensor(next_state)).detach().numpy())
    #         target_f = self.model(torch.FloatTensor(state))
    #         target_f[0][action] = target
    #         self.optimizer.zero_grad()
    #         loss = self.loss_function(target_f, self.model(torch.FloatTensor(state)))
    #         loss.backward()
    #         self.optimizer.step()
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay


    # def pad_and_stack(self, next_state):
    #     # 找到最大长度
    #     max_length = max(arr.shape[0] for arr in next_state)

    #     # 创建填充后的数组列表
    #     padded_arrays = []
    #     for arr in next_state:
    #         # 每个数组的当前形状
    #         current_length = arr.shape[0]
    #         # 计算需要填充的长度
    #         padding_length = max_length - current_length
    #         # 使用 np.pad 进行填充
    #         padded_array = np.pad(arr, (0, padding_length), mode='constant', constant_values=0)
    #         padded_arrays.append(padded_array)

    #     # 将所有填充后的数组堆叠成一个二维数组
    #     padded_state_array = np.stack(padded_arrays)
    #     return padded_state_array
    
    # 修改 pad_and_stack 函数，使得最终形状为 (21 * max_length,)
    def pad_and_stack(self, next_state):
        max_length = max(arr.shape[0] for arr in next_state)
        padded_arrays = []
        for arr in next_state:
            current_length = arr.shape[0]
            padding_length = max_length - current_length
            padded_array = np.pad(arr, (0, padding_length), mode='constant', constant_values=0)
            padded_arrays.append(padded_array)

        # 将所有填充后的数组堆叠成一个二维数组
        padded_state_array = np.stack(padded_arrays)
        # 将二维数组展平为一维数组
        return padded_state_array.flatten()  # 变为一维数组

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            # 调用填充函数
            next_state_array = self.pad_and_stack(next_state)
            
            # 填充当前状态
            # state_array = self.pad_and_stack([state])  # 将 state 填充到正确的形状
            state_array = self.pad_and_stack(next_state)  # 将 state 填充到正确的形状

            if not done:
                q_values = self.model(torch.FloatTensor(next_state_array)).detach().numpy()  # 通过模型获取 Q 值
                target += self.gamma * np.amax(q_values)  # 取最大值

            # 更新 target_f，确保其形状正确
            target_f = self.model(torch.FloatTensor(state_array)).detach()  # 获取当前状态的 Q 值
            # 转换 target 为 PyTorch 张量
            target_tensor = torch.tensor(target, dtype=torch.float32)
            # target_f[0][action] = target_tensor  # 更新对应 action 的 Q 值
            target_f[action] = target_tensor  # 使用单个索引

            # 计算损失并更新模型
            self.optimizer.zero_grad()
            loss = self.loss_function(target_f, self.model(torch.FloatTensor(state_array)))
            loss.backward()
            self.optimizer.step()

        # 减少 epsilon 值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay









def heuristic_episode(env, render=False, seed=None):
    # 根据观察空间计算状态和动作的维度
    state_size = sum([space.shape[0] for space in env.observation_space])  # 假设有多个代理
    action_size = len(env.action_id_to_coords_map)  # 动作空间大小
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    # 重置环境，并获取初始状态
    initial_state = env.reset(seed=seed)  # 假设 reset 返回初始状态
    state = initial_state[0]  # 获取状态信息
    done = False
    all_infos = []
    timestep = 0
    global_episode_return = 0
    episode_returns = np.zeros(env.num_agents)

    while not done:
        # 确保 actions 是一个列表
        actions = [agent.act(state) for _ in range(env.num_agents)]  # 为每个代理生成动作

        # 执行动作并处理返回值
        step_result = env.step(actions)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
        else:
            print(f"Unexpected step result: {step_result}")
            # 假设只返回奖励
            reward = step_result
            next_state = None  # 如果没有状态信息，需要通过其他方式获取
            terminated = False
            truncated = False
            info = {}

        # 根据返回值更新状态和统计信息
        done = all(terminated) or all(truncated)
        agent.remember(state, actions, reward, next_state, done)
        agent.replay(batch_size)  # 训练模型

        # 更新状态和奖励统计
        state = next_state
        global_episode_return += sum(reward)  # 累加奖励
        episode_returns += reward  # 累加各代理的奖励
        all_infos.append(info)
        timestep += 1

        if render:
            env.render(mode="human")

    return all_infos, global_episode_return, episode_returns



