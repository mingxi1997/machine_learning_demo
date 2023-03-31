import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 超参数
gamma = 0.99
clip_epsilon = 0.1
lr = 0.001
epochs = 10000
update_interval = 2000

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        value = self.critic(state)
        return action_prob, value

def select_action(model, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_prob, _ = model(state)
    distribution = Categorical(action_prob)
    action = distribution.sample()
    return action.item(), distribution.log_prob(action)



def train(model, optimizer, trajectory, gamma=0.99, gae_lambda=0.95, n_minibatches=4, n_epochs=4, clip_epsilon=0.2):
    states, actions, rewards, log_probs, next_states, dones = zip(*trajectory)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # 提前计算所有状态的值函数
    _, values = model(states)
    values = values.squeeze(1).detach()

    # 计算GAE
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_value = model(next_states[-1])[1].item()
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    # 计算目标值（target values）
    target_values = advantages + values

    for epoch in range(n_epochs):
        # 随机打乱数据
        indices = torch.randperm(len(states))

        # 使用minibatches进行更新
        for start_idx in range(0, len(states), n_minibatches):
            end_idx = start_idx + n_minibatches
            batch_indices = indices[start_idx:end_idx]

            # 提取minibatch
            batch_states = states[batch_indices].detach()
            batch_actions = actions[batch_indices].detach()
            batch_target_values = target_values[batch_indices].detach()
            batch_advantages = advantages[batch_indices].detach()
            batch_log_probs_old = log_probs[batch_indices].detach()

            # 计算新的动作概率和值函数
            action_probs, values = model(batch_states)
            action_probs = action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            values = values.squeeze(1)
            entropy = -(action_probs * action_probs.log()).sum(-1).mean()

            # 计算比率
            ratios = torch.exp(action_probs.log() - batch_log_probs_old)

            # 计算PPO目标损失函数
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 计算值函数损失
            value_loss = 0.5 * (batch_target_values - values).pow(2).mean()

            # 计算总损失并进行优化
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

state,_= env.reset()
total_rewards = []
total_steps = 0
trajectory = []
anchor_reward=100
for epoch in range(epochs):
    done = False
    total_reward = 0
    while not done:
        action, log_prob = select_action(model, state)
        next_state, reward, done, info, _ = env.step(action)

        total_reward += reward
        total_steps += 1

        trajectory.append((state, action, reward, log_prob, next_state, done))

        state = next_state

        if total_steps % update_interval == 0:
            train(model, optimizer, trajectory)
            trajectory = []

        if done:
            state,_ = env.reset()

    total_rewards.append(total_reward)
    

    print(f"Epoch: {epoch}, Reward: {total_reward}")

# 测试训练好的智能体
from gym.envs.classic_control import CartPoleEnv

env = CartPoleEnv(render_mode='human')
state,_= env.reset()
done = False
total_reward = 0
while not done:
    env.render()
    action, _ = select_action(model, state)
    state, reward, done, info,_ = env.step(action)
    total_reward += reward

print(f"Test Reward: {total_reward}")
env.close()
