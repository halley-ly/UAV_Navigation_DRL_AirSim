import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import collections

# ---------------- Transformer 模块构建 Actor 和 Critic 网络 ----------------

class TransformerActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=2):
        super(TransformerActor, self).__init__()
        self.input_linear = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.actor_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # state: [batch, state_dim]
        x = self.input_linear(state)  # [batch, hidden_dim]
        # Transformer 需要加入序列维度，此处设定序列长度为1
        x = x.unsqueeze(0)  # [1, batch, hidden_dim]
        x = self.transformer_encoder(x)  # 输出形状同上
        x = x.squeeze(0)  # [batch, hidden_dim]
        mean = self.actor_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class TransformerCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, nhead=4, num_layers=2):
        super(TransformerCritic, self).__init__()
        self.input_linear = nn.Linear(state_dim + action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.critic_linear = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # 将状态和动作拼接
        x = torch.cat([state, action], dim=-1)
        x = self.input_linear(x)
        x = x.unsqueeze(0)  # [1, batch, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # [batch, hidden_dim]
        q_value = self.critic_linear(x)
        return q_value

# ---------------- 经验回放缓冲区 ----------------

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ---------------- SAC Agent 定义 ----------------

class SACAgent:
    def __init__(self, state_dim, action_dim, device, hidden_dim=256,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=-1.0):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.actor = TransformerActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = TransformerCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 温度参数，用于控制策略熵
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state)
        if evaluate:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # reparameterization trick
            action = torch.tanh(x_t)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # 计算目标 Q 值
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_state)
            next_std = next_log_std.exp()
            normal = Normal(next_mean, next_std)
            next_x_t = normal.rsample()
            next_action = torch.tanh(next_x_t)
            log_prob = normal.log_prob(next_x_t) - torch.log(1 - next_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新 Actor 网络
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_new = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action_new.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度参数 alpha
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络参数
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ---------------- AirSim 环境封装 ----------------

class AirSimEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # 可根据实际需要进行更多初始化设置

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        state = self.get_state()
        return state
    def step(self, action):
        # 假设动作包含 [pitch, roll, throttle, yaw_rate]
        pitch, roll, throttle, yaw_rate = action
        # 将 numpy.float32 转换为 Python 原生 float
        pitch = float(pitch)
        roll = float(roll)
        throttle = float(throttle)
        yaw_rate = float(yaw_rate)
        # 这里调用 AirSim API 控制无人机，持续时间设置为 0.1 秒
        self.client.moveByAngleRatesThrottleAsync(pitch, roll, yaw_rate, throttle, duration=0.1).join()
        next_state = self.get_state()
        reward = self.compute_reward(next_state)
        done = self.is_done(next_state)
        return next_state, reward, done, {}


    def get_state(self):
        # 例如：使用无人机位置和姿态作为状态
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        state_vector = np.array([pos.x_val, pos.y_val, pos.z_val,
                                 orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
        return state_vector

    def compute_reward(self, state):
        # 示例奖励函数：距离目标点的负距离（目标设定为 [0, 0, -10]）
        target = np.array([0, 0, -10])
        position = state[:3]
        reward = -np.linalg.norm(position - target)
        return reward

    def is_done(self, state):
        # 示例终止条件：当无人机高度超出设定范围时终止
        position = state[:3]
        if position[2] > 0 or position[2] < -20:
            return True
        return False

# ---------------- 训练主循环 ----------------

def train():
    num_episodes = 1000
    max_steps = 200
    batch_size = 256
    replay_buffer = ReplayBuffer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据 get_state 定义状态维度，此处为 7
    state_dim = 7  
    # 假定动作维度为 4，对应 [pitch, roll, throttle, yaw_rate]
    action_dim = 4  
    agent = SACAgent(state_dim, action_dim, device)

    env = AirSimEnv()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)
            if done:
                break
        print(f"Episode: {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    train()
