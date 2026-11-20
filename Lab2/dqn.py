"""Deep Q-Network (DQN) on CartPole-v1 (Tasks 1 to 4)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class ReplayBuffer:
    """Task 1: fixed-size circular buffer; the oldest transition is overwritten when full."""

    def __init__(self, capacity, obs_dim=4, seed=0):
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.dones[i] = next_state, float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Uniform random batch as tensors: states, actions, rewards, next_states, dones (0.0 or 1.0)."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(a[idx]) for a in
                     (self.states, self.actions, self.rewards, self.next_states, self.dones))

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    """Task 2: MLP 4 -> 256 -> 256 -> 2 with ReLU, He initialisation and no softmax."""

    def __init__(self, obs_dim=4, n_actions=2, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


def dqn_loss(online_net, target_net, batch, gamma):
    """Task 3: MSE between Q(s,a) and y = r + gamma * (1 - done) * max_a' Q_target(s',a')."""
    states, actions, rewards, next_states, dones = batch
    q_all = online_net(states)
    q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    target_net.eval()
    with torch.no_grad():
        y = rewards + gamma * (1.0 - dones) * target_net(next_states).max(dim=1).values
    return F.mse_loss(q_taken, y), q_all.max(dim=1).values.mean().item()


def epsilon_at(step, warmup_steps, eps_start, eps_end, eps_decay_steps):
    """1.0 during warm-up, then linear from eps_start to eps_end by step eps_decay_steps."""
    if step < warmup_steps:
        return 1.0
    return eps_start + min(1.0, step / eps_decay_steps) * (eps_end - eps_start)


def greedy_action(net, state):
    with torch.no_grad():
        return int(net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).argmax(dim=1).item())


def evaluate(net, env, n_episodes):
    """Mean return of the greedy policy."""
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(greedy_action(net, state))
            total += reward
            done = terminated or truncated
    return total / n_episodes


def train_dqn(n_episodes=500, seed=0, gamma=0.99, lr=5e-4, batch_size=64, buffer_capacity=10000,
              warmup_steps=1000, eps_start=1.0, eps_end=0.05, eps_decay_steps=10000,
              target_sync_every=500, eval_every=20, eval_episodes=5, hidden=256):
    """
    Task 4: train DQN on CartPole. target_sync_every=1 means no real target network.

    Returns a dict with episode returns, epsilons, losses, mean max-Q per
    gradient step, and the evaluation episodes and returns.
    """
    torch.set_num_threads(1)  # one thread makes runs with the same seed identical
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    online_net = QNetwork(obs_dim, n_actions, hidden)
    target_net = QNetwork(obs_dim, n_actions, hidden)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, obs_dim, seed=seed)

    returns, epsilons, losses, mean_max_q, eval_eps, eval_rets = [], [], [], [], [], []
    step = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        if episode == 0:
            eval_env.reset(seed=seed + 1000)
        episode_return = 0.0
        done = False
        epsilons.append(epsilon_at(step, warmup_steps, eps_start, eps_end, eps_decay_steps))

        while not done:
            epsilon = epsilon_at(step, warmup_steps, eps_start, eps_end, eps_decay_steps)
            action = int(rng.integers(n_actions)) if rng.random() < epsilon else greedy_action(online_net, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store terminated, not truncated: the 500-step limit is not a real end state.
            buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            episode_return += reward
            step += 1

            if step > warmup_steps:
                loss, max_q = dqn_loss(online_net, target_net, buffer.sample(batch_size), gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                mean_max_q.append(max_q)

            if step % target_sync_every == 0:
                target_net.load_state_dict(online_net.state_dict())

        returns.append(episode_return)
        if (episode + 1) % eval_every == 0:
            eval_eps.append(episode + 1)
            eval_rets.append(evaluate(online_net, eval_env, eval_episodes))

    env.close()
    eval_env.close()
    return {"returns": np.array(returns), "epsilons": np.array(epsilons), "losses": np.array(losses),
            "mean_max_q": np.array(mean_max_q), "eval_episodes": np.array(eval_eps),
            "eval_returns": np.array(eval_rets)}


def moving_average(x, window=100):
    return np.convolve(x, np.ones(window) / window, mode="valid")