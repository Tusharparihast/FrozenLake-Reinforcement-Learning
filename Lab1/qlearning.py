import numpy as np
import gymnasium as gym


def make_env(max_episode_steps=200):
    """Create the slippery 8x8 FrozenLake environment used in this lab."""
    return gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=True,
        max_episode_steps=max_episode_steps,
    )


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action with an epsilon-greedy policy.

    With probability epsilon, explore by choosing any action uniformly.
    Otherwise, choose uniformly among the actions with the highest Q-value.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    n_actions = Q.shape[1]

    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))

    values = Q[state]
    best_actions = np.flatnonzero(values == np.max(values))
    return int(np.random.choice(best_actions))


def linear_epsilon(episode, n_episodes, eps_start=1.0, eps_end=0.01,
                   decay_fraction=0.8):
    """Linearly reduce epsilon, then keep it at eps_end for the rest of training."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if not 0.0 <= eps_end <= eps_start <= 1.0:
        raise ValueError("Require 0 <= eps_end <= eps_start <= 1")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    if episode < 0:
        raise ValueError("episode cannot be negative")

    decay_episodes = max(1, int(n_episodes * decay_fraction))
    progress = min(episode / max(decay_episodes - 1, 1), 1.0)
    return eps_start + progress * (eps_end - eps_start)


def q_learning_update(Q, state, action, reward, next_state,
                      terminated, alpha, gamma):
    """Apply one tabular, off-policy Q-learning update."""
    if terminated:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state])

    Q[state, action] += alpha * (target - Q[state, action])
    return Q


def sarsa_update(Q, state, action, reward, next_state, next_action,
                 terminated, alpha, gamma):
    """Apply one tabular, on-policy SARSA update."""
    if terminated:
        target = reward
    else:
        target = reward + gamma * Q[next_state, next_action]

    Q[state, action] += alpha * (target - Q[state, action])
    return Q


def train(env, episodes=100000, alpha=0.1, gamma=0.99,
          eps_start=1.0, eps_end=0.01, algorithm="q_learning",
          seed=0, snapshot_interval=1000, decay_fraction=0.8):
    """
    Train a Q-learning or SARSA agent in a discrete Gymnasium environment.

    The environment should be created with make_env() so the episode limit
    matches the 8x8 FrozenLake experiment configuration.

    Returns:
        Q: learned state-action value table.
        episode_returns: total reward collected in each episode.
        episode_lengths: number of steps taken in each episode.
        q_snapshots: Q-table copies indexed by completed episode number.
    """
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval cannot be negative")
    if not 0.0 <= eps_end <= eps_start <= 1.0:
        raise ValueError("Require 0 <= eps_end <= eps_start <= 1")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")

    np.random.seed(seed)
    env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)

    episode_returns = []
    episode_lengths = []
    q_snapshots = {}

    for episode in range(episodes):
        epsilon = linear_epsilon(
            episode, episodes, eps_start, eps_end, decay_fraction
        )

        if episode == 0:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        action = epsilon_greedy(Q, state, epsilon)

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated

            if algorithm == "q_learning":
                # A time-limit truncation is not a terminal MDP state:
                # bootstrap unless the environment truly terminated.
                q_learning_update(
                    Q, state, action, reward, next_state,
                    terminated, alpha, gamma
                )
                if not episode_done:
                    next_action = epsilon_greedy(Q, next_state, epsilon)

            else:
                # SARSA uses the action selected by the current policy.
                # Select it on truncation too, because truncation still bootstraps.
                if not terminated:
                    next_action = epsilon_greedy(Q, next_state, epsilon)
                else:
                    next_action = None

                sarsa_update(
                    Q, state, action, reward, next_state, next_action,
                    terminated, alpha, gamma
                )

            total_reward += reward
            steps += 1
            state = next_state

            if not episode_done:
                action = next_action

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        completed = episode + 1
        if snapshot_interval and completed % snapshot_interval == 0:
            q_snapshots[completed] = Q.copy()

    return Q, episode_returns, episode_lengths, q_snapshots
