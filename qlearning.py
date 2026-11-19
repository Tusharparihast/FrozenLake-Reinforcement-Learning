
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using an epsilon-greedy policy.

    Parameters
    ----------
    Q : numpy.ndarray
        Q-table with shape (number_of_states, number_of_actions).
    state : int
        Current state.
    epsilon : float
        Probability of selecting a random action.

    Returns
    -------
    int
        Selected action.
    """

    num_actions = Q.shape[1]

    # Exploration: select a random action
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)

    # Exploitation: identify all actions with maximum Q-value
    q_values = Q[state]
    best_actions = np.flatnonzero(q_values == np.max(q_values))

    # Randomly select among tied best actions
    return np.random.choice(best_actions)


def q_learning_update(Q, state, action, reward, next_state,
                      terminated, alpha, gamma):
    """
    Apply one tabular Q-learning update.

    Parameters
    ----------
    Q : numpy.ndarray
        Q-table with shape (number_of_states, number_of_actions).
    state : int
        State before taking the action.
    action : int
        Action taken in the current state.
    reward : float
        Reward received after taking the action.
    next_state : int
        State reached after taking the action.
    terminated : bool
        Whether the transition ended the episode (goal or hole).
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.

    Returns
    -------
    numpy.ndarray
        Updated Q-table.
    """
    # A terminal state has no future reward to bootstrap from.
    if terminated:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state])

    # Move the current estimate toward the Q-learning target.
    td_error = target - Q[state, action]
    Q[state, action] += alpha * td_error

    return Q


def sarsa_update(Q, state, action, reward, next_state, next_action,
                 terminated, alpha, gamma):
    """
    Apply one tabular SARSA update.

    SARSA is on-policy: its target uses the Q-value of the actual
    next action selected by the current policy.

    Parameters
    ----------
    Q : numpy.ndarray
        Q-table with shape (number_of_states, number_of_actions).
    state : int
        State before taking the current action.
    action : int
        Action taken in the current state.
    reward : float
        Reward received after taking the action.
    next_state : int
        State reached after taking the action.
    next_action : int or None
        Action selected in next_state. Not used for terminal transitions.
    terminated : bool
        Whether the transition ended the episode.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.

    Returns
    -------
    numpy.ndarray
        Updated Q-table.
    """
    if terminated:
        target = reward
    else:
        target = reward + gamma * Q[next_state, next_action]

    td_error = target - Q[state, action]
    Q[state, action] += alpha * td_error

    return Q


def train(env, episodes=10000, alpha=0.1, gamma=0.99,
          eps_start=1.0, eps_end=0.01, algorithm="q_learning",
          seed=0, snapshot_interval=1000):
    """
    Train a tabular Q-learning or SARSA agent.

    Parameters
    ----------
    env : gymnasium.Env
        Discrete environment with observation_space.n and action_space.n.
    episodes : int
        Number of training episodes.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    eps_start : float
        Initial epsilon for exploration.
    eps_end : float
        Final minimum epsilon.
    algorithm : str
        Either "q_learning" or "sarsa".
    seed : int
        Random seed for reproducible training.
    snapshot_interval : int
        Save a copy of the Q-table every this many episodes.
        Set to 0 to disable snapshots.

    Returns
    -------
    tuple
        (Q, episode_returns, episode_lengths, q_snapshots)
        q_snapshots maps episode numbers to copied Q-tables.
    """
    if algorithm not in ("q_learning", "sarsa"):
        raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval cannot be negative")
    if not 0.0 <= eps_end <= eps_start <= 1.0:
        raise ValueError("Require 0 <= eps_end <= eps_start <= 1")

    # Seed both NumPy's action-selection randomness and the environment.
    np.random.seed(seed)
    env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)

    episode_returns = []
    episode_lengths = []
    q_snapshots = {}

    for episode in range(episodes):
        # Linear epsilon decay across episodes.
        fraction = episode / max(episodes - 1, 1)
        epsilon = eps_start + fraction * (eps_end - eps_start)

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
            done = terminated or truncated

            if algorithm == "q_learning":
                q_learning_update(
                    Q, state, action, reward, next_state,
                    done, alpha, gamma
                )
                if not done:
                    next_action = epsilon_greedy(Q, next_state, epsilon)
            else:
                if not done:
                    next_action = epsilon_greedy(Q, next_state, epsilon)
                else:
                    next_action = None

                sarsa_update(
                    Q, state, action, reward, next_state, next_action,
                    done, alpha, gamma
                )

            total_reward += reward
            steps += 1
            state = next_state

            if not done:
                action = next_action

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        completed_episode = episode + 1
        if snapshot_interval and completed_episode % snapshot_interval == 0:
            q_snapshots[completed_episode] = Q.copy()

    return Q, episode_returns, episode_lengths, q_snapshots
