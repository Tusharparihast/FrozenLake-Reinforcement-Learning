
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

