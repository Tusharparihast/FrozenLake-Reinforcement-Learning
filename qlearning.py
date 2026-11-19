
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
