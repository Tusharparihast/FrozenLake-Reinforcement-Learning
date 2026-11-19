import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlearning import sarsa_update


def test_sarsa_update_non_terminal_transition():
    Q = np.zeros((3, 4))
    Q[0, 1] = 0.2
    Q[1] = [0.1, 0.8, 0.3, 0.4]

    updated_Q = sarsa_update(
        Q, state=0, action=1, reward=0.5, next_state=1,
        next_action=2, terminated=False, alpha=0.1, gamma=0.9
    )

    # SARSA uses Q[next_state, next_action] = Q[1, 2] = 0.3.
    # target = 0.5 + 0.9 * 0.3 = 0.77
    # updated value = 0.2 + 0.1 * (0.77 - 0.2) = 0.257
    assert np.isclose(updated_Q[0, 1], 0.257)


def test_sarsa_update_terminal_transition():
    Q = np.zeros((3, 4))
    Q[0, 1] = 0.2
    Q[2] = [10.0, 9.0, 8.0, 7.0]

    updated_Q = sarsa_update(
        Q, state=0, action=1, reward=0.5, next_state=2,
        next_action=None, terminated=True, alpha=0.1, gamma=0.9
    )

    # Terminal transitions do not bootstrap from a next action.
    # updated value = 0.2 + 0.1 * (0.5 - 0.2) = 0.23
    assert np.isclose(updated_Q[0, 1], 0.23)


if __name__ == "__main__":
    test_sarsa_update_non_terminal_transition()
    test_sarsa_update_terminal_transition()
    print("Both SARSA update tests passed.")
