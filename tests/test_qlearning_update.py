import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlearning import q_learning_update


def test_q_learning_update_non_terminal_transition():
    Q = np.zeros((3, 4))
    Q[0, 1] = 0.2
    Q[1] = [0.1, 0.8, 0.3, 0.4]

    updated_Q = q_learning_update(
        Q, state=0, action=1, reward=0.5, next_state=1,
        terminated=False, alpha=0.1, gamma=0.9
    )

    # target = 0.5 + 0.9 * 0.8 = 1.22
    # updated value = 0.2 + 0.1 * (1.22 - 0.2) = 0.302
    assert np.isclose(updated_Q[0, 1], 0.302)


def test_q_learning_update_terminal_transition():
    Q = np.zeros((3, 4))
    Q[0, 1] = 0.2
    Q[2] = [10.0, 9.0, 8.0, 7.0]

    updated_Q = q_learning_update(
        Q, state=0, action=1, reward=0.5, next_state=2,
        terminated=True, alpha=0.1, gamma=0.9
    )

    # Terminal transitions do not include a discounted future Q-value.
    # updated value = 0.2 + 0.1 * (0.5 - 0.2) = 0.23
    assert np.isclose(updated_Q[0, 1], 0.23)


if __name__ == "__main__":
    test_q_learning_update_non_terminal_transition()
    test_q_learning_update_terminal_transition()
    print("Both Q-learning update tests passed.")
