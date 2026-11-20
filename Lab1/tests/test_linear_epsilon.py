import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlearning import linear_epsilon


def test_epsilon_starts_at_initial_value():
    assert linear_epsilon(0, 100_000) == pytest.approx(1.0)


def test_epsilon_reaches_minimum_at_end_of_decay_period():
    decay_episodes = int(100_000 * 0.8)
    assert linear_epsilon(decay_episodes - 1, 100_000) == pytest.approx(0.01)


def test_epsilon_stays_at_minimum_after_decay():
    assert linear_epsilon(99_999, 100_000) == pytest.approx(0.01)


def test_epsilon_supports_a_full_training_decay():
    assert linear_epsilon(9, 10, decay_fraction=1.0) == pytest.approx(0.01)


def test_invalid_decay_fraction_is_rejected():
    with pytest.raises(ValueError):
        linear_epsilon(0, 100, decay_fraction=0.0)
