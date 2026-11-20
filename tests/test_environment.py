import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlearning import make_env


def test_frozenlake_environment_configuration():
    env = make_env()
    try:
        assert env.observation_space.n == 64
        assert env.action_space.n == 4
        assert env.spec.max_episode_steps == 200
    finally:
        env.close()
