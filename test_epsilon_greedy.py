
import numpy as np
from qlearning import epsilon_greedy

Q = np.array([
    [0.2, 0.8, 0.8, 0.1]
])

# Test 1: epsilon = 0 (only greedy actions)
np.random.seed(42)

actions = [
    epsilon_greedy(Q, 0, 0.0)
    for _ in range(1000)
]

assert set(actions) == {1, 2}
print("Test 1 passed: Greedy action selection")

# Test 2: epsilon = 1 (only random actions)
np.random.seed(42)

actions = [
    epsilon_greedy(Q, 0, 1.0)
    for _ in range(1000)
]

assert set(actions) == {0, 1, 2, 3}
print("Test 2 passed: Random action selection")

print("All epsilon-greedy tests passed!")
