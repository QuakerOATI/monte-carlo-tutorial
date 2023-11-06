"""Setup RL environment.

For this tutorial, this just wraps the Gymnasium "FrozenLake" constructor.
"""

import gymnasium as gym

env = gym.make("FrozenLake8x8-v1", render_mode="rgb_array")
