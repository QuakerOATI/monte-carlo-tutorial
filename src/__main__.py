from .env import env
from .control import monte_carlo_epsilon_soft
from .test_policy import test_policy

policy = monte_carlo_epsilon_soft(env, episodes=5000)
test_policy(policy, env)
