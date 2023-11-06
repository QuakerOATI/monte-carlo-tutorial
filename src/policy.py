from typing import Dict, TypeAlias
from gymnasium import Env

Policy: TypeAlias = Dict[int, Dict[int, float]]


def random_policy(env: Env):
    policy = {}
    for state in range(env.observation_space.n):
        policy[state] = {
            action: 1 / env.action_space.n
            for action in range(env.action_space.n)
        }
    return policy
