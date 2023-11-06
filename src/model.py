from .policy import Policy
from gymnasium import Env


def create_qmatrix(env: Env, policy: Policy):
    return {
        state: {
            action: 0.0
            for action in range(env.action_space.n)
        }
        for state in policy.keys()
    }
