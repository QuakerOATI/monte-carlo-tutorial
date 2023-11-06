"""Create a Monte Carlo optimizer for a given environment and policy.
"""

from gymnasium import Env
from collections import defaultdict

from .policy import Policy, random_policy
from .model import create_qmatrix
from .episode import play_episode
from .utils import argmax, avg


def monte_carlo_epsilon_soft(env: Env,
                             episodes: int = 500,
                             policy: Policy = None,
                             epsilon: float = 0.1) -> Policy:
    if policy is None:
        policy = random_policy(env)
    Q = create_qmatrix(env, policy)
    Q_next = defaultdict(list)

    for _ in range(episodes):
        reward = 0
        episode = play_episode(env, policy)
        state_actions = set()

        for i, tstep in enumerate(episode):
            reward += tstep.reward

            if tstep.state_action not in state_actions:
                Q_next[tstep.state_action].append(reward)
                Q[tstep.state][tstep.action] = avg(Q_next[tstep.state_action])
                best_action = argmax(Q[tstep.state])
                weight = abs(sum(policy[tstep.state].values()))
                policy[tstep.state] = {
                    a: epsilon / weight
                    for a in policy[tstep.state]
                }
                policy[tstep.state][best_action] += 1 - epsilon
                state_actions.add(tstep.state_action)
    return policy
