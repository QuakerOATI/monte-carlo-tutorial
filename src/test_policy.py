from gymnasium import Env
from .policy import Policy
from .episode import play_episode


def test_policy(policy: Policy, env: Env, runs: int = 100) -> float:
    wins = 0
    for run in range(runs):
        episode = play_episode(env, policy, display=False)
        if episode[-1].reward == 1:
            wins += 1
    ratio = wins / runs
    print(f"Ratio: {ratio}")
    return ratio
