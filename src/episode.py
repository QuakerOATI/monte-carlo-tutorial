from typing import Union, List
from dataclasses import dataclass
from gymnasium import Env

from .policy import Policy
from .io import show_env
from .utils import get_random_weighted_key


@dataclass
class Timestep:
    state: int
    action: int
    reward: Union[float, int]

    @property
    def state_action(self):
        return self.state, self.action


def play_episode(env: Env, policy: Policy, display=True) -> List[Timestep]:

    env.reset()
    episode = []
    done = False

    while not done:
        state = env.unwrapped.s
        show_env(env, display)

        action = get_random_weighted_key(policy[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        episode.append(Timestep(state=state, action=action, reward=reward))
        done = terminated or truncated

    show_env(env, display)
    return episode
