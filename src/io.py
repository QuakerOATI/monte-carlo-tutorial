from time import sleep
from gymnasium import Env

try:
    from IPython.display import clear_output
except ImportError:

    def clear_output(clear: bool) -> None:
        pass


def show_env(env: Env, display: bool, wait: int = 1):
    if display:
        clear_output(True)
        env.render()
        sleep(wait)
