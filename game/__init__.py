from .snake_game import SnakeGameAI
from .flappy_game import FlappyGameAI

ENV_SPECS = {
    "snake": {
        "cls": SnakeGameAI, 
        "input_size": 11, 
        "output_size": 3, 
        "default_best": "snake_best.pth",
        "default_latest": "snake_latest.pth",
        # "default_model": "snake_model.pth",
    },
    "flappy": {
        "cls": FlappyGameAI, 
        "input_size": 8, 
        "output_size": 2, 
        "default_best": "flappy_best.pth",
        "default_latest": "flappy_latest.pth",
        # "default_model": "flappy_model.pth"
    },
}

def make_env(env_name: str, *, render: bool, fps: int, shaping: bool):
    spec = get_env_spec(env_name)
    cls = spec["cls"]
    return cls(render=render, fps=fps, shaping=shaping)

def get_env_spec(env_name: str):
    env_name = env_name.lower()
    if env_name not in ENV_SPECS:
        raise ValueError(f"Unknown env '{env_name}'. Choose from: {list(ENV_SPECS.keys())}")
    return ENV_SPECS[env_name]
