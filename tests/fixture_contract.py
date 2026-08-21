FPS = 10
EPISODE_LENGTHS = (5, 7)
STATE_DIM = 3
ACTION_DIM = 2
MAIN_KEY = "observation.images.main"
WRIST_KEY = "observation.images.wrist"
SAMPLE_INDICES = (0, 1, 4, 5, 11)


def state_value(episode: int, frame: int) -> list[float]:
    return [episode, frame, episode * 100 + frame]


def action_value(episode: int, frame: int) -> list[float]:
    value = episode * 10 + frame
    return [value, -value]
