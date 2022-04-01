import time
from argparse import ArgumentParser

from ale_py import SDL_SUPPORT, ALEInterface, roms
from pynput import keyboard

k2i = {
    "w": 2,  # up
    "a": 4,  # left
    "s": 5,  # down
    "d": 3,  # right
    "c": 1,  # fire
    "n": 0,  # noop
}


def make_env(opt):
    ale = ALEInterface()
    ale.setInt("random_seed", 123)
    ale.setInt("max_num_frames_per_episode", int(108e3))
    ale.setFloat("repeat_action_probability", 0.25)
    ale.setInt("frame_skip", 4)  # 60fps / 5
    ale.setBool("color_averaging", False)
    if SDL_SUPPORT:
        ale.setBool("sound", True)
        ale.setBool("display_screen", True)

    ale.loadROM(getattr(roms, opt.game))

    return ale, ale.getLegalActionSet()


def set_mode_interactive(ale):
    # set modes and difficultes
    available_modes = ale.getAvailableModes()
    available_difficulties = ale.getAvailableDifficulties()
    print("Available modes:        ", available_modes)
    print("Available difficulties: ", available_difficulties)
    mode = int(input("Select mode: "))
    assert mode in available_modes, "Mode not available."
    ale.setMode(mode)
    difficulty = int(input("Select difficulty: "))
    assert difficulty in available_difficulties, "Difficulty not available."
    ale.setDifficulty(difficulty)
    ale.reset_game()


def sync_play(opt, env, action_set):

    events = keyboard.Events()
    events.start()

    ep_returns = [0 for _ in range(opt.episodes)]
    for ep in range(opt.episodes):
        env.reset_game()

        while not env.game_over():

            # listen to keyboard
            event = events.get(opt.timeout)
            if event is None:
                action = action_set[0]
            elif hasattr(event.key, "char"):
                if event.key.char == "q":
                    exit()
                action = action_set[k2i[event.key.char]]

            reward = env.act(action)
            ep_returns[ep] += reward
        print(f"{ep:02d})  Gt: {ep_returns[ep]:7.1f}")
    events.stop()


def async_play(opt, env, action_set):

    # callback
    def on_press(key):
        if hasattr(key, "char"):
            if key.char == "q":
                raise KeyboardInterrupt()
            action = action_set[k2i[key.char]]

        print(action)
        env.act(action)

    with keyboard.Listener(on_press=on_press) as listener:
        for ep in range(opt.episodes):
            env.reset_game()
            while not env.game_over():
                time.sleep(opt.timeout)
                env.act(action_set[0])


def main(opt):
    # set env
    env, action_set = make_env(opt)

    if opt.variations:
        set_mode_interactive(env)

    print("\nPress 'c' (FIRE) or 'n' (NOOP) to start the episode.")

    sync_play(opt, env, action_set)
    # async_play(opt, env, action_set)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("game", type=str, help="game name")
    parser.add_argument(
        "--timeout",
        default=0.04,
        type=float,
        help="how much to wait for a keypress. Default is 0.04",
    )
    parser.add_argument(
        "-e", "--episodes", default=10, type=int, help="number of episodes"
    )
    parser.add_argument(
        "-v",
        "--variations",
        action="store_true",
        help="set mode and difficulty, interactively",
    )
    main(parser.parse_args())
