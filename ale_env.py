""" Arcade Learning Environment wrappers for the classic and modern training protocols.

    All credits go to the original creator of this wrapper
    [@Kaixhin](https://github.com/Kaixhin/Rainbow/blob/master/env.py),
    except for the bugs, those go to me.
"""
import os
import random
from collections import deque

import torch
from ale_py import ALEInterface, LoggerMode, roms
from gym.spaces import Discrete

try:
    import cv2
except ModuleNotFoundError as err:
    print(
        "\nOpenCV is required when using the ALE env wrapper. ",
        "Try `conda install -c conda-forge opencv`.\n",
    )


__all__ = ["ALEModern"]


ALEInterface.setLoggerMode(LoggerMode.Error)  # use it to adijust the logging level


def _print_cols(arr, ncol=3):
    rows = [arr[offs : offs + ncol] for offs in range(0, len(arr), ncol)]
    for row in rows:
        tmplt = "{:<20}" * len(row)
        print(tmplt.format(*row))


def _get_rom(game):
    try:
        rom = getattr(roms, game)
    except AttributeError:
        print(f"{len(roms.__all__)} available roms:")
        _print_cols(roms.__all__)
        raise
    return rom


class ALEModern:
    """ A wrapper over atari_py, the Arcade Learning Environment python
    bindings that follows the Dopamine protocol, which in turn, follows (Machado, 2017):
        - frame concatentation of `history_len=4`
        - maximum episode length of 108,000 frames
        - sticky action probability `sticky_action_p=0.25`
        - end game after only after all lives have been lost
        - clip rewards during training to (1, -1)
        - frame skipping of 4 frames
        - minimal action set

    Returns:
        env: An ALE object with settings simillar to Dopamine's environment.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        game,
        seed,
        device,
        clip_rewards_val=1,
        history_length=4,
        sticky_action_p=0.25,
        max_episode_length=108e3,
        sdl=True,
        mode=None,
        difficulty=None,
        minimal_action_set=True,
        record_dir=None,
    ):
        # pylint: enable=bad-continuation
        self.game_name = game
        self.device = device
        self.sticky_action_p = sticky_action_p
        self.window = history_length
        self.clip_val = clip_rewards_val

        # configure ALE
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("max_num_frames_per_episode", int(max_episode_length))
        self.ale.setFloat("repeat_action_probability", self.sticky_action_p)
        self.ale.setInt("frame_skip", 1)  # we handle frame skipping in this wrapper
        self.ale.setBool("color_averaging", False)  # we use max pooling instead
        if sdl:
            self.ale.setBool("sound", True)
            self.ale.setBool("display_screen", True)
        if record_dir is not None:
            self.ale.setString("record_screen_dir", record_dir)
            self.ale.setString(
                "record_sound_filename", os.path.join(record_dir, "sound.wav")
            )

        self.ale.loadROM(_get_rom(self.game_name))

        # set mode and difficulty
        self._set_mode(mode)
        self._set_difficulty(difficulty)

        # buffer used for stacking frames
        self.state_buffer = deque([], maxlen=self.window)

        # configure action space
        actions = (
            self.ale.getMinimalActionSet()
            if minimal_action_set
            else self.ale.getLegalActionSet()
        )
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = Discrete(len(self.actions))

    def _get_state(self):
        state = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA,
        )
        return torch.tensor(state, dtype=torch.uint8, device=self.device)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(84, 84, device=self.device, dtype=torch.uint8)
            )

    def reset(self):
        """ Reset the environment, return initial observation. """
        # reset internals
        self._reset_buffer()
        self.ale.reset_game()

        # process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        """ Advance the environment given the agent's action.

        Args:
            action (int): Agent's action.
        Returns:
            tuple: The environment's observation.
        """
        # repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device, dtype=torch.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        # clip the reward
        if self.clip_val:
            clipped_reward = max(min(reward, self.clip_val), -self.clip_val)
        else:
            clipped_reward = reward

        # return state, reward, done
        state = torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()
        return state, clipped_reward, done, {"true_reward": reward}

    def close(self):
        pass

    def _set_mode(self, mode):
        if mode is not None:
            available_modes = self.ale.getAvailableModes()
            assert mode in available_modes, f"mode not in {available_modes}"
            self.ale.setMode(mode)

    def _set_difficulty(self, difficulty):
        if difficulty is not None:
            available_difficulties = self.ale.getAvailableDifficulties()
            assert (
                difficulty in available_difficulties
            ), f"difficulty not in {available_difficulties}"
            self.ale.setDifficulty(difficulty)

    def set_mode_interactive(self):
        # set modes and difficultes
        print("Available modes:        ", self.ale.getAvailableModes())
        print("Available difficulties: ", self.ale.getAvailableDifficulties())
        self._set_mode(int(input("Select mode: ")))
        self._set_difficulty(int(input("Select difficulty: ")))
        self.ale.reset_game()

    def __str__(self):
        """ User friendly representation of this class. """
        stochasticity = (
            f"{self.sticky_action_p:.2f}_sticky_action"
            if self.sticky_action_p
            else "deterministic"
        )
        return (
            "ALEModern(game={}, stochasticity={}, hist_len={}, repeat_act=4, clip_rewards={})"
        ).format(self.game_name, stochasticity, self.window, self.clip_val)


class ALEClassic(ALEModern):
    def __init__(self, game, seed, device, training=False, **kwargs):
        super().__init__(game, seed, device, sticky_action_p=0.0, **kwargs)

        self.training = training
        self.lives = 0  # life counter
        self.life_termination = False  # used to check if a life was lost

    def _random_noops(self):
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Do the random ops
            self._random_noops()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device, dtype=torch.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                # Only set flag when not truly done
                self.life_termination = not done
                done = True
            self.lives = lives
        # clip the reward
        if self.clip_val and self.training:
            reward = max(min(reward, self.clip_val), -self.clip_val)
        # Return state, reward, done
        state = torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()
        return state, reward, done, {}

    def train(self):
        """ Switches the env to training phase
            and uses the loss of life as a training signal.
        """
        self.training = True

    def eval(self):
        """ Switches the env to evaluation phase
            and uses the standard game over as a training signal.
        """
        self.training = False

    def __str__(self):
        phase = "train" if self.training else "eval"
        stochasticity = "no_op_30"

        return (
            "ALEClassic(game={}, phase={}, stochasticity={}, hist_len={},"
            " repeat_act=4, clip_rewards={})"
        ).format(self.game_name, phase, stochasticity, self.window, self.clip_val)
