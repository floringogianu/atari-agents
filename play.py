from argparse import ArgumentParser
from functools import partial
from gzip import GzipFile
from pathlib import Path

import torch
from torch import nn

from ale_env import ALEModern, ALEClassic


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """
    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val


def main(opt):
    # game/seed/model
    ckpt_path = Path(opt.path)
    game = ckpt_path.parts[-3]

    # recording
    if opt.record:
        record_dir = Path.cwd() / "movies" / Path(*ckpt_path.parts[-4:-1])
        record_dir.mkdir(parents=True, exist_ok=False)
        print("Recording@ ", record_dir)

    # set env
    ALE = ALEModern if "_modern/" in opt.path else ALEClassic
    env = ALE(
        game,
        torch.randint(100_000, (1,)).item(),
        sdl=True,
        device="cpu",
        clip_rewards_val=False,
        record_dir=str(record_dir) if opt.record else None,
    )

    if opt.variations:
        env.set_mode_interactive()

    # init model
    model = AtariNet(env.action_space.n, distributional="C51_" in opt.path)

    # sanity check
    print(env)

    # load state
    ckpt = _load_checkpoint(opt.path)
    model.load_state_dict(ckpt["estimator_state"])

    # configure policy
    policy = partial(_epsilon_greedy, model=model, eps=0.001)
    ep_returns = [0 for _ in range(opt.episodes)]

    for ep in range(opt.episodes):
        obs, done = env.reset(), False
        while not done:
            action, _ = policy(obs)
            obs, reward, done, _ = env.step(action)
            ep_returns[ep] += reward
        print(f"{ep:02d})  Gt: {ep_returns[ep]:7.1f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("game", type=str, help="game name")
    parser.add_argument("path", type=str, help="path to the model")
    parser.add_argument(
        "-e", "--episodes", default=10, type=int, help="number of episodes"
    )
    parser.add_argument(
        "-v",
        "--variations",
        action="store_true",
        help="set mode and difficulty, interactively",
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="record png screens and sound",
    )
    main(parser.parse_args())
