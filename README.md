# Trained Atari Agents
[Download models](https://console.cloud.google.com/storage/browser/bitdefender_ml_artifacts) |
[Getting started](#how-to-use-it) |
[What's included](#whats-included) |
[Acknowledgements](#acknowledgements)

A source of :two::five:,:five::zero::zero: checkpoints (and growing) of curated DQN, M-DQN and C51 agents trained on modern and classic protocols, matching or besting the performance reported in the literature.

We release these models hoping it will help to advance the research in:

- Reproducing DRL results
- Imitation Learning
- Batch/Offline RL
- Multi-task learning


## What's included

Checkpoints for DQN, M-DQN and C51 agents across two or three training seeds, on `modern` or `classic` protocols.


### How many checkpoints?

An agent trained on 200M frames usually produces 200 checkpoints times the number of training seeds. In order not to make the download size overly large **we only include 51 checkpoints per training run**. These are sampled geometrically, with denser checkpoints towards the end of the training. This results in the last 20 checkpoints of the full 200 (last 10% of the training run) and then sparser checkpoints towards the beginning of the run, with only 10 out of 51 from the first half. It looks a bit like this:

![checkpoint sampling](/imgs/sampling.png)

Note it's not mandatory the best performing checkpoint is included since on some combinations of algorithms and agents the peak performance occurs earlier in training. However this sampling should characterize fairly well the performance of an agent most of the time.

 :exclamation::raised_hand: **If there is demand we can provide the full list of checkpoints for a given agent.**

Agents have been trained using PyTorch and the models are stored as compressed [state_dict](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html) pickle files. Since the networks used on ALE are fairly simple these could easily be converted for use in other deep learning frameworks.


### A word on training and evaluation protocols

There are two common training and evaluation protocols encountered in the literature. We will call them `classic` and `modern` across this project:

- `classic`: it originates from (Mnih, 2015)[^2] Nature paper and it mostly appears in DeepMind papers.
- `modern`: it originates from (Machado, 2017)[^1] and a variation of it was adopted by Dopamine[^6]. Since then it started to show more and more often.

The main two differences between the two are the way stochasticity is induced in the environment and how the loss of a life is treated.

We mention again that while we use Dopamine's protocol and sometimes hyperparameters, our agents are trained in PyTorch.


## Available agents

Check the table below for a summary.

| Algorithm          | Protocol  | Games | Seeds | Observations   |
| :----------------- | --------- | :---: | :---: | :------------- |
| **DQN**            | `modern`  |  60   |   3   | DQN agent using the settings from [dopamine](https://github.com/google/dopamine/blob/master/dopamine/jax/agents/dqn/configs/dqn.gin). It's optimised with Adam and uses MSE instead of Huber loss. **A surprisingly strong agent on this protocol**. |
| **M-DQN** | `modern`  |  60   |   3   | DQN above but using the **Munchausen trick**[^7]. Even stronger performance. |
| **C51**            | `classic` | 28/57 |   3   | Closely follows the original paper[^3]. |
| **DQN Adam**       | `classic` | 28/57 |   2   | A DQN agent trained according to the Rainbow paper[^4]. The exact settings and plots can be found in our paper[^5]. |

Right off-the bat you can notice that on the `classic` protocol there are only 28 games out of the usual 57. We trained the two agents on this protocol over one year ago using the now deprecated `atari-py` project which officially provided the ALE Python bindings in OpenAI's Gym. Unfortunately the package came with a large number of ROMs that are not supported by the current, official, [ale-py](https://github.com/mgbellemare/Arcade-Learning-Environment) library. The agents trained on the `modern` protocol (as well as the code we provide for visualising agents) all use the new `ale-py`. Therefore we decided against providing support for the older library event if it meant dropping half of the trained models. A great resource for reading about this issue is Jesse's Farebrother [ALE v0.7 release notes](https://brosa.ca/blog/ale-release-v0.7/#rom-management). Importantly, we found out about the issue while checking the performance of the trained models on the new `ale-py` back-end and we provide plots showing the remaining 28 agents perform as expected ([C51_classic](https://github.com/floringogianu/atari-agents/blob/main/imgs/c51_g28_confirmation.png), [DQN_classic](https://github.com/floringogianu/atari-agents/blob/main/imgs/dqn_g28_confirmation.png)).


## How to use it

### Installation

[:arrow_double_down: Download :arrow_double_down:](https://console.cloud.google.com/storage/browser/bitdefender_ml_artifacts) the saved models.

Using `gsutil` you can download all the models from the command line:

```shell
gsutil -m cp -R gs://bitdefender_ml_artifacts/atari ./
```

or select certain checkpoints like this:

```shell
gsutil -m cp -R gs://bitdefender_ml_artifacts/atari/[ALGORITHM]/[GAME]/[SEED]/model_50000000.gz ./
```

Install the `conda` environment using `conda env create -f environment.yml`. If this fails for some reason the main requirements are:

- `pytorch 1.11.0`
- `ale-py 0.7.4`
- `opencv 4.5.2`

An easy way to install `ale-py`, download and install the ROMs is to just install `gym`:

```shell
pip install 'gym [atari,accept-rom-license]'
```

If for some reason the `SDL` support is not just right, you might have better luck cloning [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) and installing from source using `pip install .`. Just make sure then to use register the ROM files again:

```
ale-import-roms path/to/roms
```

See [this excellent](https://brosa.ca/blog/ale-release-v0.7) post about what's new in `ALE 0.7` and how to install ROMs.


### Play using a saved model

Just do:

```shell
python play.py models/AGENT/GAME/SEED/model_STEP.gz
```
Passing the `-r/--record` flag will create a `./movies` folder and save the screens and audio.

We also support game modes and difficulty levels introduced by Machado, 2017[^1]. You can use `-v` to activate an interactive mode for selecting game modes and difficulty levels:

```shell
python play.py models/AGENT/GAME/SEED/model_STEP.gz -v
```

### Folder structure

There are some conventions encoded in the folder structure used by `play.py` to configure the model and the environment using the name of the directory containing the checkpoints. For example `DQN_modern` will configure a DQN network and evaluate it on the `modern` protocol while `C51_classic` will configure a C51-style network and evaluate it on the `classic` protocol.

You should end with something like this after downloading all the agents:

```shell
.
├── ale_env.py
├── human_play.py
├── play.py
├── README.md
├── models
│   ├── C51_classic
│       └── ...
│   ├── DQN_classic_adam
│       └── ...
│   └── DQN_modern
│       ├── AirRaid
│       │   ├── 0
│       │   ├── 1
│       │   └── 2
│      ...
│       └── Zaxxon
│           ├── 0
│           ├── 1
│           └── 2
```


## Just how well trained are these agents?

Our PyTorch implementation of DQN trained using Adam on the modern protocol compares favourable to the exact same agent trained using Dopamine. The plots below have been generated using the tools provided by [rliable](https://github.com/google-research/rliable).

![dopamine_vs_pytorch](/imgs/rliable_comparison.png)

Some more comparisons can be found [here](https://github.com/floringogianu/atari-agents/tree/main/imgs).

A detailed discussion about the performance of DQN + Adam and C51 trained on the `classic` protocol can be found in our paper[^5], where we used these checkpoints as baselines.


## Acknowledgements

- [Bitdefender](https://www.bitdefender.com/), for providing all the material resources that made possible this project and my colleagues in Bitdefender's [Machine Learning & Crypto Research Unit](https://bit-ml.github.io/) for all their support.
- [Kai Arulkumaran](https://github.com/Kaixhin/), for providing the `atari-py`/`ale-py` wrapper I used extensively in my research and who helped me many times figuring out some of the more arcane details of the various training and evaluation protocols in DRL.
- [Dopamine baselines and configs](https://github.com/google/dopamine), which I used extensively for comparing the performance of our implementations and for figuring various hyperparameters.


## Related projects

- [Stable Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) -- agents for seven Atari games.
- [Kai Arulkumaran](https://github.com/Kaixhin/) provides a number of ALE checkpoints together with his [Rainbow implementation](https://github.com/Kaixhin/Rainbow/releases).
- Uber Research [Atari Model Zoo](https://github.com/uber-research/atari-model-zoo) -- large number agents trained with Dopamine and OpenAI Baselines. However the [availability of these agents](https://github.com/uber-research/atari-model-zoo/issues/7) is not clear at the moment.


## Giving credit

If you use these checkpoints in your research and published work, please consider citing this project:

```
@misc{gogianu2022agents,
  title  = {Atari Agents},
  author = {Florin Gogianu and Tudor Berariu and Lucian Bușoniu and Elena Burceanu},
  year   = {2022},
  url    = {https://github.com/floringogianu/atari-agents},
}
```

[^1]: [Machado, et al. 2017, _Revisiting the Arcade Learning Environment..._](https://arxiv.org/abs/1709.06009)
[^2]: [Mnih, et al. 2015, _Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
[^3]: [Bellemare, et al. 2017, _A distributional perspective..._](http://proceedings.mlr.press/v70/bellemare17a.html)
[^4]: [Hessel, et al. 2017, _Combining Improvements in Deep RL_](https://arxiv.org/abs/1710.02298)
[^5]: [Gogianu, et al. 2021, _Spectral Normalisation..._](https://www.semanticscholar.org/paper/Spectral-Normalisation-for-Deep-Reinforcement-an-Gogianu-Berariu/cf04c05f69022f71b60c7b7252af94f11cad5ef1)
[^6]: [Castro, et al. 2018, _Dopamine: A Research Framework for Deep RL_](http://arxiv.org/abs/1812.06110)
[^7]: [Vieillard, et al. 2020, _Munchausen Reinforcement Learning_](https://arxiv.org/abs/2007.14430)

