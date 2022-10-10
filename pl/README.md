## How to use this code?
run `python rlhp/main.py` with appropriate arguments. See argparse help and comments in `main.py` to understand possible arguments you can pass. You can pass these arguments through either a (1) config file (2) bash command using argparse.

You will require to install following packages in addition to mujoco_py to run the code in this repo.
`pip install numpy scipy torch matplotlib pandas tqdm gym wandb nvidia-ml-py3`


## Known Bugs
- RLfHP might not work correctly in the case `init_comparisions_pct == 0`. However, I don't think it makes sense to have `init_comparisons_pct = 0`, so, its unlikely I will work on a fix for this.
- ~~reward normalization does not work correctly when using SAC+RM. This is because if the train_env is passed normalize_reward, then the ReplayBuffer is configured to normalize reward using train_env's reward mean, std when SAC samples from the buffer. I will try and fix this asap.~~ (This has now been fixed)

## Feature Requests/ToDos
- Run a meta script which runs learning from reward model + training a RL policy on that frozen reward model and both runs get logged to the same wandb group.
- Add SAC's parameters in argparse.
- Add label annealing.
- Add learning rate annealing for reward model.
- Put in error messages and warnings where needed.

## FAQ
1. Why is there a clone of SB3 in this repo?
This clone comes from one of my previous projects. This has some minor but useful customizations e.g. PPO uses a customized RolloutBuffer implementation which also stores next observations.
