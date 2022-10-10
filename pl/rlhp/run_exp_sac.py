"""
This is an example wrapper script that trains RLfHP on HalfCheetah-v3 and
then runs let's you call mutiple python scripts in sequence.
"""
from rlhp.main import main
import os

if __name__ == '__main__':
    # wandb logging params
    PROJECT_ID = "AntFromState"    # recommended: use environment name e.g.
                                  # HCFromPixel
    GROUP_ID   = "SACPolicy-t-2M-TC-10000"
                          # recommended: use str that is descriptive of
                          # reward mode training e.g.
                          # HighCapacityRMLowCapacityPPOPolicy

    # Compile any argparse args you want to pass like this
    rm_training_args = ['-p', PROJECT_ID,
                        '-g', GROUP_ID,
                        '-tei', 'AntUnhealthy1000-v0',
                        '-eei', 'AntUnhealthy1000-v0',
                        '-t', '2e6',
                        '-tc', '10000',
                        '-gc', '10',
                        '--save_all_reward_models',
                        '--skip_video',
                        '--rl_algo', 'sac'
                       ]

    ant_training_args_ppo = ['-tk', '0.01',
                         '--batch_size', '128',
                         '--gae_lambda', '0.9',
                         '--n_epochs', '20',
                         '--learning_rate', '3e-5',
                         '--clip_range', '0.4']
 
    assert(all(isinstance(a, str) for a in rm_training_args))

    # We want to make sure that we are training RM in this stage
    assert '--dont_use_reward_model' not in rm_training_args
    assert '--dont_train_reward_model' not in rm_training_args
    train_run_dir = main(rm_training_args)

    # check that reward_model directory is not empty
    rm_dir = os.path.join(train_run_dir, 'reward_models')
    all_reward_models = os.listdir(rm_dir)
    all_reward_models.sort(key= lambda x: x.split('.')[0])
    assert len(all_reward_models) > 0, "No reward model was saved"

    SEED = str(59)
    # Now we train a RL policy on trained reward model
    for i in range(len(all_reward_models)):
        EXP_NAME = "PPO-RM-"+(all_reward_models[i])+"-2M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '--seed', SEED,
                            '-tei', 'AntUnhealthy1000-v0',
                            '-eei', 'AntUnhealthy1000-v0',
                            '--dont_train_reward_model',
                            '--skip_video',
                            '--rl_algo', 'ppo',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '2e6',
                           ]
        rl_training_args += ant_training_args_ppo

        assert(all(isinstance(a, str) for a in rl_training_args))
        #test_run_dir = main(rl_training_args)

        EXP_NAME = "SAC-RM-"+(all_reward_models[i])+"-2M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '--seed', SEED,
                            '-tei', 'AntUnhealthy1000-v0',
                            '-eei', 'AntUnhealthy1000-v0',
                            '--dont_train_reward_model',
                            '--skip_video',
                            '--rl_algo', 'sac',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '2e6'
                           ]
        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

