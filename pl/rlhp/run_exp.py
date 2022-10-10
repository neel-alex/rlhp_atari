"""
This is an example wrapper script that trains RLfHP on HalfCheetah-v3 and
then runs let's you call mutiple python scripts in sequence.
"""
from rlhp.main import main
import os

if __name__ == '__main__':
    # wandb logging params
    PROJECT_ID = "HCFromState"    # recommended: use environment name e.g.
                                  # HCFromPixel
    GROUP_ID   = "PPOPolicy-t-2M-TC-10000"
                          # recommended: use str that is descriptive of
                          # reward mode training e.g.
                          # HighCapacityRMLowCapacityPPOPolicy

    # Compile any argparse args you want to pass like this
    rm_training_args = ['-p', PROJECT_ID,
                        '-g', GROUP_ID,
                        '-t', '2000000',
                        '-tc', '10000',
                        '-gc', '10',
                        '--save_all_reward_models',
                        '--rl_algo', 'ppo'
                       ]
    
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

    # Now we train a RL policy on trained reward model
    for i in range(len(all_reward_models)):
        EXP_NAME = "PPO-RM-"+(all_reward_models[i])+"-5M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '-t', '5000000'
                            '--dont_train_reward_model',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '1e6'
                           ]
        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

        EXP_NAME = "SAC-RM-"+(all_reward_models[i])+"-5M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '-t', '5000000'
                            '--rl_algo', 'sac',
                            '--disable_all_saving',
                            '--dont_train_reward_model',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '1e6'
                           ]
        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

