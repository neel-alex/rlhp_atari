"""
This is an example wrapper script that trains RLfHP on HalfCheetah-v3 and
then runs let's you call mutiple python scripts in sequence.
"""
from rlhp.main import main
import os

if __name__ == '__main__':
    # wandb logging params
    PROJECT_ID = "HCWithPos1000"    # recommended: use environment name e.g.
                                  # HCFromPixel
    GROUP_ID   = "PPOPolicy-t-2M-TC-10000"
                          # recommended: use str that is descriptive of
                          # reward mode training e.g.
                          # HighCapacityRMLowCapacityPPOPolicy

    # Compile any argparse args you want to pass like this
    rm_training_args = ['-p', PROJECT_ID,
                        '-g', GROUP_ID,
                        '-tei', 'HCWithPos1000-v0',
                        '-eei', 'HCWithPos1000-v0',
                        '-t', '2e6',
                        '-tc', '10000',
                        '-gc', '5',
                        '--save_all_reward_models',
                        '--skip_video',
                        '--rl_algo', 'ppo',
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

    SEED = str(23)
    # Now we train a RL policy on trained reward model
    for i in range(len(all_reward_models)):
        for env in ['HCWithPos2000-v0',
                    'HCWithPos1000-v0',
                    'HCWithPos500-v0',
                    'HCWithPos250-v0']:
            EXP_NAME = "PPO-RM-"+(all_reward_models[i])+"-"+env+"-2M"
            rl_training_args = ['-p', PROJECT_ID,
                                '-g', GROUP_ID,
                                '-n', EXP_NAME,
                                '--seed', SEED,
                                '-tei', env,
                                '-eei', env,
                                '--dont_train_reward_model',
                                '--skip_video',
                                '--rl_algo', 'ppo',
                                '--reward_model_path', os.path.join(rm_dir,
                                                       all_reward_models[i]),
                                '-t', '2e6',
                               ]

        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

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
        #test_run_dir = main(rl_training_args)

