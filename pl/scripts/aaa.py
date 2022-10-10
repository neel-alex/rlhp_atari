"""
This is an example wrapper script that trains RLfHP on HalfCheetah-v3 and
then runs let's you call mutiple python scripts in sequence.
"""
from rlhp.main import main as main
from rlhp.offline_rlfhp import main as offline_rlfhp
import os

if __name__ == '__main__':
    # wandb logging params
    PROJECT_ID = "HCOffline2"    # recommended: use environment name e.g.
                                  # HCFromPixel
    GROUP_ID   = "SAC1"
                          # recommended: use str that is descriptive of
                          # reward mode training e.g.
                          # HighCapacityRMLowCapacityPPOPolicy

    # Compile any argparse args you want to pass like this
    rm_training_args = ['-p', PROJECT_ID,
                        '-g', GROUP_ID,
                        '-tc', '20000',
                        '--use_data_from', 'sac',
                        '--rm_epochs', '100',
                        '--save_every', '20',
                        '--eval_every', '5',
                        '--plot_every', '20',
                        #'--reward_net_layers', ''
                        '-dd', 'data/HC',
                        '--rm_learning_rate', '3e-3',
                       ]
    
    assert(all(isinstance(a, str) for a in rm_training_args))

    # We want to make sure that we are training RM in this stage
    assert '--dont_use_reward_model' not in rm_training_args
    assert '--dont_train_reward_model' not in rm_training_args
    train_run_dir = offline_rlfhp(rm_training_args)

    # check that reward_model directory is not empty
    rm_dir = os.path.join(train_run_dir, 'reward_models')
    all_reward_models = os.listdir(rm_dir)
    all_reward_models.sort(key= lambda x: x.split('.')[0])
    assert len(all_reward_models) > 0, "No reward model was saved"

    # Now we train a RL policy on trained reward model
    for i in range(len(all_reward_models)):
        EXP_NAME = "PPO-RM-"+(all_reward_models[i])+"-2M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '--dont_train_reward_model',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '2e6',
                            '--skip_video',
                           ]
        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

        EXP_NAME = "SAC-RM-"+(all_reward_models[i])+"-2M"
        rl_training_args = ['-p', PROJECT_ID,
                            '-g', GROUP_ID,
                            '-n', EXP_NAME,
                            '--dont_train_reward_model',
                            '--reward_model_path', os.path.join(rm_dir,
                                                   all_reward_models[i]),
                            '-t', '2e6',
                            '--skip_video',
                            '--rl_algo', 'sac'
                           ]
        assert(all(isinstance(a, str) for a in rl_training_args))
        test_run_dir = main(rl_training_args)

