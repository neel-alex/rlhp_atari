import argparse
import subprocess
import itertools


def main(args):
    PROJECT_ID = "HCFromPixelYYYY"    # recommended: use environment name e.g.
                                  # HCFromPixel

    # Compile any argparse args you want to pass like this
    base_training_args = ['-p', PROJECT_ID,
                          '-t', '5e6',
                          '--save_all_reward_models',
                          '--use_pixel_observation',
                          '--reward_net_type', 'borja_cnn',
                          '-rcfd', '64',
                          '--reward_scheme', 's',
                          '--device', 'cuda',
                          '--skip_video',
                         ]

    args = [
                ['-g', "PPOPolicy-BorjaCNN-t-5M-TC-25000-gc-20-IC-40",
                 '-tc', '25000',
                 '-icpct', '0.4',
                 '-gc', '20',
                 '--rm_epochs', '20',
                 ],

                 ['-g', "PPOPolicy-BorjaCNN-t-5M-TC-25000-gc-20-IC-60",
                 '-tc', '25000',
                 '-icpct', '0.6',
                 '-gc', '20',
                 '--rm_epochs', '20',
                 ],

                 ['-g', "PPOPolicy-BorjaCNN-t-5M-TC-25000-gc-20-IC-10",
                 '-tc', '25000',
                 '-icpct', '0.1',
                 '-gc', '20',
                 '--rm_epochs', '20',
                 ],
                 
        #         ['-g', "PPOPolicy-BorjaCNN-t-5M-TC-25000-gc-20-IC-20",
        #         '-tc', '25000',
        #         '-icpct', '0.2',
        #         '-gc', '20',
        #         '--rm_epochs', '40',
        #         ]
            ]
    
    cpu_list = ['4-7', '8-11', '12-13']

    for i in range(3):
        python_command = f"CUDA_VISIBLE_DEVICES={i+1} taskset --cpu-list {cpu_list[i]} python rlhp/main.py " +\
                (" ").join(base_training_args + args[i])
        tmux_launch = f"tmux new-session -d \\; " \
                      f"send-keys \"source ~/.zshrc; rlhp;" \
                      f"export PYTHONPATH=\".\";" \
                      f"sleep {50*i}; {python_command}\" C-m"
        subprocess.Popen([tmux_launch], shell=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_run', action='store_true', help='only launch a single job')
    parser.add_argument('--gpu', action='store_true', help='use the GPU script')
    parser.add_argument('--srun', action='store_true', help='use srun instead of sbatch')
    parser.add_argument('--print_config', action='store_true', help='add print_config to the imitation command')

    input_args = parser.parse_args()
    main(input_args)
