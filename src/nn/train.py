import yaml
import argparse

from src.nn.archs.train_fn import *






# ********************* launch training ***********************

# cmd to launch : python -W ignore -m src.nn.train --config ./src/nn/config/config_endovis.yml > ./src/nn/results/results_1.txt
# cmd to tune : python -W ignore -m src.nn.train --config ./src/nn/config/config_endovis.yml -b 1 -o "adam" -l 0.01 -n "model_name"
# cmd to launch loop : ./src/nn/launch_train.sh
# cmd to visualize : tensorboard --logdir=./src/nn/weights/fine_tuned_m_r_cnn_0/log_dir/ --port=8013
#                    tensorboard --logdir=/data/user/DATA_SSD1/__adri/weights/fine_tuned_m_r_cnn_0_1/log_dir/ --port=8013

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'nn training')
    parser.add_argument('-c', '--config', type=str, required=True, help='path to yaml config')

    # **** tunable parameters in bash file *****
    parser.add_argument('-b', '--batch_size', type=int, required=False)
    parser.add_argument('-l', '--lr', type=float, required=False)
    parser.add_argument('-n', '--model_name', type=str, required=False)
    parser.add_argument('-o', '--optimizer', type=str, required=False)

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.model_name is not None:
        config['model_name'] = args.model_name
    if args.optimizer is not None:
        config['optimizer'] = args.optimizer

    main(config)