import yaml
import argparse

from src.nn.archs.train_fn import *






# ********************* launch training ***********************
# cmd to launch : python -W ignore -m src.nn.train --config ./src/nn/config/config.yml > ./src/nn/results/results_1.txt
# cmd to visualize : tensorboard --logdir=./src/nn/weights/fine_tuned_m_r_cnn/log_dir/ --port=8013

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'nn training')
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)