import yaml
import os.path as osp






# ****************** helper  *******************

def log_config(config, model_dir):
    
    config_path = osp.join(model_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)