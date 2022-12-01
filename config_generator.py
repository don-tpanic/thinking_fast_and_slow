import os
import yaml
import numpy as np

from utils import load_config


def generate_configs(hparam_streams, config_counter, template_config):  
    template = load_config(template_config)

    slow_configs = hparam_streams['slow_config']
    fast_configs = hparam_streams['fast_config']
    slow_config_key = 'lr_dnn'
    fast_config_key = 'lr_clustering'

    for slow_param in slow_configs[slow_config_key]:
        for fast_param in fast_configs[fast_config_key]:
            config = template.copy()
            config['slow_config'][slow_config_key] = slow_param
            config['fast_config'][fast_config_key] = fast_param
            config_version = f'config{config_counter}'
            config['config_version'] = f'config{config_counter}'
            config_counter += 1
            with open(f'configs/{config_version}.yaml', 'w') as f:
                yaml.dump(template, f, sort_keys=False)
            print(
                f'{config_version}', 
                config['slow_config']['lr_dnn'], 
                config['fast_config']['lr_clustering']
            )


if __name__ == '__main__':
    hparam_streams = {
        'fast_config': \
            {'lr_clustering': [0.001, 0.01, 0.1, 1]},
        'slow_config': \
            {'lr_dnn': [0.01, 0.1, 0.5, 1, 1.5, 3]}
    }
    
    generate_configs(
        hparam_streams=hparam_streams, 
        config_counter=6, 
        template_config='config3'
    )