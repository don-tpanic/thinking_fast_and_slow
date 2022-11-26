import os
import yaml
import numpy as np

from utils import load_config


def generate_configs(hparam_ranges, start_config_counter, template_config):  
    template = load_config(template_config)
    
    for hparam_name in hparam_ranges.keys():
        hparam_range = hparam_ranges[hparam_name]

        for hparam_val in hparam_range:
            config_version = f'config{start_config_counter}'
            template['config_version'] = config_version
            template[hparam_name] = hparam_val

            with open(f'configs/{config_version}.yaml', 'w') as f:
                yaml.dump(template, f)
            
            start_config_counter += 1


if __name__ == '__main__':

    hparam_ranges = {
        'lr_dnn': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0, 3, 5],
        'lr_clustering': [0.001, 0.01, 0.1, 0.5, 1, 2, 3],
    }
    
    generate_configs(
        hparam_ranges=hparam_ranges, 
        start_config_counter=6, 
        template_config='config3'
    )