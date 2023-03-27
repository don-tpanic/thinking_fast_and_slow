import os
import yaml
import numpy as np

from utils import load_config


# def generate_configs(hparam_streams, config_counter, template_config):  
#     template = load_config(template_config)

#     slow_configs = hparam_streams['slow_config']
#     fast_configs = hparam_streams['fast_config']
#     slow_config_key = 'lr_dnn'
#     fast_config_key = 'lr_nn'

#     for slow_param in slow_configs[slow_config_key]:
#         for fast_param in fast_configs[fast_config_key]:
#             config = template.copy()
#             config['slow_config'][slow_config_key] = slow_param
#             config['fast_config'][fast_config_key] = fast_param
#             config_version = f'config_dl_dnn_{config_counter}'
#             config['config_version'] = f'{config_version}'
#             config_counter += 1
#             with open(f'configs/{config_version}.yaml', 'w') as f:
#                 yaml.dump(template, f, sort_keys=False)
#             print(
#                 f'{config_version}', 
#                 config['slow_config'][slow_config_key], 
#                 config['fast_config'][fast_config_key]
#             )

def generate_configs(hparam_streams, config_counter, template_config):  
    template = load_config(template_config)

    slow_configs = hparam_streams['slow_config']
    fast_configs = hparam_streams['fast_config']

    for fast_param_key1 in fast_configs['lr_nn']:
        for fast_param_key2 in fast_configs['attn_weighting']:
            for slow_param_key1 in slow_configs['lr_dnn']:
                for slow_param_key2 in slow_configs['n_units']:
                    template['fast_config']['lr_nn'] = fast_param_key1
                    template['fast_config']['attn_weighting'] = fast_param_key2
                    template['slow_config']['lr_dnn'] = slow_param_key1
                    template['slow_config']['n_units'] = slow_param_key2
                    config_version = f'config_dlMU_dnn_{config_counter}'
                    template['config_version'] = f'{config_version}'
                    config_counter += 1
                    with open(f'configs/{config_version}.yaml', 'w') as f:
                        yaml.dump(template, f, sort_keys=False)
                    print(
                        f'{config_version}', 
                        template['fast_config']['lr_nn'], 
                        template['fast_config']['attn_weighting'],
                        template['slow_config']['lr_dnn'],
                        template['slow_config']['n_units']
                    )


if __name__ == '__main__':
    hparam_streams = {
        'fast_config': \
            {
                'lr_nn': [0.001],
                'attn_weighting': [0.05]
            }, 
        'slow_config': \
            {
                'lr_dnn': [0.1],
                'n_units': [16, 32, 64, 128, 256, 512, 1024]
            }
    }
    
    generate_configs(
        hparam_streams=hparam_streams, 
        config_counter=127,  # should be the larget existing config+1. 
        template_config='config_dlMU_dnn_1'
    )