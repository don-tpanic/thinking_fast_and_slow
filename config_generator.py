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
                template['fast_config']['lr_nn'] = fast_param_key1
                template['fast_config']['attn_weighting'] = fast_param_key2
                template['slow_config']['lr_dnn'] = slow_param_key1
                config_version = f'config_dl_dnn_{config_counter}'
                template['config_version'] = f'{config_version}'
                config_counter += 1
                with open(f'configs/{config_version}.yaml', 'w') as f:
                    yaml.dump(template, f, sort_keys=False)
                print(
                    f'{config_version}', 
                    template['fast_config']['lr_nn'], 
                    template['fast_config']['attn_weighting'],
                    template['slow_config']['lr_dnn'],
                )


if __name__ == '__main__':
    hparam_streams = {
        'fast_config': \
            {
                'lr_nn': [0.0001, 0.0005, 0.001, 0.005, 0.01],
                'attn_weighting': [0.02, 0.05, 0.1, 0.5, 1.]
            }, 
        'slow_config': \
            {'lr_dnn': [0.1, 0.5, 1, 1.5, 3]}
    }
    
    generate_configs(
        hparam_streams=hparam_streams, 
        config_counter=2, 
        template_config='config_dl_dnn_1'
    )