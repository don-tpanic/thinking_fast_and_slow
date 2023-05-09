# Thinking Fast and Slow - A novel multi-region computational-modelling framework for conceptual knowledge learning and consolidation (WIP)

### Warning: this code base is part of an active research project which is constantly evolving and subject to refactoring.

### Description
Humans can learn new concepts remarkably quickly. One long-standing question is how conceptual knowledge is learnt and consolidated in the brain. Prior research focused on either early learning, associated with the hippocampus, or semantic knowledge after consolidation, associated with the anterior temporal lobe (ATL), and therefore have not assessed how regions interact during the concept formation process. This project will track concept formation by recording behaviour and brain activity over time and apply a novel multi-region computational-modelling approach to evaluate how the hippocampus and ATL interact to support learning and consolidation of conceptual knowledge.

### Currently working on branch
```
new-stimuli-fit
```

### Structure and main components
The code base is structured on two levels: dual-stream level (top) and single-stream level (module). Both levels follow roughly the same organisation where the main components typically are
```
    .
    ├── main.py                 # Main execution script
    ├── models.py               # Model definitions
    ├── layers.py               # Layer definitions
    ├── train.py                # Training logic called by main.py
    ├── evaluations.py          # Analyses
    ├── utils.py                # Utils including configuration and data loading etc
    ├── config_generator.py     # Script for generating configuration files in batch
    ├── configs/                # Model configurations (variants)
    ├── results/                # Experiment results (grouped by configs)
```
1. Dual-stream level
  * `clustering_model`, ``dnn_model``, `multiunit_clustering`, `dl_model` are single stream module definitions that the dual stream framework imports. For example, a specific dual-stream model can be the integration of a fast stream module from `multiunit_clustering` and a slow stream module from `dnn_model`. The dual-stream model is assembled in top-level `models.py`, 
  ```python
from clustering_model.models import ClusteringModel 
from dnn_model.models import NeuralNetwork
from multiunit_clustering.models import MultiUnitCluster
from dl_model.models import DistrLearner, DistrLearnerMU


class FastSlow(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # initializing models
        if config['fast'] == 'clustering':
            self.FastModel = ClusteringModel(config=config['fast_config'])

        elif config['fast'] == 'multiunit_clustering':
            self.FastModel = MultiUnitCluster(config=config['fast_config'])

        elif config['fast'] == 'distr_learner':
            self.FastModel = DistrLearner(config=config['fast_config'])
        
        elif config['fast'] == 'distr_learner_mu':
            self.FastModel = DistrLearnerMU(config=config['fast_config'])

        if config['slow'] == 'dnn':
            self.SlowModel = NeuralNetwork(config=config['slow_config'])
  ```
  
2. Single-stream level
  * Each single stream module, is designed to be in itself a standalone code base which can be taken out of the dual-stream framework and be fit to learning problems as single-stream models. All layer and model definitions are created at this level. 

### Environment setup
```
conda env create -f environment.yml
```

### Example usage (under active development)
1. Running the dual-stream model with different configurations (hyper-parameter search)
```
python main.py --logging True --config None --begin 0 --end 999
```
Configs 0-999, each will be run on a single process with model weights and other analytical results saved both locally and synced on weights & biases.

### About experiments logging
Currently trained model weights are saved only locally. Analytical results such as learning curves are saved both locally and on weights & biases. One can disable w&b syncing by setting the `--logging` flag to `False`. To modify logging location on w&b, one should look into
```python
# main.py
if args.config:
    config_version = args.config
    config = load_config(config_version)
    if args.logging:
        wandb.init(
            project="<project-name>",
            entity="<project-entity>",
            config=config,
            reinit=True,
        )
        wandb.run.name = f'{config_version}'
```
