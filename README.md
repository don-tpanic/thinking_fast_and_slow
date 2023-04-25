# Thinking Fast and Slow - A novel multi-region computational-modelling framework for conceptual knowledge learning and consolidation (WIP)

### Warning: this code base is part of an active research project which is constantly evolving and subject to refactorying.

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
  * `clustering_model`, `multiunit_clustering`, `dnn_model`, `dl_model` are single stream module definitions that the dual stream framework calls. For example, a specific dual-stream model can be the integration of a fast stream module from `multiunit_clustering` and a slow stream module from `dnn_model`. The dual-stream model is assembled in top-level `models.py`.
  
2. Single-stream level
  * Each single stream module, is itself a standalone code base which can individually fit the learning problems.

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
