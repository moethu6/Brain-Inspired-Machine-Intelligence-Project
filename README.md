# Folder Description

## Repository Root

- `Readme.md`  
  Project overview and onboarding notes.
- `FolderDescription.md`  
  Detailed file/folder descriptions for the repository.
- `Topic Modeling with Spiking Neural Networks/`  
  Main research workspace containing code, notebooks, datasets, models, logs, and embeddings.

## `Topic Modeling with Spiking Neural Networks/`

- `dataset.py`  
  Defines `AbstractDataset`, including CSV loading, token/label preprocessing, and multi-label target encoding.
- `snn_util.py`  
  Spiking model definitions (`AbstractSNN_1`, `AbstractSNN_2`, `AbstractHybrid`) and SNN-specific training/loss utilities.
- `cnn_util.py`  
  CNN baseline architectures and training utility functions.
- `*.ipynb`  
  Notebook-based experiments for training, evaluation, and comparison workflows.
- `CleanedAVdata.csv`  
  Preprocessed dataset used by the experiments.
- `TrainingLogs/`  
  Training output logs and run records.
- `models/`  
  Active/current model checkpoints.
- `models_OLD/`  
  Older or archived model checkpoints.
- `wordvectors/`  
  Current word vector files used for embedding initialization.
- `wordvectors_OLD/`  
  Older or archived word vector assets.
- `.ipynb_checkpoints/`  
  Auto-generated notebook checkpoint files.


