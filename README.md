# Brain-Inspired Machine Intelligence Project

This repository contains experiments on **topic modeling with spiking neural networks (SNNs)** and CNN baselines for patent abstract classification.

## Project Structure

- `Topic Modeling with Spiking Neural Networks/`
  - `dataset.py`: PyTorch dataset utilities for loading abstracts and multi-label targets.
  - `snn_util.py`: SNN model definitions and related training utilities.
  - `cnn_util.py`: CNN baseline model definitions and training helpers.
  - `*.ipynb`: experiment notebooks for model development and evaluation.
  - `models/`, `models_OLD/`: saved model artifacts.
  - `wordvectors/`, `wordvectors_OLD/`: word vector assets used by the models.

## Notes

- This codebase appears research-oriented and notebook-driven.
- Model checkpoints and vector files are included in the repository and can be large.
- If you are getting started, begin by opening the notebooks in the `Topic Modeling with Spiking Neural Networks/` directory.
