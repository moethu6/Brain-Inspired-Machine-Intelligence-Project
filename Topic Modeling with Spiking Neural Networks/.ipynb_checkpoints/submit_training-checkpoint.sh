#!/bin/bash
# ## Job name
# SBATCH -J Train_Abstract_LSTM_SNN
# SBATCH -o /home/hice1/khom9/"CSE 8803 BMI Final Project"/output.out
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=khom9@gatech.edu
# ## Queue name
# ## Specify the number of nodes and thread (ppn) for your job.
# SBATCH -N1 --ntasks=8
# SBATCH --mem-per-cpu=3G
# SBATCH --gres=gpu:1
# ## Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
# SBATCH -t 03:00:00
# ################################

module load pytorch/2.1.0
python -u cnn.py
