#!/bin/bash
### Job name
#SBATCH -J Train_Abstract2_LSTM_CNN6_and_SNN8
#SBATCH -o /home/hice1/khom9/"CSE 8803 BMI Final Project"/TrainingLogs/output_cnn6_snn8.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=khom9@gatech.edu
### Queue name
### Specify the number of nodes and thread (ppn) for your job.
#SBATCH -N1 --ntasks=8
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:H100:1
### Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
#SBATCH -t 07:00:00
#################################

module load pytorch/2.1.0
python -u cnn2.py
python -u snn2.py
