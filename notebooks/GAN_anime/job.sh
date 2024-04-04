#!/bin/bash -l


# Submit job with the command: qsub job.qsub
# Request 2 CPUs
#$ -pe omp 4

# Request 1 GPU
#$ -l gpus=1
# force cuda capacity 7.0 (basically, ask for a relatively new gpu)
#$ -l gpu_c=7.0
# currently off (previous line already does this), but would  force gpu memory > 16GB
##$ -l gpu_memory=40G

# Set the runtime limit (default 12 hours):
#$ -l h_rt=40:59:00

#$ -P ivc-ml


# Send email when the job is done (default: no email is sent)
#$ -m e

# Give the job a name (default: script name)
#$ -N dl523_training

# Combine output and error streams
#$ -j y

conda activate conda_env


# Load the desired version of Python.
cd /projectnb/dl523/students/ellywang/EC523_Project_G/notebooks/StyleGAN-anime/
python3 train_loss_gan.py