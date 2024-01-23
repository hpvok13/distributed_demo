#!/bin/bash
#SBATCH -J dist_job
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -o dist_job.out

source /etc/profile
module load anaconda/Python-ML-2023b

python distributed_demo.py
