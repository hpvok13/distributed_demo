#!/bin/bash
#SBATCH -J dist_demo_job
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --constraint=rocky8
#SBATCH --mem 10G
#SBATCH -o job.out

source ~/.bashrc
conda activate pytorch-3.10

python distributed_demo.py
