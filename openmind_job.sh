#!/bin/bash
#SBATCH -J distributed_demo_job
#SBATCH -t 00:30:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH --gres gpu:2
#SBATCH --constraint=rocky8
#SBATCH --mem 30G
#SBATCH -o job.out

source ~/.bashrc
conda activate pytorch-3.10

srun python -u distributed_demo.py
