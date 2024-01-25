#!/bin/bash
#SBATCH -J distributed_demo_job
#SBATCH -t 00:30:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH --gres gpu:volta:2
#SBATCH --reservation iap
#SBATCH --mem 20G
#SBATCH -o job.out

# Load the necessary modules
source /etc/profile
module load anaconda/Python-ML-2023b
module load /home/gridsan/groups/datasets/ImageNet/modulefile 

# Run the distributed demo (-u for unbuffered output)
srun python -u distributed_demo.py
