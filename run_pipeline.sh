#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=ReplicableTL
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=cpu
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python analyze_only.py