#!/bin/bash
#SBATCH --job-name=installcudaoncomputendoes
#SBATCH --time=00:10:00
# SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
# SBATCH --gpus=1 --constraint='l40s|a6000|a5000' 
#SBATCH --mem=4G             
#SBATCH --output=logs/installcudaoncomputendoes_%j.out
#SBATCH --error=logs/installcudaoncomputendoes_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



. utils/base.sh
source .miniconda3/etc/profile.d/conda.sh
conda activate inter25_vllm


