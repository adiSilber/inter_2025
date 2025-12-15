#!/bin/bash
#SBATCH --job-name=qwen_inf
# SBATCH --array=0-1        
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
# SBATCH --nodelist=n-501 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000' 
#SBATCH --mem=16G             
#SBATCH --output=/specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/logs/qwen-infr_play_%j.out
#SBATCH --error=/specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

. /specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/scripts/workdir.sh
. scripts/base.sh
. scripts/activate_env.sh

python play/forward_ds_qwen_7b.py