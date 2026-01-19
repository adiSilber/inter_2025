#!/bin/bash
#SBATCH --job-name=qwen_inf_play
# SBATCH --array=0-1        
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
# SBATCH --nodelist=n-805 
#SBATCH --gpus=1 --constraint='l40s|a6000' 
#SBATCH --mem=32G
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



export PROJECT_HOME=$(pwd)
export CACHE_BASE="$PROJECT_HOME/.cache"
mkdir -p "$CACHE_BASE"
export HF_HOME="$CACHE_BASE/huggingface"
export TRANSFORMERS_CACHE="$CACHE_BASE/huggingface/transformers"
export HF_DATASETS_CACHE="$CACHE_BASE/huggingface/datasets"
export VLLM_USAGE_SOURCE="production"
export VLLM_DO_NOT_TRACK="1"
export FLASHINFER_WORKSPACE_DIR="$CACHE_BASE/flashinfer"
export PYTHONPATH=$PROJECT_HOME:$PYTHONPATH

. utils/base.sh
set +u
source /specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/.miniconda3/bin/activate inter2025_vllm


python -u play/main.py

