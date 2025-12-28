#!/bin/bash
#SBATCH --job-name=qwen_inf_play
# SBATCH --array=0-1        
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
# SBATCH --nodelist=n-501 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000' 
#SBATCH --mem=16G             
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



export PROJECT_HOME=$(pwd)
export CACHE_BASE="$PROJECT_HOME/.cache"
mkdir -p "$CACHE_BASE"

# HuggingFace cache directories
export HF_HOME="$CACHE_BASE/huggingface"
export TRANSFORMERS_CACHE="$CACHE_BASE/huggingface/transformers"
export HF_DATASETS_CACHE="$CACHE_BASE/huggingface/datasets"

# vLLM settings - disable usage stats
export VLLM_USAGE_SOURCE="production"
export VLLM_DO_NOT_TRACK="1"

# FlashInfer workspace directory
export FLASHINFER_WORKSPACE_DIR="$CACHE_BASE/flashinfer"

. utils/base.sh
source .miniconda3/etc/profile.d/conda.sh
conda activate inter25_vllm


python play/check_abillity_to_recover.py

