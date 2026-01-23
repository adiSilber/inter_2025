#!/bin/bash
#SBATCH --job-name=lm_eval_exp
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' # 
#SBATCH --mem=32G
#SBATCH --output=logs/lm_eval_exp_%j.out
#SBATCH --error=logs/lm_eval_exp_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

lm-eval run --model hf --log_samples --model_args pretrained="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B" --tasks r1_math500  --device cuda:0 --output_path /home/ADV_2526a/evyataroren/inter_2025/artifacts/res_lm_eval_%j.json
