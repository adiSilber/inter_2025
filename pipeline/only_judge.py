#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' 
#SBATCH --mem=32G
# SBATCH --nodelist=n-801
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



from math import ceil
import hashlib
import os
import sys
from time import time





sys.stderr = sys.stdout
sys.stdout.reconfigure(line_buffering=True) #type: ignore
PROJECT_HOME = os.getcwd()
CACHE_BASE = os.path.join(PROJECT_HOME, ".cache")
os.makedirs(CACHE_BASE, exist_ok=True)

os.environ["PROJECT_HOME"] = PROJECT_HOME
os.environ["CACHE_BASE"] = CACHE_BASE
os.environ["HF_HOME"] = os.path.join(CACHE_BASE, "huggingface")
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "datasets")
os.environ["VLLM_USAGE_SOURCE"] = "production"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["FLASHINFER_WORKSPACE_DIR"] = os.path.join(CACHE_BASE, "flashinfer")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONPATH"] = PROJECT_HOME + ":" + os.environ.get("PYTHONPATH", "")

# Add PROJECT_HOME to Python path so 'play' module can be imported
sys.path.insert(0, PROJECT_HOME)
sys.path.insert(0, os.path.join(PROJECT_HOME, "pipeline"))


import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


import torch
from typing import Any, Optional, cast, cast
from pipeline.interface import (
    Experiment, JudgeGenerationConfig, 
    SamplingParams
)
from pipeline.judge_correctness import JudgeDecision, CorrectnessJudge
from pipeline.prompts import ShortAnswerPromptTemplate, MathEvaluatorJudgePrompt, SimpleEvaluatorJudgePrompt, DiffJudgePrompt, SunOrNotSunBetterJudgePrompt



judge_model_path = "/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"

RUN_ID = "1772053577"

ARTIFACTS_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts/"
EXPERIMENT_FILE = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith(RUN_ID) and "experiment" in f][0]
NO_ACTIVATIONS_DATAPOINTS_FILE = [f for f in os.listdir(ARTIFACTS_DIR) if f.startswith(RUN_ID) and "datapoints_NO_ACTIVATIONS" in f][0]


experiment = Experiment.load(ARTIFACTS_DIR + EXPERIMENT_FILE)
experiment.load_datapoints(ARTIFACTS_DIR + NO_ACTIVATIONS_DATAPOINTS_FILE)

judge_config = JudgeGenerationConfig(
    judge_name="qwen-7B-judge",
    judge_model_path=judge_model_path,
    judge_prompt=SunOrNotSunBetterJudgePrompt(),

    sampling_params=SamplingParams(
        temperature=0.0,
        top_k=None,
        top_p=None,
        take_dumb_max=True,
        max_new_tokens=1024,
    ),
    dtype=torch.bfloat16
)

experiment.unique_id = str(experiment.unique_id) + "_rejudged_at_" + str(int(time()))
experiment.judge_generation_config = judge_config

print("unique_id", experiment.unique_id)

def run_judge():
    print("=" * 80)
    print(f"Start Judging {experiment.name}")
    print("=" * 80)

    # The model-based judge had accuracy issues with semantic matching
    print("\n3. Running judge validation...")
    judge = CorrectnessJudge(experiment, device='cuda')
    judge.run(batch_size=8)
    judge.unload_model()
    
    print("\n4. Done generating judge decisions. datapoints populated")
    
    # Compute and log judge decision statistics
    total_judged = len(experiment.datapoints)
    
    # Count each decision type
    decision_counts = {}
    for decision in JudgeDecision:
        decision_counts[decision] = sum(1 for dp in experiment.datapoints if dp.judge_decision == decision)
    
    
    # Update wandb with all decision statistics
    wandb_summary :dict[str,Any]= {
        "judge_total": total_judged,
        "num_datapoints_used": total_judged,
    }
    for decision, count in decision_counts.items():
        wandb_summary[f"judge_{decision.value}_count"] = count
        wandb_summary[f"judge_{decision.value}_ratio"] = count / total_judged if total_judged else 0.0
    
    print(f"   Judge decision breakdown:")
    for decision, count in decision_counts.items():
        ratio = count / total_judged if total_judged else 0.0
        print(f"      {decision.value}: {count}/{total_judged} ({ratio:.4f})")

    # Save a lightweight version of all datapoints (no activations) for quick loading
    print("  Saving lightweight datapoints (no activations) for quick analysis...")
    experiment.store_datapoints_without_activations(ARTIFACTS_DIR)

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_judge()
