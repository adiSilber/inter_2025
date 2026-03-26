#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
# #SBATCH --array=0
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

JOB_ID = 0
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if 'SLURM_ARRAY_TASK_MIN' in os.environ:
        JOB_ID = JOB_ID - int(os.environ.get('SLURM_ARRAY_TASK_MIN', 0))

NUM_JOBS_ENV = os.environ.get('SLURM_ARRAY_TASK_COUNT') or os.environ.get('NUM_JOBS')
if NUM_JOBS_ENV is not None:
    NUM_JOBS = int(NUM_JOBS_ENV)
elif 'SLURM_ARRAY_TASK_MAX' in os.environ:
    NUM_JOBS = int(os.environ['SLURM_ARRAY_TASK_MAX']) + 1
else:
    NUM_JOBS = 1

import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)


import torch
from typing import Any, Optional, cast, cast
from pipeline.interface import (
    ActivationCapturerV2, Experiment,  ModelGenerationConfig, JudgeGenerationConfig,
    SamplingParams
)
from pipeline.dataset_loaders import aggregated_dataset_loader, aggregate_shuffle_strategy, SimpleDatasetLoader, MATH500Loader, OneSolutionSimpleLoader
from pipeline.judge_correctness import JudgeDecision, CorrectnessJudge
from pipeline.hooks import AttentionHeadClipper, AttentionMapCapturerClipActivationsV2
from pipeline.injections import (
    SunWeightRedirectInjection,
)

from pipeline.stops import SentenceEndStopCondition, EOSTokenStopCondition, ImmediateStopCondition
from pipeline.wandb_utils import experiment_config_for_wandb
import wandb
from pipeline.generate_batched import GenerateBatched
from pipeline.prompts import ShortAnswerPromptTemplate, MathEvaluatorJudgePrompt, SimpleEvaluatorJudgePrompt, DiffJudgePrompt, SunOrNotSunBetterJudgePrompt



output_dir = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
model_path = "/home/ADV_2526a/evyataroren/inter_2025/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
judge_model_path = "/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
dataset_path = "/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets"






model_config = ModelGenerationConfig(
    model_name="DS/Qwen-1.5",
    model_path=model_path,
    should_stop=ImmediateStopCondition(),
    get_injection=SunWeightRedirectInjection(),
    global_stop=EOSTokenStopCondition(),  # Allow generation after injection
    question_prompt_template=ShortAnswerPromptTemplate(),
    sampling_params=SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        take_dumb_max=False,
        max_new_tokens=1024
    ),
    dtype=torch.bfloat16
)


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

dataset = aggregated_dataset_loader(
    datasets=[OneSolutionSimpleLoader],  # Simple questions with single definitive answers
    seed=42,
    strategy=aggregate_shuffle_strategy.SEQUENTIAL,
    base_path=dataset_path
)

def run_generation():
    
    # Load model ONCE
    generator = None
    all_experiments = []

    for layer_idx in [4,22]:
        experiment = Experiment(
            name=f"clip-layer-{layer_idx}-1.5B-batched",
            runner_name="evya",
            dataset=dataset,
            model_generation_config=model_config,
            judge_generation_config=judge_config,
            seed=42,
            unique_id=os.environ.get("SLURM_ARRAY_JOB_ID", str(int(time()))) + f"-L{layer_idx}",
            clip_max_val=0,
            activation_head_clipping={layer_idx: list(range(12))},
            activation_capturer=AttentionMapCapturerClipActivationsV2()
        )
        all_experiments.append(experiment)
        
        if generator is None:
            generator = GenerateBatched(experiment, device='cuda')
        else:
            generator.experiment = experiment

        print("unique_id", experiment.unique_id)

        print(f"   Populating datapoints from dataset...")
        experiment.populate_datapoints(num=50)
        # experiment.datapoints = experiment.datapoints[:50]
        for dp in experiment.datapoints:
            dp.should_capture_activations = True

        print(f"   Loaded {len(experiment.datapoints)} datapoints from dataset.")

        print("=" * 80)
        print(f"Starting Generation {experiment.name}")
        print("=" * 80)

        run = wandb.init(
            entity="inter_2026",
            project="Aha-moments",
            name=experiment.name,
            config=experiment_config_for_wandb(experiment, output_dir),
            reinit=True 
        )
        if run is not None:
            experiment.wandb_run_id = run.id


        print("WANDB id for this run:", experiment.wandb_run_id)

        # Time how long it takes to store experiment metadata/datapoints (initial save)
        _t0 = time()
        experiment.store(save_dir=output_dir)
        _elapsed = time() - _t0
        print(f"   experiment.store() completed in {_elapsed:.3f}s")
        NUM_OBJECTS_IN_PICKLE = 16


        # Run generation
        print("\n2. Running model generation (capturing attention maps)...")
        def callback(saved_batches, start_idx, end_idx):
            print(f"  Saving datapoints {start_idx} to {end_idx} to disk...")
            _t_start = time()
            experiment.store_datapoints_only(output_dir, start_index=start_idx, end_index=end_idx, offset_relative_to_experiment=0)
            
            # Immediately clear the activations for this batch to avoid OOM
            print(f"  Clearing activations from memory for datapoints {start_idx} to {end_idx}...")
            experiment.clear_activations(start_idx, end_idx)

            _t_store = time() - _t_start
            print(f"  Stored datapoints {start_idx}-{end_idx} in {_t_store:.3f}s")
            try:
                if run is not None:
                    run.log({
                        "store_datapoints_batch_time_s": _t_store,
                        "store_datapoints_batch_start_idx": start_idx,
                        "store_datapoints_batch_end_idx": end_idx,
                    })
            except Exception:
                pass

        # Update generator with current experiment
        generator.experiment = experiment
        # generator.generate calls capturer bind/unbind via context manager, so this is safe assuming unexpected side effects don't occur.
        generator.generate(batch_size=NUM_OBJECTS_IN_PICKLE, datapoints_callback=callback)

        print("   Generation complete!")

        # Unload generator to free GPU memory before loading judge
        # Actually, we shouldn't unload if we want to reuse the model across the 28 iterations!
        # So we skip unload_model here.
        
        # Save a lightweight version of all datapoints (no activations) for quick loading
        print("  Saving lightweight datapoints (no activations) for quick analysis...")
        experiment.store_datapoints_without_activations(output_dir)

        print("\n" + "=" * 80)
        print(f"Experiment {experiment.name} Complete!")
        print("=" * 80)

        if run:
            run.finish()

    # After all 28 layers:
    if generator:
        generator.unload_model()

    print("\n" + "=" * 80)
    print("Running judge on all experiments...")
    print("=" * 80)
    
    # Run judge for all experiments
    judge = CorrectnessJudge(all_experiments[0], device='cuda')
    for exp in all_experiments:
        print(f"\nRunning judge for experiment {exp.name}...")
        judge.experiment = exp
        judge.run(batch_size=8)
        
        # Save updated lightweight datapoints (with judge decisions)
        print("  Saving updated lightweight datapoints with judge decisions...")
        exp.store_datapoints_without_activations(output_dir)
        
    judge.unload_model()

    print("\nAll experiments judged and updated.")

if __name__ == "__main__":
    run_generation()
