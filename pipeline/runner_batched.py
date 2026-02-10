#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
# #SBATCH --array=0-4
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' 
#SBATCH --mem=32G
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
from typing import Any
from pipeline.interface import (
    Experiment,  ModelGenerationConfig, JudgeGenerationConfig, 
    SamplingParams
)
from pipeline.dataset_loaders import aggregated_dataset_loader, aggregate_shuffle_strategy, SimpleDatasetLoader
from pipeline.judge_correctness import JudgeDecision
from pipeline.judge_correctness import CorrectnessJudge
from pipeline.hooks import AttentionHeadClipper
from pipeline.injections import (
    SunWeightRedirectInjection,
)

from pipeline.stops import SentenceEndStopCondition, EOSTokenStopCondition
from pipeline.wandb_utils import experiment_config_for_wandb
import wandb
from pipeline.generate_batched import GenerateBatched
from pipeline.prompts import ShortAnswerPromptTemplate, MathEvaluatorJudgePrompt



output_dir = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
model_path = "/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
judge_model_path = "/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
dataset_path = "/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets"
model_config = ModelGenerationConfig(
    model_name="qwen-7B",
    model_path=model_path,
    should_stop=SentenceEndStopCondition(),
    get_injection=SunWeightRedirectInjection(),
    global_stop=EOSTokenStopCondition(),
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
    judge_prompt=MathEvaluatorJudgePrompt(),
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
    datasets=[SimpleDatasetLoader],
    seed=42,
    strategy=aggregate_shuffle_strategy.SEQUENTIAL,
    base_path=dataset_path
)

experiment = Experiment(
    name="batched_test_ds_simple,inject_eoSen,attention_clip_0,inject_sunWeight",
    runner_name="batched runner",
    dataset=dataset,
    model_generation_config=model_config,
    judge_generation_config=judge_config,
    seed=42,
    unique_id=os.environ.get("SLURM_ARRAY_JOB_ID", str(int(time()))) ,
    activation_head_clipping={0:[0,1,2]},
    clip_max_val=1e-6,
    activation_capturer=AttentionHeadClipper()
)

print(f"   Populating datapoints from dataset...")
experiment.populate_datapoints(num=100)
experiment.datapoints = experiment.datapoints[:10]
for dp in experiment.datapoints:
    dp.should_capture_activations = True

print(f"   Loaded {len(experiment.datapoints)} datapoints from dataset.")


def run_generation():
    print("=" * 80)
    print(f"Starting Generation {experiment.name}")
    print("=" * 80)

    run = wandb.init(
        entity="inter_2026",
        project="Aha-moments",
        name=experiment.name,
        config=experiment_config_for_wandb(experiment, output_dir),
    )
    if run is not None:
        experiment.wandb_run_id = run.id

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

    generator = GenerateBatched(experiment, device='cuda')
    generator.generate(batch_size=NUM_OBJECTS_IN_PICKLE, datapoints_callback=callback)

    print("   Generation complete!")
    generator.unload_model()
    experiment.datapoints_to_cpu()

    
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
    
    if run is not None:
        run.summary.update(wandb_summary)

    print("Updated wandb run with judge results.")

    print(f"   Judge decision breakdown:")
    for decision, count in decision_counts.items():
        ratio = count / total_judged if total_judged else 0.0
        print(f"      {decision.value}: {count}/{total_judged} ({ratio:.4f})")

    print("resaving datapoints with judge decisions...") 
    _t_start_resave = time()
    experiment.store_datapoints_only(output_dir,override=True)
    _t_resave = time() - _t_start_resave
    print(f"  Resaved all datapoints (with judge decisions) in {_t_resave:.3f}s")
    try:
        if run is not None:
            run.log({"resave_datapoints_time_s": _t_resave})
    except Exception:
        pass
    
    # Verify activations were captured (before clearing)
    print("\n2a. Verifying captured activations...")
    total_activations = 0
    for dp_idx, dp in enumerate(experiment.datapoints):
        activation_types = [
            (attr_name, getattr(dp, attr_name))
            for attr_name in dir(dp)
            if attr_name in {"activations_after_injection", "activations_injection", "activations_question", "activations_upto_injection"}
        ]
        
        print(f"\n   DataPoint {dp_idx} ({dp.question_id}):")
        for activation_name, activation_dict in activation_types:
            if activation_dict is not None:
                print(f"      {activation_name}:")
                print(f"         Keys: {list(activation_dict.keys())}")
                for key, tensor_list in activation_dict.items():
                    num_tensors = len(tensor_list)
                    # Build a robust shapes summary for reporting. Handles:
                    #  - None entries
                    #  - torch.Tensor entries
                    #  - nested lists/tuples of tensors
                    shapes = []
                    try:
                        for t in tensor_list:
                            if t is None:
                                shapes.append(None)
                            elif isinstance(t, torch.Tensor):
                                shapes.append(tuple(t.shape))
                            elif isinstance(t, (list, tuple)):
                                inner_shapes = []
                                for e in t:
                                    if isinstance(e, torch.Tensor):
                                        inner_shapes.append(tuple(e.shape))
                                    else:
                                        inner_shapes.append(type(e).__name__)
                                shapes.append(inner_shapes)
                            else:
                                shapes.append(type(t).__name__)
                    except Exception as e:
                        shapes = [f"<shape-error: {e}>"]

                    # Always print the number of entries and the shapes summary
                    print(f"            {key}: {num_tensors} tensors, shapes: {shapes}")
                    total_activations += num_tensors
            else:
                print(f"      {activation_name}: None")
    


    
    print("  Clearing activations from memory...")
    experiment.clear_activations(0, len(experiment.datapoints))
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Total questions: {len(experiment.datapoints)}")
    print(f"  - Judge decision breakdown:")
    for decision, count in decision_counts.items():
        ratio = count / total_judged if total_judged else 0.0
        print(f"      {decision.value}: {count}/{total_judged} ({ratio:.4f})")
    print(f"  - Total tokens with activations: {total_activations}")

    if run:
        run.finish()

if __name__ == "__main__":
    run_generation()
