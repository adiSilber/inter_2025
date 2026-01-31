#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=layer_sweep
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090'
#SBATCH --mem=32G
#SBATCH --output=logs/layer_sweep_%A_%a.out
#SBATCH --error=logs/layer_sweep_%A_%a.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

"""
Per-layer attention clipping sweep.
SLURM array job: each task clips attention to question tokens at ONE layer only.
Uses 10 simple questions with SunWeightRedirectInjection + SentenceEndStopCondition.
No judge. No attention weight saving.
"""

from datetime import datetime
from math import ceil
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

PROJECT_HOME = os.getcwd()
CACHE_BASE = os.path.join(PROJECT_HOME, ".cache")
os.makedirs(CACHE_BASE, exist_ok=True)

os.environ["PROJECT_HOME"] = PROJECT_HOME
os.environ["CACHE_BASE"] = CACHE_BASE
os.environ["HF_HOME"] = os.path.join(CACHE_BASE, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "datasets")
os.environ["VLLM_USAGE_SOURCE"] = "production"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["FLASHINFER_WORKSPACE_DIR"] = os.path.join(CACHE_BASE, "flashinfer")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONPATH"] = PROJECT_HOME + ":" + os.environ.get("PYTHONPATH", "")

sys.path.insert(0, PROJECT_HOME)

# ---------------------------------------------------------------------------
# Resolve layer index from SLURM array task ID
# ---------------------------------------------------------------------------
LAYER_IDX = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
from pipeline.interface import (
    Experiment, ModelGenerationConfig,
    SamplingParams,
)
from pipeline.dataset_loaders import aggregated_dataset_loader, SimpleDatasetLoader, aggregate_shuffle_strategy
from pipeline.generate_normal import GenerateSimple
from pipeline.hooks import AttentionMapCapturerClipActivations
from pipeline.injections import (
    SunWeightRedirectInjection, SentenceEndStopCondition,
    EOSTokenStopCondition, ShortAnswerPromptTemplate,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
date_str = datetime.now().strftime("%y%m%d")
output_base = os.path.join(
    "/home/ADV_2526a/evyataroren/inter_2025/artifacts",
    f"layer_sweep_simple_sun_{date_str}",
)
output_dir = os.path.join(output_base, f"layer_{LAYER_IDX}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Build experiment
# ---------------------------------------------------------------------------
model_config = ModelGenerationConfig(
    model_name="qwen-7B",
    model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
    should_stop=SentenceEndStopCondition(),
    get_injection=SunWeightRedirectInjection(),
    global_stop=EOSTokenStopCondition(),
    question_prompt_template=ShortAnswerPromptTemplate(),
    sampling_params=SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        take_dumb_max=False,
        max_new_tokens=1024,
    ),
    dtype=torch.bfloat16,
)

dataset = aggregated_dataset_loader(
    datasets=[SimpleDatasetLoader],
    seed=42,
    strategy=aggregate_shuffle_strategy.SEQUENTIAL,
    base_path="/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets",
)

capturer = AttentionMapCapturerClipActivations(
    layers_to_clip={LAYER_IDX},
    capture_weights=False,
)

experiment = Experiment(
    name=f"layer_sweep_clip_layer_{LAYER_IDX}",
    dataset=dataset,
    model_generation_config=model_config,
    judge_generation_config=None,
    seed=42,
    activation_capturer=capturer,
)

print(f"Populating datapoints from dataset...")
experiment.populate_datapoints(num=10)
for dp in experiment.datapoints:
    dp.should_capture_activations = True

print(f"Loaded {len(experiment.datapoints)} datapoints. Layer to clip: {LAYER_IDX}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_generation():
    print("=" * 80)
    print(f"Layer sweep — clipping layer {LAYER_IDX}")
    print("=" * 80)

    # Store experiment metadata
    experiment.store(save_dir=output_dir)

    # Run generation
    print("\nRunning model generation ...")
    generator = GenerateSimple(experiment, device="cuda")

    NUM_OBJECTS_IN_PICKLE = 10
    saved_batches = 0
    total = len(experiment.datapoints)

    for idx, dp in enumerate(experiment.datapoints):
        if idx % 5 == 0:
            print(f"Processing datapoint {idx}/{total} ...")
        generator._generate_single_datapoint(dp)

        if (idx + 1) % NUM_OBJECTS_IN_PICKLE == 0 or (idx + 1) == total:
            start_sub = saved_batches * NUM_OBJECTS_IN_PICKLE
            end_sub_excl = min((saved_batches + 1) * NUM_OBJECTS_IN_PICKLE, total)
            print(f"  Saving datapoints {start_sub} to {end_sub_excl} ...")
            experiment.store_datapoints_only(
                output_dir,
                start_index=start_sub,
                end_index=end_sub_excl,
                offset_relative_to_experiment=0,
            )
            saved_batches += 1

    print("Generation complete!")
    generator.unload_model()

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f"Layer {LAYER_IDX} sweep complete — {total} datapoints")
    print("=" * 80)


if __name__ == "__main__":
    run_generation()
