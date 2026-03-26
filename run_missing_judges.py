#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=missing_judges
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090'
#SBATCH --mem=32G
#SBATCH --output=logs/missing_judges_%j.out
#SBATCH --error=logs/missing_judges_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable

"""
Run the judge on all clip-layer artifact directories (layers 0-27) for any
datapoints that are missing a judge_decision.

- Loads datapoints_NO_ACTIVATIONS.pkl from each directory
- Skips directories where every datapoint already has a decision
- Loads the judge model ONCE for all directories
- Writes the updated file back in-place
"""

import os
import sys
import re
from time import time

sys.stderr = sys.stdout
sys.stdout.reconfigure(line_buffering=True)  # type: ignore

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
sys.path.insert(0, os.path.join(PROJECT_HOME, "pipeline"))

import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
import dill
from tqdm import tqdm
import re as _re

from pipeline.interface import JudgeGenerationConfig, SamplingParams, CPUUnpickler
from pipeline.judge_correctness import JudgeDecision, CorrectnessJudge
from pipeline.prompts import SunOrNotSunBetterJudgePrompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
JUDGE_MODEL_PATH = "/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
JUDGE_BATCH_SIZE = 8

judge_config = JudgeGenerationConfig(
    judge_name="qwen-7B-judge",
    judge_model_path=JUDGE_MODEL_PATH,
    judge_prompt=SunOrNotSunBetterJudgePrompt(),
    sampling_params=SamplingParams(
        temperature=0.0,
        top_k=None,
        top_p=None,
        take_dumb_max=True,
        max_new_tokens=1024,
    ),
    dtype=torch.bfloat16,
)

# ---------------------------------------------------------------------------
# Collect all clip-layer artifact directories (all layers 0-27, incl. dupes)
# ---------------------------------------------------------------------------

def collect_artifact_dirs(artifacts_dir: str) -> list[str]:
    """Return every clip-layer-* subdirectory, sorted so lower layers come first."""
    dirs = []
    for name in os.listdir(artifacts_dir):
        full = os.path.join(artifacts_dir, name)
        if os.path.isdir(full) and "clip-layer" in name and "1.5B-batched" in name:
            no_act_file = os.path.join(full, "datapoints_NO_ACTIVATIONS.pkl")
            if os.path.exists(no_act_file):
                dirs.append(full)
    # Sort by the layer index embedded in the folder name, then by timestamp prefix
    def sort_key(d):
        m = _re.search(r"clip-layer-(\d+)", os.path.basename(d))
        layer = int(m.group(1)) if m else 999
        return (layer, os.path.basename(d))
    dirs.sort(key=sort_key)
    return dirs


# ---------------------------------------------------------------------------
# Load / save helpers (no Experiment needed)
# ---------------------------------------------------------------------------

def load_no_activations(pkl_path: str):
    """Load datapoints_NO_ACTIVATIONS.pkl → (datapoints, start, end)."""
    with open(pkl_path, "rb") as f:
        data = CPUUnpickler(f).load()
    if isinstance(data, tuple) and len(data) == 3:
        datapoints, start, end = data
    elif isinstance(data, list):
        datapoints = data
        start, end = 0, len(data)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")
    return datapoints, start, end


def save_no_activations(pkl_path: str, datapoints, start: int, end: int):
    """Overwrite pkl_path with the updated datapoints tuple."""
    with open(pkl_path, "wb") as f:
        dill.dump((datapoints, start, end), f)
    print(f"   Saved {len(datapoints)} datapoints → {pkl_path}")


# ---------------------------------------------------------------------------
# Standalone judge runner (does not need an Experiment object)
# ---------------------------------------------------------------------------

def run_judge_on_datapoints(judge: CorrectnessJudge, datapoints, batch_size: int = 8):
    """
    Run the judge on a list of DataPoint objects (in-place update).
    Mirrors the logic in CorrectnessJudge.run() but operates on a plain list.
    """
    config = judge.config

    for i in tqdm(range(0, len(datapoints), batch_size), desc="  Judging"):
        batch = datapoints[i: i + batch_size]
        prompts = []

        for dp in batch:
            full_model_response = dp.upto_injection_string + dp.after_injection_string
            model_answer = judge._extract_model_final_answer(full_model_response)
            prompt = config.judge_prompt.format(
                question=dp.question_contents,
                model_answer=model_answer,
                correct_answer=dp.question_correct_answer,
                injection=dp.injection,
            )
            prompts.append(prompt)

        inputs = judge.tokenizer(prompts, return_tensors="pt", padding=True).to(judge.device)

        with torch.no_grad():
            output_ids = judge.model.generate(
                **inputs,
                max_new_tokens=config.sampling_params.max_new_tokens,
                temperature=config.sampling_params.temperature,
                top_p=config.sampling_params.top_p,
                top_k=config.sampling_params.top_k,
                do_sample=(config.sampling_params.temperature or 0) > 0,
                pad_token_id=judge.tokenizer.pad_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]

        for j, dp in enumerate(batch):
            judge_full_response = judge.tokenizer.decode(
                generated_ids[j], skip_special_tokens=True
            )
            dp.judge_response = [
                t.replace("Ġ", " ").replace("Ċ", "\n")
                for t in judge.tokenizer.convert_ids_to_tokens(
                    generated_ids[j].tolist(), skip_special_tokens=True
                )
            ]
            dp.judge_prompt = [
                t.replace("Ġ", " ").replace("Ċ", "\n")
                for t in judge.tokenizer.convert_ids_to_tokens(
                    inputs.input_ids[j].tolist(), skip_special_tokens=True
                )
            ]
            dp.judge_decision = judge._parse_judge_decision(judge_full_response)
            print(f"    dp {dp.question_id}: {dp.judge_decision}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    artifact_dirs = collect_artifact_dirs(ARTIFACTS_DIR)

    print(f"Found {len(artifact_dirs)} clip-layer artifact directories:")
    for d in artifact_dirs:
        print(f"  {os.path.basename(d)}")

    # First pass: identify which dirs need judging
    needs_judging: list[tuple[str, str]] = []  # (dir_path, pkl_path)
    for d in artifact_dirs:
        pkl_path = os.path.join(d, "datapoints_NO_ACTIVATIONS.pkl")
        datapoints, _, _ = load_no_activations(pkl_path)
        missing = [dp for dp in datapoints if dp.judge_decision is None]
        if missing:
            pct = 100 * len(missing) / len(datapoints)
            print(f"  [{os.path.basename(d)}] {len(missing)}/{len(datapoints)} datapoints need judging ({pct:.0f}%)")
            needs_judging.append((d, pkl_path))
        else:
            print(f"  [{os.path.basename(d)}] all {len(datapoints)} datapoints already judged — skipping")

    if not needs_judging:
        print("\nNothing to judge. All done.")
        return

    print(f"\n{len(needs_judging)} directories need judging. Loading judge model (once)...")
    t0 = time()

    # Build a minimal stub so CorrectnessJudge.__init__ doesn't raise.
    # We won't call judge.run(); instead we call run_judge_on_datapoints() directly.
    class _MinimalExperiment:
        judge_generation_config = judge_config
        datapoints = []

    judge = CorrectnessJudge(_MinimalExperiment(), device="cuda")  # type: ignore
    print(f"Judge loaded in {time() - t0:.1f}s")

    # Second pass: judge + save
    for d, pkl_path in needs_judging:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(d)}")
        datapoints, start, end = load_no_activations(pkl_path)
        missing = [dp for dp in datapoints if dp.judge_decision is None]
        print(f"  {len(missing)} / {len(datapoints)} datapoints to judge")

        t1 = time()
        run_judge_on_datapoints(judge, missing, batch_size=JUDGE_BATCH_SIZE)
        elapsed = time() - t1
        print(f"  Judging done in {elapsed:.1f}s")

        # Print summary
        for dec in JudgeDecision:
            count = sum(1 for dp in datapoints if dp.judge_decision == dec)
            print(f"  {dec.name}: {count}")
        none_left = sum(1 for dp in datapoints if dp.judge_decision is None)
        print(f"  Still None: {none_left}")

        save_no_activations(pkl_path, datapoints, start, end)

    judge.unload_model()
    print("\nAll done.")


if __name__ == "__main__":
    main()
