#!/specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/.miniconda3/envs/inter2025_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' 
#SBATCH --mem=32G
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



import os
import sys


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
os.environ["PYTHONPATH"] = PROJECT_HOME + ":" + os.environ.get("PYTHONPATH", "")

# Add PROJECT_HOME to Python path so 'play' module can be imported
sys.path.insert(0, PROJECT_HOME)




import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)



import pickle
import torch
from typing import Dict, List
from datetime import datetime
from play.interface import Experiment, ModelGenerationConfig, JudgeGenerationConfig, SamplingParams, DataPoint, ActivationCapturer
from play.dataset_loaders import aggregated_dataset_loader, MATH500Loader, aggregate_shuffle_strategy
from play.judge_correctness import run_judge_validation
from play.generate_normal import GenerateSimple


def global_stop_fn(tokens):
    return any(tok in ['<|endoftext|>', '<|im_end|>', '</think>', '<|end_of_sentence|>'] for tok in tokens)

def question_prompt_template(q):
    return f"<|begin_of_sentence|><|User|> {q}<|Assistant|><think>"

def create_math500_experiment() -> Experiment:
    """Create an experiment for MATH-500 dataset with attention capture."""
    
    # Model generation config (no injection)
    model_config = ModelGenerationConfig(
        model_name="qwen-7B",
        model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
        should_stop_fn=lambda tokens: False,
        get_injection_fn=lambda tokens: None,
        global_stop_fn=global_stop_fn,
        question_prompt_template=question_prompt_template,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            take_dumb_max=False,
            max_new_tokens=1024
        ),
        dtype=torch.bfloat16
    )
    
    # Load MATH-500 dataset
    dataset = aggregated_dataset_loader(
        datasets=[MATH500Loader],
        seed=42,
        strategy=aggregate_shuffle_strategy.SEQUENTIAL,
        base_path="datasets/datasets"
    )
        
    # Create experiment
    experiment = Experiment(
        dataset=dataset,
        model_generation_config=model_config,
        judge_generation_config=None,
        seed=42,
    )
    
    # Populate datapoints from dataset
    experiment.populate_datapoints()    
    # Mark all datapoints to capture activations
    for dp in experiment.datapoints:
        dp.should_capture_activations = True
    
    return experiment


def main():
    print("=" * 80)
    print("Starting Clean MATH-500 Generation")
    print("=" * 80)
    
    # Create experiment
    print("\n1. Creating experiment configuration...")
    experiment = create_math500_experiment()
    print(f"   Loaded {len(experiment.datapoints)} datapoints from MATH-500")
    
    # Run generation
    print("\n2. Running model generation (capturing attention maps)...")
    generator = GenerateSimple(experiment, device='cuda')
    generator.generate()
    print("   Generation complete!")
    
    # Unload model to free memory
    generator.unload_model()
    
    
    # Save experiment
    output_dir = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"math500_clean_no_activations_{timestamp}.pkl")
    
    print(f"\n4. Saving datapoints to {output_path} (only datapoints will be pickled)...")
    # Save only the datapoints list to avoid pickling non-serializable model/code objects
    with open(output_path, 'wb') as f:
        pickle.dump(experiment.datapoints, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("   Datapoints saved successfully!")
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Total questions: {len(experiment.datapoints)}")
    print(f"  - Correct answers: N/A (judge validation disabled)")
    print(f"  - Datapoints pickled to: {output_path}")
    

if __name__ == "__main__":
    main()
