#!/home/ADV_2526a/evyataroren/inter_2025/.miniconda3/envs/inter25_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' 
#SBATCH --mem=32G
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable



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
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "datasets")
os.environ["VLLM_USAGE_SOURCE"] = "production"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["FLASHINFER_WORKSPACE_DIR"] = os.path.join(CACHE_BASE, "flashinfer")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONPATH"] = PROJECT_HOME + ":" + os.environ.get("PYTHONPATH", "")

# Add PROJECT_HOME to Python path so 'play' module can be imported
sys.path.insert(0, PROJECT_HOME)

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



import pickle
import torch
from typing import Dict, List, Optional
from pipeline.interface import (
    Experiment, GenerationMode, ModelGenerationConfig, JudgeGenerationConfig, 
    SamplingParams, DataPoint, ActivationCapturer,
    ShouldStop, Injection, ModelPromptTemplate, JudgePromptTemplate
)
from pipeline.dataset_loaders import aggregated_dataset_loader, MATH500Loader, aggregate_shuffle_strategy, SimpleDatasetLoader
from pipeline.generate_normal import GenerateSimple
from pipeline.judge_correctness import CorrectnessJudge
from pipeline.hooks import AttentionMapCapturer, AttentionMapCapturerClipActivations
from pipeline.injections import SunWeightRedirectInjection, QuadraticFormulaRedirectInjection, NearMissInjection


class SentenceEndStopCondition(ShouldStop):
    """Stop when encountering sentence-ending punctuation after 20 tokens."""
    def should_stop(self, previous_tokens: Optional[list[str]] = None) -> bool:
        if previous_tokens is None or len(previous_tokens) < 20:
            return False
        if previous_tokens[-1] in {'.', '!', '?'} or previous_tokens[-1].endswith(('.', '!', '?')):
            return True
        return False

class EOSTokenStopCondition(ShouldStop):
    """Stop when encountering the end-of-sentence special token."""
    def should_stop(self, previous_tokens: Optional[list[str]] = None) -> bool:
        if previous_tokens == [] or previous_tokens is None:
            return False
        return previous_tokens[-1] == '<｜end▁of▁sentence｜>'

class ShortAnswerPromptTemplate(ModelPromptTemplate):
    """Format question with instruction to answer shortly."""
    def format(self, question: str) -> str:
        return f'<｜begin▁of▁sentence｜>Answer the question in short<｜User｜>{question}<｜Assistant｜><think>\n'

class MathEvaluatorJudgePrompt(JudgePromptTemplate):
    """Format judge prompt for problem evaluation."""
    def format(self, question: str, model_answer: str, correct_answer: str) -> str:
        return (
"""
<｜begin▁of▁sentence｜>'
You are an expert problem evaluator. Compare the student's answer with the correct answer.<｜User｜>"
Question: \"\"\"{question} \"\"\"


Correct Answer: \"\"\"{correct_answer} \"\"\"


Student's Response:  \"\"\" {model_answer} \"\"\"


1. Does the student's response correctly answer the question?"
2. Do they try to answer the question but have an incorrect answer?"
3. Do they fail to provide an answer at all?"
4. Or do they talk about a totaly different subject?
Respond with ONLY with:
'correct' for option 1.
'incorrect' for option 2.
'no_answer' for option 3.
'irrelevant' for option 4.
<｜Assistant｜>\n"
"""        
)

output_dir = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
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
        max_new_tokens=1024
    ),
    dtype=torch.bfloat16
)

judge_config = JudgeGenerationConfig(
    judge_name="qwen-7B-judge",
    judge_model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
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
    base_path="/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets"
)

experiment = Experiment(
    name="ds_simple,inject_EoSen,attention_capture_and_clip,inject_sun,with_judge",
    dataset=dataset,
    model_generation_config=model_config,
    judge_generation_config=judge_config,
    seed=42,
    activation_capturer=AttentionMapCapturer()
)

print(f"   Populating datapoints from dataset...")
experiment.populate_datapoints(num=50)
for dp in experiment.datapoints:
    dp.should_capture_activations = True

print(f"   Loaded {len(experiment.datapoints)} datapoints from dataset.")



def run_generation():
    print("=" * 80)
    print(f"Starting Generation {experiment.name}")
    print("=" * 80)
    


    if JOB_ID == 0:
        print (f"\n1. Storing experiment metadata to {output_dir}...")
        experiment.store(save_dir=output_dir) 

    total_datapoints = len(experiment.datapoints)
    per_job = ceil(total_datapoints / NUM_JOBS)
    start = min(per_job * JOB_ID, total_datapoints)
    end = min(per_job * (JOB_ID + 1), total_datapoints)
    print(f"Task ID {JOB_ID} (env {JOB_ID}): Processing datapoints {start} to {end} (total {total_datapoints}, jobs {NUM_JOBS})")
    experiment.datapoints = experiment.datapoints[start:end]
    
    # Run generation
    print("\n2. Running model generation (capturing attention maps)...")
    generator = GenerateSimple(experiment, device='cuda')

    # Save every N datapoints as they complete to avoid waiting until the end
    NUM_OBJECTS_IN_PICKLE = 10
    # `start` is the global index offset in the original full datapoints list (computed above)

    saved_batches = 0
    total_subset = len(experiment.datapoints)
    for idx, dp in enumerate(experiment.datapoints):
        if idx % 10 == 0:
            print(f"Processing datapoint {idx}/{total_subset} (subset)...")
        # generate single datapoint (keeps attention capture behavior)
        generator._generate_single_datapoint(dp)

        # Periodically save completed datapoints in batches (indices are relative to this subset)
        if (idx + 1) % NUM_OBJECTS_IN_PICKLE == 0 or (idx + 1) == total_subset:
            # subset slice indices (inclusive/exclusive)
            start_sub = saved_batches * NUM_OBJECTS_IN_PICKLE
            end_sub_excl = min((saved_batches + 1) * NUM_OBJECTS_IN_PICKLE, total_subset)

            
            print(f"  Saving datapoints {start_sub} to {end_sub_excl} to disk...")
            experiment.store_datapoints_only(output_dir, start_index=start_sub, end_index=end_sub_excl, offset_relative_to_experiment=start)
            # NOTE: Don't clear activations here! We need them for the final save after judge
            # print("  Clearing activations from memory...")
            # experiment.clear_activations(start_sub, end_sub_excl)
            # print(f" Cleared activations for datapoints {start_sub} to {end_sub_excl}")
            saved_batches += 1

    print("   Generation complete!")
    # Unload model to free memory
    generator.unload_model()
    
    # Force garbage collection and clear CUDA cache before loading judge
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    print("\n3. Running judge validation...")
    judge = CorrectnessJudge(experiment, device='cuda')
    judge.run(batch_size=8)
    judge.unload_model()

    print("\n4. Done generating judge decisions. datapoints populated")

    print("resaving datapoints with judge decisions...")
    experiment.store_datapoints_only(output_dir, start_index=0, end_index=len(experiment.datapoints), offset_relative_to_experiment=start,override=True)
    
    # Verify activations were captured (before clearing)
    print("\n2a. Verifying captured activations...")
    total_activations = 0
    for dp_idx, dp in enumerate(experiment.datapoints):
        activation_types = [
            ('activations_question', dp.activations_question),
            ('activations_upto_injection', dp.activations_upto_injection),
            ('activations_injection', dp.activations_injection),
            ('activations_after_injection', dp.activations_after_injection),
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
    


    # Note: datapoints are saved incrementally during generation.
    print("   Datapoints saved successfully (incremental saves during generation)!")
    
    # Clear activations from memory after verification
    print("  Clearing activations from memory...")
    experiment.clear_activations(0, len(experiment.datapoints))
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Total questions: {len(experiment.datapoints)}")
    # TODO: make this a real count of judge answers...
    # print(f"  - Correct answers: N/A (judge validation disabled)")
    print(f"  - Total tokens with activations: {total_activations}")
if __name__ == "__main__":
    run_generation()