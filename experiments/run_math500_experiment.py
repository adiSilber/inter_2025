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


class AttentionMapCapturer(ActivationCapturer):
    """Captures attention maps from all layers and heads during model forward pass."""
    
    def __init__(self):
        super().__init__()
        self.hooks = []
        self.model = None
    
    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare to capture attention maps."""
        self.model = model
        
        # Find all layers with attention modules
        # For transformers, typically model.layers[i].self_attn or similar
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Could not find model layers for attention capture")
        
        # Initialize activation storage for each layer
        for layer_idx in range(len(layers)):
            layer_name = f"layer_{layer_idx}_attention"
            self.activations[layer_name] = []
        # Ensure model forwards return attentions by default when possible.
        try:
            if hasattr(self.model, 'config'):
                # Prefer to enable attention outputs so hooks can see weights.
                setattr(self.model.config, 'output_attentions', True)
                # Some implementations expose an attn implementation flag; prefer eager.
                try:
                    setattr(self.model.config, 'attn_implementation', 'eager')
                except Exception:
                    pass
        except Exception:
            pass
    
    def __enter__(self):
        """Register hooks to capture attention outputs."""
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")
        
        # Find layers and register hooks
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find model layers")
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Qwen/DeepSeek attention modules typically return a tuple where
                # output[1] are the attention weights when `output_attentions=True`.
                attention_weights = None
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]

                # If weights are missing, raise instructive error so caller can
                # ensure `output_attentions=True` or that model config was set.
                if attention_weights is None:
                    raise ValueError(
                        f"Layer {layer_idx}: attention weights are None. "
                        "Ensure the model is called with `output_attentions=True` "
                        "or set `model.config.output_attentions = True` and use an "
                        "attn implementation that exposes weights (e.g. 'eager')."
                    )

                # Offload to CPU and detach to save GPU memory and break graph
                try:
                    cpu_attn = attention_weights.detach().cpu()
                except Exception:
                    # If not a tensor, store as-is
                    cpu_attn = attention_weights

                self.activations[f"layer_{layer_idx}_attention"].append(cpu_attn)

            return hook
        
        # Register hooks on attention modules
        for layer_idx, layer in enumerate(layers):
            # Different model architectures use different names
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            else:
                continue
            
            hook_handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks after capture."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# Define picklable functions for model generation
def should_stop_fn(tokens):
    return False

def get_injection_fn(tokens):
    return ""

def global_stop_fn(tokens):
    return any(tok in ['<|endoftext|>', '<|im_end|>', '</think>', '<|end_of_sentence|>'] for tok in tokens)

def question_prompt_template(q):
    return f"<|begin_of_sentence|><|User|> {q}<|Assistant|><think>"

def judge_prompt_fn(question, correct_answer, generated_text):
    return (
        f"<|begin_of_sentence|><|User|> "
        f"You are an expert math problem evaluator. Compare the model's answer with the correct answer.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Model's Response: {generated_text}\n\n"
        f"Does the model's response correctly answer the question? "
        f"Respond with ONLY 'yes' or 'no' at the end of your evaluation."
        f"<|Assistant|><think>"
    )


def create_math500_experiment() -> Experiment:
    """Create an experiment for MATH-500 dataset with attention capture."""
    
    # Model generation config (no injection)
    model_config = ModelGenerationConfig(
        model_name="qwen-7B",
        model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
        should_stop_fn=should_stop_fn,
        get_injection_fn=get_injection_fn,
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
    
    # Judge config
    judge_config = JudgeGenerationConfig(
        judge_name="qwen-7B-judge",
        judge_model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
        judge_prompt=judge_prompt_fn,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_k=None,
            top_p=None,
            take_dumb_max=True,
            max_new_tokens=256,
            
        ),
        dtype=torch.bfloat16
    )
    
    # Load MATH-500 dataset
    dataset = aggregated_dataset_loader(
        datasets=[MATH500Loader],
        seed=42,
        strategy=aggregate_shuffle_strategy.SEQUENTIAL,
        base_path="/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets"
    )
    
    # Create attention capturer
    attention_capturer = AttentionMapCapturer()
    
    # Create experiment
    experiment = Experiment(
        dataset=dataset,
        model_generation_config=model_config,
        judge_generation_config=judge_config,
        seed=42,
        activation_capturer=attention_capturer
    )
    
    # Populate datapoints from dataset
    experiment.populate_datapoints()
    
    # Limit to first 10 samples for testing
    experiment.datapoints = experiment.datapoints[:10]
    
    # Mark all datapoints to capture activations
    for dp in experiment.datapoints:
        dp.should_capture_activations = True
    
    return experiment


def main():
    print("=" * 80)
    print("Starting MATH-500 Experiment (No Injection, Attention Capture)")
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
    
    # Verify activations were captured
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
    
    # # Run judge validation
    # print("\n3. Running judge validation...")
    # run_judge_validation(experiment)
    
    # # Count results
    # correct = sum(1 for dp in experiment.datapoints if dp.judge_decision)
    # total = len(experiment.datapoints)
    # print(f"   Judge results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    
    # Save experiment
    output_dir = "/home/ADV_2526a/evyataroren/inter_2025/experiments"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"math500_no_injection_{timestamp}.pkl")
    
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
    print(f"  - Total tokens with activations: {total_activations}")
    print(f"  - Datapoints pickled to: {output_path}")
    

if __name__ == "__main__":
    main()
