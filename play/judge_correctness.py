"""
Judge Correctness Script

Use vLLM to run a judge to verify if the model answered correctly.
"""

from pathlib import Path
from typing import Iterable
from datetime import datetime
from vllm import LLM, SamplingParams

from play.interface import InterpretabilityDataPoint, Experiment
from play.utils import extract_judge_decision


# Paths
BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "models" / "DS-qwen-7B" / "DeepSeek-R1-Distill-Qwen-7B"
PROMPTS_PATH = BASE_PATH / "prompts" / "play"
JUDGE_PROMPT_PATH = PROMPTS_PATH / "verification" / "judge_verification_prompt.md"

# Load judge prompts and formats
# The prompt can have placeholders for "question", and "model_answer".
with open(JUDGE_PROMPT_PATH, 'r', encoding='utf-8') as f:
    JUDGE_PROMPT = f.read()

# Sampling parameters for generation
SAMPLING_PARAMS = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=512,
    stop=["<|endoftext|>", "\n\n\n"],
    skip_special_tokens=False  # Don't skip special tokens in output
)

# ============================================================================
# Main Experiment
# ============================================================================

def run_judge_validation(experiment: Experiment,
                         judge_prompt: str = JUDGE_PROMPT):
    """
    Run judge validation on the provided datapoints to verify correctness.
    """
    model_config = experiment.judge_generation_config
    model_config.judge_model_path = MODEL_PATH
    model_config.judge_prompt = judge_prompt
    model_config.judge_name = MODEL_PATH.name
    model_config.sampling_params.max_new_tokens = SAMPLING_PARAMS.max_tokens
    model_config.sampling_params.seed = SAMPLING_PARAMS.seed
    model_config.sampling_params.max_new_tokens = SAMPLING_PARAMS.max_new_tokens
    model_config.sampling_params.temperature = SAMPLING_PARAMS.temperature
    model_config.sampling_params.top_p = SAMPLING_PARAMS.top_p

    datapoints: Iterable[InterpretabilityDataPoint] = experiment.datapoints
    # Load model with vLLM for efficient inference
    llm = LLM(
        model=str(MODEL_PATH),
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        skip_tokenizer_init=False,
        max_model_len=4096,
        enable_prefix_caching=True
    )
    
    texts = [judge_prompt.format(
        question=dp.question,
        correct_answer=dp.correct_answer,
        generated_text=dp.model_response
    ) for dp in datapoints]

    # Use add_special_tokens=False to only add BOS, no chat template tokens
    judge_outputs = llm.generate(
        texts,
        SAMPLING_PARAMS,
        use_tqdm=False
    )

    count_yes, count_no, count_unclear = 0, 0, 0
    for idx, (datapoint, output) in enumerate(zip(datapoints, judge_outputs)):
        judge_response = output.outputs[0].text.strip()
        # Extract judge's final answer (after their thinking)
        judge_decision = extract_judge_decision(judge_response)
        count_yes += 1 if judge_decision == 'yes' else 0
        count_no += 1 if judge_decision == 'no' else 0
        count_unclear += 1 if judge_decision == 'unclear' else 0

        judge_decision = True if judge_decision == 'yes' else False
        # Merge metadata from the original item so results are traceable
        datapoint.update({
            "original_index": idx,
            "judge_full_response": judge_response,
            "judge_decision": judge_decision,
        })

    total = len(datapoints)
    # Analyze recovery success based on judge decisions
    
    print(f"\nJudge Evaluation Results:")
    print(f"  Correct (yes): {count_yes}/{total} ({100*count_yes/total:.1f}%)")
    print(f"  Incorrect (no): {count_no}/{total} ({100*count_no/total:.1f}%)")
    print(f"  Unclear: {count_unclear}/{total} ({100*count_unclear/total:.1f}%)")
