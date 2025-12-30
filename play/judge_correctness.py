
"""Judge Correctness Script

Use Hugging Face transformers + torch to run a judge to verify if the
model answered correctly. This replaces the previous vLLM-based
implementation with a simpler `transformers` generation loop.
"""

from pathlib import Path
from typing import Iterable, List
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from play.interpetability_data_point import InterpretabilityDataPoint
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

# Generation parameters for transformers-based generation
GENERATION_CONFIG = dict(
    temperature=0.3,
    top_p=0.95,
    max_new_tokens=512,
    stop_strings=["<|endoftext|>", "\n\n\n"],
    skip_special_tokens=False,
)


def _cut_at_stop(s: str, stops: List[str]) -> str:
    """Cut string `s` at the first occurrence of any substring in `stops`.

    Returns the prefix up to the first stop (exclusive). If no stop is found,
    returns `s` unchanged.
    """
    indices = [s.find(t) for t in stops if t in s]
    if not indices:
        return s
    first = min(indices)
    return s[:first]


def run_judge_validation(datapoints: Iterable[InterpretabilityDataPoint],
                         judge_prompt: str = JUDGE_PROMPT,
                         model_path: Path = MODEL_PATH,
                         generation_config: dict = GENERATION_CONFIG,
                         batch_size: int = 8):
    """
    Run judge validation on the provided datapoints to verify correctness.
    """

    # Ensure datapoints are a list so we can index and update them
    datapoints = list(datapoints)
    # Prepare generation inputs
    texts = [judge_prompt.format(
        question=dp.question,
        correct_answer=dp.correct_answer,
        generated_text=dp.model_response,
    ) for dp in datapoints]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)

    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Batch generation to avoid memory issues
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_new_tokens=generation_config["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # For each generated output, extract newly generated tokens and postprocess
        for i, gen in enumerate(generated_ids):
            # length of original input for this example (before padding/truncation)
            orig_len = (attention_mask[i] == 1).sum().item()
            new_tokens = gen[orig_len:]
            judge_response = tokenizer.decode(new_tokens, skip_special_tokens=generation_config["skip_special_tokens"]).strip()
            # Extract judge's final answer (after their thinking) and cut at stop sequences
            judge_response = _cut_at_stop(judge_response, generation_config["stop_strings"]).strip()
            judge_decision = extract_judge_decision(judge_response)

            datapoint = datapoints[start + i]
            datapoint.update({
                "original_index": start + i,
                "judge_full_response": judge_response,
                "judge_decision": judge_decision,
                "timestamp": datetime.now().isoformat(),
            })

    total = len(datapoints)
    # Analyze recovery success based on judge decisions
    successful_recoveries = sum(
        1 for r in datapoints
        if r.get("judge_decision") == "yes"
    )
    failed_recoveries = sum(
        1 for r in datapoints
        if r.get("judge_decision") == "no"
    )
    unclear_cases = sum(
        1 for r in datapoints 
        if r.get("judge_decision") == "unclear"
    )
    
    print(f"\nJudge Evaluation Results:")
    print(f"  Correct (yes): {successful_recoveries}/{total} ({100*successful_recoveries/total:.1f}%)")
    print(f"  Incorrect (no): {failed_recoveries}/{total} ({100*failed_recoveries/total:.1f}%)")
    print(f"  Unclear: {unclear_cases}/{total} ({100*unclear_cases/total:.1f}%)")

