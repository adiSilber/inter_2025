"""
Prompt Injection Recovery Test Script

Tests if a language model can recover from prompt injections during generation.
Uses vLLM for efficient batched inference with KV caching.

Experiment design:
- N questions from MATH-500 dataset
- K different prompts to format each question
- M injection texts that are inserted during generation
- Cartesian product: N x K x M combinations
- Second pass to verify if model can recover and answer correctly
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from itertools import product
from datetime import datetime
import re

# vLLM for efficient inference
from vllm import LLM, SamplingParams


# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / "models" / "DS-qwen-7B" / "DeepSeek-R1-Distill-Qwen-7B"
DATASET_PATH = BASE_PATH / "datasets" / "datasets" / "normalized_datasets.json"
OUTPUT_PATH = BASE_PATH / "play" / "prompt_injection_results.json"
PROMPTS_PATH = BASE_PATH / "prompts" / "play"

# Experiment parameters
N_QUESTIONS = 10  
K_PROMPTS = 3     # Number of question format prompts to use
M_INJECTIONS = 4  # Number of injection texts to use 

# Sampling parameters for generation
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    stop=["<|endoftext|>", "\n\n\n"],
    skip_special_tokens=False  # Don't skip special tokens in output
)

# Recovery verification parameters (judge model)
RECOVERY_SAMPLING_PARAMS = SamplingParams(
    temperature=0.3,
    top_p=0.95,
    max_tokens=512,  # Enough for thinking + yes/no answer
    skip_special_tokens=False  # Don't skip special tokens in output
)


# ============================================================================
# Prompt Loading Functions
# ============================================================================

def load_prompts_from_directory(directory: Path) -> List[Dict[str, str]]:
    """
    Load all markdown prompts from a directory.
    Returns a list of dicts with 'name' and 'content' keys.
    """
    prompts = []
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return prompts
    
    for md_file in sorted(directory.glob("*.md")):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            prompts.append({
                "name": md_file.stem,
                "content": content
            })
    
    return prompts


def load_all_prompts() -> tuple:
    """
    Load all prompts from the prompts directory structure.
    Returns: (question_prompts, injection_texts, recovery_prompt)
    """
    # Load question format prompts (K prompts)
    question_prompts = load_prompts_from_directory(
        PROMPTS_PATH / "question_formats"
    )
    
    # Load injection texts (M injections)
    injection_texts = load_prompts_from_directory(
        PROMPTS_PATH / "injections"
    )
    
    # Load recovery verification prompt
    recovery_path = PROMPTS_PATH / "verification" / "recovery_check.md"
    if recovery_path.exists():
        with open(recovery_path, 'r', encoding='utf-8') as f:
            recovery_prompt = f.read()
    else:
        recovery_prompt = "Extract the final answer from the text above."
    
    return question_prompts, injection_texts, recovery_prompt


# ============================================================================
# Helper Functions
# ============================================================================

def load_math500_questions(n: int = N_QUESTIONS) -> List[Dict[str, Any]]:
    """Load N random questions from MATH-500 dataset."""
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    math500 = data.get("MATH-500", [])
    if len(math500) < n:
        print(f"Warning: Only {len(math500)} questions available, using all.")
        return math500
    
    return random.sample(math500, n)


def should_inject(current_generation: str, question: str) -> bool:
    """
    Decide whether to inject at this point in the generation.
    Inject after the model has started reasoning but before conclusion.
    """
    # Inject if we have some reasoning (at least 50 tokens worth) but not yet at conclusion
    if len(current_generation) < 200:  # Not enough reasoning yet
        return False
    
    # Don't inject if we're near the end (conclusion indicators)
    conclusion_indicators = [
        "therefore", "thus", "final answer", "answer is", 
        "conclusion", "in summary", "\\boxed"
    ]
    
    lower_gen = current_generation.lower()
    recent_text = lower_gen[-200:]  # Check last 200 chars
    
    if any(indicator in recent_text for indicator in conclusion_indicators):
        return False
    
    # Inject at a natural break point (after sentence or paragraph)
    if current_generation.endswith(('.', '!', '?', '\n\n')):
        return True
    
    return False


def inject_text_callback(generated_so_far: str, injection_text: str, injected: List[bool]) -> str:
    """
    Callback function to inject text during generation.
    Only injects once when should_inject returns True.
    """
    if injected[0]:  # Already injected
        return ""
    
    if should_inject(generated_so_far, ""):
        injected[0] = True
        return injection_text
    
    return ""


def extract_judge_decision(text: str) -> str:
    """Extract yes/no decision from judge's response.
    
    Looks for 'yes' or 'no' AFTER the </think> tag is generated.
    The judge should output yes/no after finishing their thinking process.
    """
    # Look for the closing think tag (case-insensitive)
    think_close_patterns = [
        r'</think>',
        r'</thinking>',
        r'<\/think>',
        r'<\/thinking>'
    ]
    
    # Find where thinking ends
    think_end_pos = -1
    lower_text = text.lower()
    
    for pattern in think_close_patterns:
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            think_end_pos = match.end()
            break
    
    # Extract text after thinking (or use full text if no think tag found)
    if think_end_pos > 0:
        answer_portion = text[think_end_pos:].strip()
    else:
        # No think tag found - look in the last portion
        answer_portion = text[-500:].strip()
    
    # Convert to lowercase for matching
    answer_lower = answer_portion.lower()
    
    # Find all yes/no occurrences with word boundaries
    yes_matches = list(re.finditer(r'\byes\b', answer_lower))
    no_matches = list(re.finditer(r'\bno\b', answer_lower))
    
    # Collect all matches with their positions
    all_matches = []
    for match in yes_matches:
        all_matches.append(('yes', match.start()))
    for match in no_matches:
        all_matches.append(('no', match.start()))
    
    # Sort by position and take the last one
    if all_matches:
        all_matches.sort(key=lambda x: x[1])
        return all_matches[-1][0]
    
    # Fallback: check if the very last word is yes/no
    words = answer_lower.split()
    if words:
        last_word = words[-1].strip('.,!?;:\'"')
        if last_word in ['yes', 'no']:
            return last_word
    
    return "unclear"


# ============================================================================
# Main Experiment
# ============================================================================

def run_prompt_injection_experiment():
    """Run the full prompt injection recovery experiment."""
    print("="*80)
    print("PROMPT INJECTION RECOVERY EXPERIMENT")
    print("="*80)
    
    # Load all prompts from markdown files
    print("\n[1/6] Loading prompts from markdown files...")
    all_question_prompts, all_injection_texts, recovery_verification_prompt = load_all_prompts()
    
    # Select top K and M from loaded prompts
    question_prompts = all_question_prompts[:K_PROMPTS]
    injection_texts = all_injection_texts[:M_INJECTIONS]
    
    print(f"  Loaded {len(all_question_prompts)} question format prompts (using top {len(question_prompts)}):")
    for i, p in enumerate(question_prompts, 1):
        print(f"    {i}. {p['name']}")
    print(f"  Loaded {len(all_injection_texts)} injection texts (using top {len(injection_texts)}):")
    for i, inj in enumerate(injection_texts, 1):
        print(f"    {i}. {inj['name']}")
    
    print(f"\nModel: {MODEL_PATH}")
    print(f"Questions: {N_QUESTIONS}")
    print(f"Prompts (K): {len(question_prompts)}")
    print(f"Injections (M): {len(injection_texts)}")
    print(f"Total combinations: {N_QUESTIONS * len(question_prompts) * len(injection_texts)}")
    print("="*80)
    
    # Load model with vLLM for efficient inference
    print("\n[2/6] Loading model with vLLM...")
    llm = LLM(
        model=str(MODEL_PATH),
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        skip_tokenizer_init=False,
    )
    
    # Load questions
    print(f"\n[3/6] Loading {N_QUESTIONS} questions from MATH-500...")
    questions = load_math500_questions(N_QUESTIONS)
    print(f"Loaded {len(questions)} questions")
    
    # Generate all combinations
    print(f"\n[4/6] Generating combinations...")
    combinations = list(product(
        range(len(questions)),
        range(len(question_prompts)),
        range(len(injection_texts))
    ))
    print(f"Total combinations: {len(combinations)}")
    
    # Prepare all prompts for batched inference
    print(f"\n[5/6] Running inference on {len(combinations)} combinations...")
    batch_prompts = []
    combination_metadata = []
    
    for q_idx, p_idx, i_idx in combinations:
        question_data = questions[q_idx]
        question_text = question_data["question"]
        
        # Format the prompt using loaded template
        formatted_prompt = question_prompts[p_idx]["content"].format(question=question_text)
        
        # For this implementation, we'll inject the text at a fixed position
        # In a more sophisticated version, you'd use streaming and inject dynamically
        # Here we inject after a reasonable portion of the expected output
        prompt_with_placeholder = formatted_prompt
        
        batch_prompts.append(prompt_with_placeholder)
        combination_metadata.append({
            "question_idx": q_idx,
            "prompt_idx": p_idx,
            "injection_idx": i_idx,
            "question": question_text,
            "correct_answer": question_data["answer"],
            "prompt_name": question_prompts[p_idx]["name"],
            "injection_name": injection_texts[i_idx]["name"],
            "injection_text": injection_texts[i_idx]["content"]
        })
    
    # Generate initial responses (before injection)
    print("  Generating initial responses...")
    # Use add_special_tokens=False to only add BOS, no chat template tokens
    initial_outputs = llm.generate(
        batch_prompts, 
        SAMPLING_PARAMS,
        use_tqdm=False
    )
    
    # Now create injected versions
    print("  Creating injected versions...")
    injected_prompts = []
    for i, output in enumerate(initial_outputs):
        generated_text = output.outputs[0].text
        
        # Find a good injection point (middle of generation)
        injection_point = len(generated_text) // 2
        # Find the nearest sentence end after midpoint
        for j in range(injection_point, len(generated_text)):
            if generated_text[j] in '.!?\n':
                injection_point = j + 1
                break
        
        # Create injected prompt
        original_prompt = batch_prompts[i]
        partial_generation = generated_text[:injection_point]
        injection_text = combination_metadata[i]["injection_text"]
        
        injected_prompt = original_prompt + partial_generation + injection_text
        injected_prompts.append(injected_prompt)
        
        # Store initial generation
        combination_metadata[i]["initial_generation"] = generated_text
        combination_metadata[i]["injection_point"] = injection_point
    
    # Generate continuations after injection
    print("  Generating post-injection continuations...")
    # Use add_special_tokens=False to only add BOS, no chat template tokens
    injected_outputs = llm.generate(
        injected_prompts,
        SAMPLING_PARAMS,
        use_tqdm=False
    )
    
    # Process results
    print(f"\n[6/6] Processing results and running recovery verification...")
    results = []
    
    for i, (metadata, output) in enumerate(zip(combination_metadata, injected_outputs)):
        full_generation = output.outputs[0].text
        
        result = {
            **metadata,
            "injected_generation": full_generation,
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(combinations)} combinations...")
    
    # Run recovery verification in batch (judge model)
    print("  Running judge model verification...")
    recovery_prompts = [
        recovery_verification_prompt.format(
            question=r["question"],
            correct_answer=r["correct_answer"],
            generated_text=r["injected_generation"]
        )
        for r in results
    ]
    
    # Use add_special_tokens=False to only add BOS, no chat template tokens
    recovery_outputs = llm.generate(
        recovery_prompts,
        RECOVERY_SAMPLING_PARAMS,
        use_tqdm=False
    )
    
    for result, output in zip(results, recovery_outputs):
        judge_response = output.outputs[0].text.strip()
        judge_decision = extract_judge_decision(judge_response)
        result["judge_response"] = judge_response
        result["judge_decision"] = judge_decision
    
    # Save results
    print(f"\n[DONE] Saving results to {OUTPUT_PATH}...")
    output_data = {
        "experiment_config": {
            "n_questions": N_QUESTIONS,
            "k_prompts": len(question_prompts),
            "m_injections": len(injection_texts),
            "total_combinations": len(combinations),
            "model_path": str(MODEL_PATH),
            "timestamp": datetime.now().isoformat(),
            "question_prompt_names": [p["name"] for p in question_prompts],
            "injection_names": [inj["name"] for inj in injection_texts]
        },
        "results": results
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    total = len(results)
    print(f"Total combinations tested: {total}")
    
    # Analyze recovery success based on judge decisions
    successful_recoveries = sum(
        1 for r in results 
        if r.get("judge_decision") == "yes"
    )
    failed_recoveries = sum(
        1 for r in results 
        if r.get("judge_decision") == "no"
    )
    unclear_cases = sum(
        1 for r in results 
        if r.get("judge_decision") == "unclear"
    )
    
    print(f"\nJudge Evaluation Results:")
    print(f"  Correct (yes): {successful_recoveries}/{total} ({100*successful_recoveries/total:.1f}%)")
    print(f"  Incorrect (no): {failed_recoveries}/{total} ({100*failed_recoveries/total:.1f}%)")
    print(f"  Unclear: {unclear_cases}/{total} ({100*unclear_cases/total:.1f}%)")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("="*80)


if __name__ == "__main__":
    run_prompt_injection_experiment()

