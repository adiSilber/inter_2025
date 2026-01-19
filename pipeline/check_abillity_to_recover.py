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

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from itertools import product
from datetime import datetime
import re
import uuid
from vllm import LLM, RequestOutput, SamplingParams
import torch
import uuid
from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
import torch
from typing import List, Callable, Optional
from vllm import LLM, SamplingParams

from pipeline.utils import extract_final_answer, extract_judge_decision

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


def should_inject(current_generation: str, question: str, num_tokens: int) -> bool:
    """
    Decide whether to inject at this point in the generation.
    Inject at the first end of a sentence after 70 tokens.
    """
    # Need at least 70 tokens generated
    if num_tokens < 70:
        return False
    
    # Inject at first sentence ending (after period, exclamation, or question mark)
    if current_generation.endswith(('.', '!', '?')):
        return True
    
    return False



# Type alias for the injection criteria function
InjectionCriteria = Callable[[str, str, int], bool]

class InjectionLogitsProcessor:
    """
    A stateful logits processor that monitors generation. 
    If the injection criteria is met, it forces an EOS token to stop generation.
    """
    def __init__(
        self, 
        tokenizer, 
        should_inject: InjectionCriteria, 
        prompt: str, 
        eos_token_id: int
    ):
        self.tokenizer = tokenizer
        self.should_inject = should_inject
        self.prompt = prompt
        self.eos_token_id = eos_token_id
        
        # State tracking
        self.injection_triggered = False
        self.stopped_at_token_len = 0

    def __call__(
        self, 
        prompt_token_ids: List[int], 
        generated_token_ids: List[int], 
        logits: torch.Tensor
    ) -> torch.Tensor:
        # 1. Optimization: Early exit if already triggered
        if self.injection_triggered:
            return self._force_eos(logits)

        # 2. Decode current generation to string
        # NOTE: Decoding at every step adds CPU overhead. 
        current_generation = self.tokenizer.decode(generated_token_ids)
        num_tokens = len(generated_token_ids)

        # 3. Check User Criteria
        if self.should_inject(current_generation, self.prompt, num_tokens):
            self.injection_triggered = True
            self.stopped_at_token_len = num_tokens
            return self._force_eos(logits)

        return logits

    def _force_eos(self, logits: torch.Tensor) -> torch.Tensor:
        """Sets all logits to -inf except the EOS token."""
        logits[:] = -float("inf")
        logits[self.eos_token_id] = 0.0
        return logits

# Criteria function signature
InjectionCriteria = Callable[[str, str, int], bool]

def generate_with_injection(
    llm: LLM, 
    prompt: str, 
    injection_text: str, 
    sampling_params: SamplingParams,
    should_inject: InjectionCriteria
) -> str:
    """
    Manually drives the vLLM engine to allow conditional stopping and injection.
    """
    
    # 1. Define a helper to run a single generation pass with monitoring
    def run_monitored_pass(current_prompt: str, is_first_pass: bool) -> str:
        request_id = f"req_{time.time()}"
        
        # Add request to the engine
        # We access the internal engine directly to control stepping
        llm.llm_engine.add_request(
            request_id, 
            current_prompt, 
            sampling_params
        )

        generated_text = ""
        
        # Drive the engine step-by-step
        while llm.llm_engine.has_unfinished_requests():
            # Performs one step of inference (one token for all active requests)
            step_outputs = llm.llm_engine.step()

            for output in step_outputs:
                if output.request_id != request_id:
                    continue

                # Get the newly generated text relative to this specific pass
                # VLLM outputs contain the full text, so we just read it.
                generated_text = output.outputs[0].text
                
                # If the request is finished normally (EOS token), return result
                if output.finished:
                    return generated_text

                # --- INTERVENTION POINT ---
                # Only check injection criteria during the first pass
                if is_first_pass:
                    # Current generation includes the prompt in some VLLM versions, 
                    # but usually output.outputs[0].text is just the completion.
                    # We pass the completion to your check function.
                    token_count = len(output.outputs[0].token_ids)
                    
                    if should_inject(generated_text, prompt, token_count):
                        # Criteria met! Stop this request immediately.
                        llm.llm_engine.abort_request(request_id)
                        
                        # Return the partial text + Injection Indicator (special flag)
                        return generated_text + "###INJECTION_TRIGGERED###"
        
        return generated_text

    # 2. Execute Pass 1 (Monitor for Criteria)
    pass_1_output = run_monitored_pass(prompt, is_first_pass=True)

    # 3. Check if Injection was triggered
    if "###INJECTION_TRIGGERED###" in pass_1_output:
        # Clean the flag
        partial_text = pass_1_output.replace("###INJECTION_TRIGGERED###", "")
        
        # Construct new prompt: Original + Partial + Injection
        combined_prompt = prompt + partial_text + injection_text
        
        # 4. Execute Pass 2 (Resume Generation)
        # We run this as a normal pass (is_first_pass=False) so we don't inject again.
        # Note: You can make this recursive by setting is_first_pass=True if needed.
        pass_2_output = run_monitored_pass(combined_prompt, is_first_pass=False)
        
        return partial_text + injection_text + pass_2_output
    
    else:
        # Injection never happened; return original result
        return pass_1_output





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
        max_model_len=4096,
        enable_prefix_caching=True
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
    print(f"\n[5/6] Running token-by-token generation with injection on {len(combinations)} combinations...")
    results = []
    
    for idx, (q_idx, p_idx, i_idx) in enumerate(combinations):
        question_data = questions[q_idx]
        question_text = question_data["question"]
        
        # Format the prompt using loaded template
        formatted_prompt = question_prompts[p_idx]["content"].format(question=question_text)
        injection_text = injection_texts[i_idx]["content"]
        
        # Generate with injection token-by-token
        full_generation, injection_point, pre_injection_text = generate_with_injection(
            llm, 
            formatted_prompt, 
            injection_text, 
            SAMPLING_PARAMS,
            should_inject=should_inject
        )
        
        result = {
            "question_idx": q_idx,
            "prompt_idx": p_idx,
            "injection_idx": i_idx,
            "question": question_text,
            "correct_answer": question_data["answer"],
            "prompt_name": question_prompts[p_idx]["name"],
            "injection_name": injection_texts[i_idx]["name"],
            "injection_text": injection_text,
            "full_generation": full_generation,
            "injection_point": injection_point,
            "pre_injection_text": pre_injection_text,
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(combinations)} combinations...")
    
    # Process results
    print(f"\n[6/6] Running recovery verification with judge model...")
    
    # Extract final answers (text after </think>) for judge evaluation
    recovery_prompts = []
    for r in results:
        # Extract only the final answer portion (after </think>)
        final_answer = extract_final_answer(r["full_generation"])
        r["final_answer_for_judge"] = final_answer
        
        # Create judge prompt with only the final answer
        recovery_prompts.append(
            recovery_verification_prompt.format(
                question=r["question"],
                correct_answer=r["correct_answer"],
                generated_text=final_answer  # Only the final answer, not the thinking
            )
        )
    
    # Use add_special_tokens=False to only add BOS, no chat template tokens
    recovery_outputs = llm.generate(
        recovery_prompts,
        RECOVERY_SAMPLING_PARAMS,
        use_tqdm=False
    )
    
    for result, output in zip(results, recovery_outputs):
        judge_response = output.outputs[0].text.strip()
        # Extract judge's final answer (after their thinking)
        judge_final_answer = extract_final_answer(judge_response)
        judge_decision = extract_judge_decision(judge_final_answer)
        
        result["judge_full_response"] = judge_response
        result["judge_final_answer"] = judge_final_answer
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
            "injection_names": [inj["name"] for inj in injection_texts],
            "injection_methodology": "end of sentence min 70 tokens"
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

