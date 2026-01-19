from pathlib import Path
from typing import Iterable
from datetime import datetime
from vllm import LLM, SamplingParams

from play.interface import DataPoint, Experiment
from play.utils import extract_judge_decision



def run_judge_validation(experiment: Experiment):
    """
    Run judge validation on the provided datapoints to verify correctness.
    """
    datapoints: list[DataPoint] = experiment.datapoints
    
    # Get judge configuration from experiment
    judge_config = experiment.judge_generation_config
    judge_prompt_template = judge_config.judge_prompt
    
    # Load model with vLLM for efficient inference
    llm = LLM(
        model=str(judge_config.judge_model_path),
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        skip_tokenizer_init=False,
        max_model_len=4096,
        enable_prefix_caching=True
    )
    
    # Build judge prompts
    texts = []
    for dp in datapoints:
        question = dp.question_contents
        correct_answer = dp.question_correct_answer
        model_response = ''.join(dp.model_response) if isinstance(dp.model_response, list) else dp.model_response
        
        prompt = judge_prompt_template(question, correct_answer, model_response)
        texts.append(prompt)

    # Convert SamplingParams dataclass to vLLM SamplingParams
    vllm_sampling_params = SamplingParams(
        temperature=judge_config.sampling_params.temperature,
        top_k=judge_config.sampling_params.top_k,
        top_p=judge_config.sampling_params.top_p,
        max_tokens=judge_config.sampling_params.max_new_tokens
    )
    
    judge_outputs = llm.generate(
        texts,
        vllm_sampling_params,
        use_tqdm=False
    )

    count_yes, count_no, count_unclear = 0, 0, 0
    for idx, (datapoint, output) in enumerate(zip(datapoints, judge_outputs)):
        judge_response_text = output.outputs[0].text.strip()
        # Extract judge's final answer (after their thinking)
        judge_decision_str = extract_judge_decision(judge_response_text)
        count_yes += 1 if judge_decision_str == 'yes' else 0
        count_no += 1 if judge_decision_str == 'no' else 0
        count_unclear += 1 if judge_decision_str == 'unclear' else 0

        # Update dataclass fields directly
        datapoint.judge_response = [judge_response_text] if isinstance(datapoint.judge_response, list) else judge_response_text
        datapoint.judge_decision = True if judge_decision_str == 'yes' else False