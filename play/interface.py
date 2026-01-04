from dataclasses import dataclass
from typing import Callable, List

from play.dataset_loaders import aggregated_dataset_loader

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = None
    top_p: float = None
    take_dumb_max: bool = True
    seed: int = 42 # The seed used for the random number generator
    max_new_tokens: int = 100

@dataclass
class DataPoint:
    question_id: str
    question_contents: list[str]
    question_correct_answer: list[str]
    model_injection_position: int = 0 # The index of the first token of the injection in the response
    model_cot_upto_injection: list[str] # Substring of model_response up to the injection point
    model_cot_after_injection: list[str]
    model_response: list[str] # Full text of the model's response
    judge_response: list[str] # Full text of the judge model's response
    judge_decision: bool
    aha_moment_first_tokens: list[int]# The index of the first token of the aha moment in the response
    aha_moment_last_tokens: list[int]# The index of the last token of the aha moment in the response

@dataclass
class ModelGenerationConfig:
    model_name: str
    should_stop_fn: Callable[[List[int]], bool]
    get_injection_fn: Callable[[], str] 
    global_stop_fn: Callable[[List[int]], bool]
    question_prompt_template: list[str] # The prompt template used
    sampling_params: SamplingParams
@dataclass
class JudgeGenerationConfig:
    judge_name: str
    judge_prompt: list[str]
    sampling_params: SamplingParams
@dataclass
class Experiment:
    dataset: aggregated_dataset_loader
    model_generation_config: ModelGenerationConfig
    judge_generation_config: JudgeGenerationConfig
    datapoints: list[DataPoint]  # List of indices of datapoints to use from the dataset
    
    
   
