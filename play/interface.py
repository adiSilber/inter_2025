from dataclasses import dataclass
from typing import Callable, List, Optional, Dict, Any
from abc import ABC, abstractmethod
import torch

from play.dataset_loaders import aggregated_dataset_loader

class ActivationCapturer(ABC):
    def __init__(self):
        self.activations: Dict[str, List[torch.Tensor]] = {}

    @abstractmethod
    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare hooks."""
        pass
    def captured_activations(self) -> Dict[str, List[torch.Tensor]]:
        return self.activations
    def clean_captured_activations(self): # we never remove the indices from the arrays as we rely on them for accessing the current positions in the generators. we only empty the tensors.
        for arr in self.activations.values():
            for i in range(len(arr)):
                arr[i] = None

    def kill_activations_array_reset_index(self):
        for key in self.activations:
            self.activations[key] = []
    @abstractmethod
    def __enter__(self):
        """Register hooks."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks."""
        pass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = None
    top_p: float = None
    take_dumb_max: bool = True
    max_new_tokens: int = 1000

@dataclass
class DataPoint:
    question_id: str
    question_correct_answer: str
    question_contents:str
    question_formatted_contents_tokenized: list[str]
    

    injection:str
    injection_tokenized: list[str]  # The injection text as a list of tokens


    upto_injection_tokens: list[str] # Substring of model_response up to the injection point
    after_injection_tokens: list[str]


    judge_response: list[str]
    judge_decision: bool

    aha_moment_first_tokens: list[int]# The index of the first token of the aha moment in the response
    aha_moment_last_tokens: list[int]# The index of the last token of the aha moment in the response
    should_capture_activations: bool = False


    activations_question: Optional[Dict[str, List[torch.Tensor]]] = None
    activations_upto_injection: Optional[Dict[str, List[torch.Tensor]]] = None
    activations_injection: Optional[Dict[str, List[torch.Tensor]]] = None
    activations_after_injection: Optional[Dict[str, List[torch.Tensor]]] = None

@dataclass
class ModelGenerationConfig:
    model_name: str
    model_path: str
    should_stop_fn: Callable[[List[str]], bool] = lambda:False
    get_injection_fn: Callable[[List[str]], str] 
    global_stop_fn: Callable[[List[str]], bool]
    question_prompt_template: Callable[[str], str]
    sampling_params: SamplingParams
@dataclass
class JudgeGenerationConfig:
    judge_name: str
    judge_model_path: str
    judge_prompt: Callable[[str, str, str], str]
    sampling_params: SamplingParams

@dataclass
class Experiment:
    dataset: aggregated_dataset_loader
    model_generation_config: ModelGenerationConfig
    judge_generation_config: JudgeGenerationConfig
    datapoints: list[DataPoint] = []  # List of indices of datapoints to use from the dataset
    seed: int = 42
    activation_capturer: Optional[ActivationCapturer] = None
    
    def populate_datapoints(self):
        for question_item in self.dataset:
            data_point = DataPoint(
                question_id=question_item.id,
                question_contents=question_item.q,
                question_contents_tokenized=[],
                injection_contents=[],
                question_correct_answer=question_item.a,
                model_injection=[],
                model_injection_position=0,
                model_cot_upto_injection=[],
                model_cot_after_injection=[],
                model_response=[],
                judge_response=[],
            )
            self.datapoints.append(data_point)
