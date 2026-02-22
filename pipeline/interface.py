from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import torch
import dill
import io
from pathvalidate import sanitize_filename


class CPUUnpickler(dill.Unpickler):
    """Unpickler that forces all torch tensors to load on CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)
import gc
from contextlib import contextmanager

if TYPE_CHECKING:
    from dataset_loaders import aggregated_dataset_loader

class JudgeDecision(Enum):
    INCORRECT = 'incorrect'
    CORRECT = 'correct'
    NO_ANSWER = 'no_answer'
    IRRELEVANT = 'irrelevant'

class GenerationMode(Enum):
    QUESTION_PREFILL = "question"   
    INJECTION = "injection"
    UPTO_INJECTION = "upto_injection"
    AFTER_INJECTION = "after_injection" 

def get_unique_path(path: str) -> str:
    """Returns a unique path by appending _v1, _v2, etc., if the file already exists."""
    if not os.path.exists(path):
        return path
    
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        new_path = f"{base}_v{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

class ActivationCapturer(ABC):
    def __init__(self):
        self.activations: Dict[str, List[Optional[torch.Tensor]]] = {}
        self.model: Optional[torch.nn.Module] = None
        self.generation_mode: Optional[list[GenerationMode]] = None
        self.datapoints: Optional[list[DataPoint]] = None

    @abstractmethod
    def bind(self, model: torch.nn.Module,**kwargs):
        """Analyze the model and prepare hooks."""
        pass
    def captured_activations(self) -> Dict[str, List[Optional[torch.Tensor]]]:
        return self.activations
    
    def clean_captured_activations(self): # we never remove the indices from the arrays as we rely on them for accessing the current positions in the generators. we only empty the tensors.
        if torch.cuda.is_available():
            print("   Memory in use before cleaning activations arrays:", torch.cuda.memory_allocated() / 1e9, "GB")
        else:
            print("   Clearing activations arrays... (CUDA not available, so can't report memory usage)")
        for arr in self.activations.values():
            for i in range(len(arr)):
                arr[i] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("   Memory in use after cleaning activations arrays:", torch.cuda.memory_allocated() / 1e9, "GB")

    def kill_activations_array_reset_index(self):
        for key in self.activations:
            self.activations[key] = []
    def capturer(self, modes: list[GenerationMode], datapoints: list[DataPoint], **kwargs) -> ActivationCapturer:
        self.generation_mode = modes
        self.datapoints = datapoints
        return self 
    
    def __enter__(self):
        if self.generation_mode is None or self.datapoints is None:
            raise ValueError("you must get the context from '.capturer'")
        self.attach_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generation_mode = None
        self.datapoints = None  
        self.remove_hooks()
    
    @abstractmethod
    def attach_hooks(self) -> None:
        pass
    
    @abstractmethod
    def remove_hooks(self) -> None:
        pass


class ActivationCapturerV2(ActivationCapturer):
    """Generation-scoped activation capturer with hooks attached once per generation.
    
    Extends ActivationCapturer to support generation-level hook lifecycle where hooks
    are attached once at the start of generation and removed at the end, rather than
    per forward pass. Context data (datapoints, modes) is passed via context managers.
    """
    
    def __init__(self):
        super().__init__()
        self._batch_context_store = {}
        self._batch_context_keys = []
        self._forward_context_store = {}

    @abstractmethod
    def unbind(self):
        """Remove hooks after generation completes."""
        pass
    
    @contextmanager
    def generation_context(self, model: torch.nn.Module, **kwargs):
        """Context manager for generation lifecycle - binds on enter, unbinds on exit."""
        self.bind(model, **kwargs)
        print("called bind")
        try:
            yield self
        finally:
            print("called unbind")
            self.unbind()
    
    @contextmanager
    def set_batch_context(self, **kwargs):
        """Set batch-level context (datapoints, num_pad_tokens, etc.) for the hooks."""
        self._batch_context_keys = list(kwargs.keys())
        self._batch_context_store.update(kwargs)
        try:
            yield
        finally:
            for k in self._batch_context_keys:
                self._batch_context_store.pop(k, None)
            self._batch_context_keys = []

    @contextmanager
    def set_forward_context(self, **kwargs):
        """Set forward-pass context (modes, etc.) for the hooks."""
        self._forward_context_store.update(kwargs)
        try:
            yield
        finally:
            for k in kwargs:
                self._forward_context_store.pop(k, None)

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    take_dumb_max: bool = True
    max_new_tokens: int = 1000

class ShouldStop(ABC):
    @abstractmethod
    def should_stop(self, previous_tokens: Optional[list[str]]=None) -> bool:
        raise NotImplementedError
class DontStop(ShouldStop):
    def should_stop(self, previous_tokens: Optional[list[str]]=None) -> bool:
        return False

class ModelPromptTemplate(ABC):
    prompt_tokens_before_content: int
    prompt_tokens_after_content: int
    @abstractmethod
    def format(self, question: str) -> str:
        raise NotImplementedError
class JudgePromptTemplate(ABC):
    @abstractmethod
    def format(self, question: str,model_answer:str,correct_answer:str) -> str:
        raise NotImplementedError
class Injection(ABC):
    @abstractmethod
    def get_injection(self, previous_tokens: Optional[list[str]]=None, datapoint=None) -> str:
        raise NotImplementedError

@dataclass
class DataPoint:
    question_id: str
    question_correct_answer: str
    question_contents:str
    question_formatted_contents_tokenized: list[str] = field(default_factory=list)
    

    injection:str = ""
    injection_tokens: list[str] = field(default_factory=list)  # The injection text as a list of tokens


    upto_injection_tokens: list[str] = field(default_factory=list) # Substring of model_response up to the injection point
    after_injection_tokens: list[str] = field(default_factory=list)

    upto_injection_string: str = ""
    after_injection_string: str = ""


    judge_response: list[str] = field(default_factory=list)
    judge_decision: Optional[JudgeDecision] = None
    judge_prompt: list[str] = field(default_factory=list)

    aha_moment_first_tokens: list[int] = field(default_factory=list)# The index of the first token of the aha moment in the response
    aha_moment_last_tokens: list[int] = field(default_factory=list)# The index of the last token of the aha moment in the response
    
    recovery_starting_indices: list[int] = field(default_factory=list) # [start, end] tokens where model starts questioning injection
    recovery_complete_indices: list[int] = field(default_factory=list) # [start, end] tokens where model fully recovers
    
    recovery_token_index: int = -1  # Single token index where recovery starts (-1 = never recovered)
    meaningful_tokens_in_question: list[int] = field(default_factory=list)  # Indices of meaningful tokens in the question
    
    should_capture_activations: bool = False

    pad_length: int = 0
    activations_question: dict[str, List[Optional[torch.Tensor]]] = field(default_factory=dict) 
    activations_upto_injection: dict[str, List[Optional[torch.Tensor]]] = field(default_factory=dict) 
    activations_injection: dict[str, List[Optional[torch.Tensor]]] = field(default_factory=dict) 
    activations_after_injection: dict[str, List[Optional[torch.Tensor]]] = field(default_factory=dict) 

@dataclass
class ModelGenerationConfig:
    model_name: str
    model_path: str
    get_injection: Injection
    global_stop: ShouldStop
    question_prompt_template: ModelPromptTemplate
    sampling_params: SamplingParams
    should_stop: ShouldStop = DontStop()
    dtype: torch.dtype = torch.float16
@dataclass
class JudgeGenerationConfig:
    judge_name: str
    judge_model_path: str
    judge_prompt: JudgePromptTemplate
    sampling_params: SamplingParams
    dtype: torch.dtype = torch.float16


@dataclass
class Experiment:
    name: str
    dataset: aggregated_dataset_loader
    model_generation_config: ModelGenerationConfig
    judge_generation_config: Optional[JudgeGenerationConfig] = None
    datapoints: list[DataPoint] = field(default_factory=list)  # List of indices of datapoints to use from the dataset
    seed: int = 42
    activation_capturer: Optional[ActivationCapturer] = None
    runner_name: str = "unknown"
    wandb_run_id: Optional[str] = None  # Set by job 0 for log_wandb_judge_overall to resume and log overall metrics
    unique_id: Optional[str] = None  # 8-digit id for experiment/datapoint filenames; set by runner so all array jobs share it

    activation_head_clipping : Optional[dict[int,list[int]]] = None
    clip_max_val: float = 1e-6
    
    def populate_datapoints(self,num:Optional[int]=None):
        count = 0
        for batch in self.dataset:
            # Dataset iterator returns lists of question_items
            for question_item in batch:
                data_point = DataPoint(
                    question_id=question_item.question_id,
                    question_contents=question_item.q,
                    question_correct_answer=question_item.a,
                )
                self.datapoints.append(data_point)
                count += 1
                if num is not None and count >= num:
                    break
            if num is not None and count >= num:
                break
        self.dataset.reset_iterator()
    
    def store(self, save_dir:str,filename: Optional[str]=None, without_datapoints: bool = True,override: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            base = f"{self.name.replace(' ', '_')}"
            if self.unique_id:
                base = f"{self.unique_id}_{base}"
            filename = f"{base}_experiment.pkl"
        filename = sanitize_filename(filename)

        save_path = os.path.join(save_dir, filename)
        if override:
            save_path = save_path
        else:
            save_path = get_unique_path(save_path)
        
        temp_datapoints = self.datapoints 

        
        if without_datapoints:
            self.datapoints = []

        try:
            with open(save_path, "wb") as f:
                dill.dump(self, f)
        finally:
            if without_datapoints:
                self.datapoints = temp_datapoints

    def datapoints_to_cpu(self, start_index: int = 0, end_index: Optional[int] = None):
        if end_index is None:
            end_index = len(self.datapoints)
        if torch.cuda.is_available():
            print("moving activations to CPU, memory in use before moving activations to CPU:", torch.cuda.memory_allocated() / 1e9, "GB")
        else:
            print("moving activations to CPU... (CUDA not available, so can't report memory usage)")
        Experiment._datapoints_to_cpu(self.datapoints, start_index, end_index)
        print("   Memory in use after moving activations to CPU and cleaning activations arrays:", torch.cuda.memory_allocated() / 1e9, "GB")
    
    @classmethod
    def _datapoints_to_cpu(cls, datapoints: list[DataPoint], start_index: int = 0, end_index: Optional[int] = None):
        if end_index is None:
            end_index = len(datapoints)
        activation_attrs = (
            "activations_question",
            "activations_upto_injection",
            "activations_injection",
            "activations_after_injection",
        )
        for datapoint in datapoints[start_index:end_index]:
            for attr in activation_attrs:
                activation_dict = getattr(datapoint, attr)
                cpu_activation_dict = {k: [t.cpu() if t is not None else None for t in v] for k, v in activation_dict.items()}
                for k, v in cpu_activation_dict.items():
                    for i in range(len(v)):
                        v[i] = None
                del activation_dict
                setattr(
                    datapoint,
                    attr,
                    cpu_activation_dict,
                )
        gc.collect()
        if torch.cuda.is_available():
            print("emptying CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def store_datapoints_only(self, save_dir, filename: Optional[str]=None,start_index:int=0, end_index:Optional[int]=None,offset_relative_to_experiment=0,override: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        if end_index is None:
            end_index = len(self.datapoints)
        if filename is None:
            base = f"{self.name.replace(' ', '_')}"
            if self.unique_id:
                base = f"{self.unique_id}_{base}"
            filename = f"{base}_datapoints__{start_index+offset_relative_to_experiment}_{end_index+offset_relative_to_experiment}.pkl"
        filename = sanitize_filename(filename)

        save_path = os.path.join(save_dir, filename)
        if override:
            save_path = save_path
        else:
            save_path = get_unique_path(save_path)

        try:
            with open(save_path, "wb") as f:
                dill.dump((self.datapoints[start_index:end_index],start_index+offset_relative_to_experiment,end_index+offset_relative_to_experiment), f) 
        finally:
           pass
    def clear_activations(self,start_index,end_index):
        for datapoint in self.datapoints[start_index:end_index]:
            datapoint.activations_question = {}
            datapoint.activations_upto_injection = {}
            datapoint.activations_injection = {}
            datapoint.activations_after_injection = {}

    def store_datapoints_without_activations(self, save_dir: str, filename: Optional[str] = None):
        """Save all datapoints WITHOUT activations - creates a small, quick-to-load file.

        This is useful for analysis where you only need question/answer/judge data
        but not the heavy activation tensors.
        """
        import copy
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            base = f"{self.name.replace(' ', '_')}"
            if self.unique_id:
                base = f"{self.unique_id}_{base}"
            filename = f"{base}_datapoints_NO_ACTIVATIONS.pkl"
        filename = sanitize_filename(filename)
        save_path = os.path.join(save_dir, filename)

        # Create shallow copies of datapoints with cleared activations
        light_datapoints = []
        for dp in self.datapoints:
            # Shallow copy the datapoint
            light_dp = copy.copy(dp)
            # Clear activation fields (set to empty dicts, not copying the heavy tensors)
            light_dp.activations_question = {}
            light_dp.activations_upto_injection = {}
            light_dp.activations_injection = {}
            light_dp.activations_after_injection = {}
            light_datapoints.append(light_dp)

        with open(save_path, "wb") as f:
            dill.dump((light_datapoints, 0, len(light_datapoints)), f)

        print(f"   Saved {len(light_datapoints)} datapoints (no activations) to {save_path}")
        return save_path

    def load_datapoints(self, filepath: str, map_location: str | None = None):
        """Load datapoints from a pickle file.

        Args:
            filepath: Path to the pickle file
            map_location: Device to load tensors to. Use 'cpu' to avoid GPU OOM.
                         Default None loads to original device (GPU if saved on GPU).
        """
        with open(filepath, "rb") as f:
            if map_location == 'cpu':
                data = CPUUnpickler(f).load()
            else:
                data = dill.load(f)
        if isinstance(data, tuple) and len(data) == 3:
            datapoints, start, end = data
        elif isinstance(data, list):
            datapoints = data
            start, end = 0, len(datapoints)
        else:
            raise TypeError(f"Unexpected data type in file: {type(data)}")
        
        # Ensure the list is large enough to handle the slice assignment
        if len(self.datapoints) < end:
            self.datapoints.extend([None] * (end - len(self.datapoints))) # type: ignore
        self.datapoints[start:end] = datapoints

    @classmethod
    def load(cls, filepath: str) -> Experiment:
        with open(filepath, "rb") as f:
            experiment = dill.load(f)
        return experiment
    


