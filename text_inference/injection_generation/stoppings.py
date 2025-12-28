from typing import List, Set
import torch
from vllm import LLM, SamplingParams


class SequenceStopper:
    def should_stop(self, prompt_ids: List[int], generated_ids: List[int], logits: torch.Tensor) -> bool:
        raise NotImplementedError("Subclasses should implement this method.")
    def get_name(self) -> str:
        return self.__class__.__name__
  
class LogitsSequenceStopper:
    """
    A LogitsProcessor that enforces a minimum token count and then stops 
    at the next sentence delimiter.
    """
    def __init__(self, tokenizer, sequence_stopper: SequenceStopper):
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size
        self.sequence_stopper = sequence_stopper
    

    def __call__(self, prompt_ids: List[int], generated_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if self.sequence_stopper.should_stop(prompt_ids, generated_ids, logits):
            # Mask all tokens to -inf
            logits[:] = -float('inf')
            # Set EOS to 0 (effectively probability 1.0 after softmax)
            logits[self.eos_token_id] = 0.0
        
        return logits
          
class EndOfSenSequenceStopper(SequenceStopper):
    """
    A LogitsProcessor that stops generation when the last generated token is a sentence delimiter.
    """
    def __init__(self, tokenizer, delimiter_chars: Set[str] = {'.', '?', '!'}, min_tokens: int = 70):
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size
        self.min_tokens = min_tokens
        # Identify all tokens that represent sentence delimiters.
        self.delimiter_token_ids = set()
        
        for i in range(self.vocab_size):
            token_str = tokenizer.decode([i])
            if token_str.strip() in delimiter_chars:
                self.delimiter_token_ids.add(i)


    def should_stop(self, prompt_ids: List[int], generated_ids: List[int], logits: torch.Tensor) -> bool:
        if len(generated_ids) >= self.min_tokens and generated_ids[-1] in self.delimiter_token_ids:
            return True
        return False