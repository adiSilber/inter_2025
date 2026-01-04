import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Callable, Optional

@dataclass
class BatchState:
    is_finished: bool = False
    injection_queue: List[int] = None # Queue of tokens waiting to be injected
    has_injected: bool = False
    before_injection_ids: List[int] = None
    injection_ids: List[int] = None
    after_injection_ids: List[int] = None

@dataclass 
class InjectedResult:
    prompt: str
    upto_injection: str
    injection:str
    after_injection: str


class InjectionGeneration:
    def __init__(self, model_name: str, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if "cuda" in device else torch.float32 
        ).to(self.device)
        self.model.eval()

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        if not do_sample or temperature == 0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature
        
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(
        self,
        batch_size: int, 
        should_stop_fn: Callable[[List[int]], bool], 
        prompts: List[str],
        get_injection_fn: Callable[[], str],         
        global_stop_fn: Callable[[List[int]], bool], 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> List[InjectedResult]:
        
        all_results = []
        total_prompts = len(prompts)

        for i in range(0, total_prompts, batch_size):

            batch_prompts = prompts[i : i + batch_size]
            current_batch_size = len(batch_prompts)
            
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            batch_states = [
                BatchState(
                    injection_queue=[],
                    before_injection_ids=[],
                    injection_ids=[],
                    after_injection_ids=[]
                ) 
                for _ in range(current_batch_size)
            ]
            
            all_generated_ids = input_ids.tolist()
            

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            current_step = 0
            
            while current_step < max_new_tokens:
                
                
                final_next_input_ids = []
                
                for idx in range(current_batch_size):
                    state = batch_states[idx]
                    
                    # finished? pad
                    if state.is_finished:
                        final_next_input_ids.append(self.tokenizer.pad_token_id)
                        continue
                    
                    # want to inject? force
                    if state.injection_queue and len(state.injection_queue) > 0:
                        forced_token = state.injection_queue.pop(0)
                        final_next_input_ids.append(forced_token)
                        state.injection_ids.append(forced_token)
                    
                    # standard generation
                    else:
                        pred_token = self._sample_token(
                            next_token_logits[idx:idx+1],
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=do_sample
                        )
                        pred_token = pred_token.item()
                        history_plus_candidate = all_generated_ids[idx] + [pred_token]
                        
                        if not state.has_injected and should_stop_fn(history_plus_candidate):
                            injection_text = get_injection_fn()

                            # use add_special_tokens=False to avoid adding BOS/EOS inside sentence
                            injection_tokens = self.tokenizer(injection_text, add_special_tokens=False).input_ids

                            if len(injection_tokens) <= 0:
                                raise ValueError("Injection text resulted in empty token list.")

                            state.injection_queue.extend(injection_tokens)
                            state.has_injected = True
                        
                        final_next_input_ids.append(pred_token)

                        if state.has_injected:
                            if len(state.injection_ids) > 0:
                                state.after_injection_ids.append(pred_token)
                            else:
                                state.before_injection_ids.append(pred_token)
                        else:
                            state.before_injection_ids.append(pred_token)

                    # check history + the token we just decided on
                    current_seq_history = all_generated_ids[idx] + [final_next_input_ids[-1]]
                    if global_stop_fn(current_seq_history):
                        state.is_finished = True
                
                # check if entire batch is finished
                if all(s.is_finished for s in batch_states):
                    break

                next_input_tensor = torch.tensor(
                    final_next_input_ids, device=self.device
                ).unsqueeze(1) # (batch, 1)
                
                # update attention mask (append 1s for active, 0s for finished)
                new_attention_column = torch.ones((current_batch_size, 1), device=self.device, dtype=attention_mask.dtype)
                for idx in range(current_batch_size):
                    if batch_states[idx].is_finished:
                        new_attention_column[idx, 0] = 0
                
                attention_mask = torch.cat([attention_mask, new_attention_column], dim=1)
                
                # forward pass (step)
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=next_input_tensor,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # update history
                for idx in range(current_batch_size):
                    if not batch_states[idx].is_finished:
                        all_generated_ids[idx].append(next_input_tensor[idx].item())
                
                current_step += 1
            
            batch_results = []
            for idx in range(current_batch_size):
                state = batch_states[idx]
                prompt_text = batch_prompts[idx]
                upto_injection_text = self.tokenizer.decode(state.before_injection_ids, skip_special_tokens=True)
                injection_text = self.tokenizer.decode(state.injection_ids, skip_special_tokens=True)
                after_injection_text = self.tokenizer.decode(state.after_injection_ids, skip_special_tokens=True)
                
                batch_results.append(InjectedResult(
                    prompt=prompt_text,
                    upto_injection=upto_injection_text,
                    injection=injection_text,
                    after_injection=after_injection_text
                ))

            all_results.extend(batch_results)

            del past_key_values, inputs, outputs, next_token_logits
            torch.cuda.empty_cache()

        return all_results
