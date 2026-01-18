import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
from play.interface import DataPoint as DataPoint, Experiment
from contextlib import nullcontext

@dataclass
class BatchState:
    is_finished: bool = False
    injection_queue: List[int] = None # Queue of tokens waiting to be injected
    has_injected: bool = False
    before_injection_ids: List[int] = None
    injection_ids: List[int] = None
    after_injection_ids: List[int] = None


class InjectionGeneration:
    def __init__(self, experiment: Experiment, device: str = 'cuda'):
        self.device = device
        self.experiment = experiment
        
        print("Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(experiment.model_generation_config.model_path, padding_side='left')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model...", flush=True)
        # Using device_map=None and manual .to(device) to avoid accelerate hang
        self.model = AutoModelForCausalLM.from_pretrained(
            experiment.model_generation_config.model_path, 
            dtype=torch.float32,
            # device_map="cpu" if device == "cpu" else "auto",
            attn_implementation="eager"
        ).to(self.device)
        self.model.eval()
        print("Model loaded.", flush=True)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        take_dumb_max: bool = True
    ) -> torch.Tensor:
        if take_dumb_max or temperature == 0:
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
    ) -> Experiment:
        
        capturer = self.experiment.activation_capturer
        if capturer:
            capturer.bind(self.model)

        total_prompts = len(self.experiment.datapoints)
        for i in range(0, total_prompts, batch_size):

            batch_datapoints = self.experiment.datapoints[i : i + batch_size]
            current_batch_size = len(batch_datapoints)
            
            for dp in batch_datapoints:
                if dp.should_capture_activations:
                    dp.activations = []

            batch_prompts = [self.experiment.model_generation_config.question_prompt_template(datapoint.question_contents) for datapoint in batch_datapoints]
            
            # Ensure left padding for generation
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # Create correct position IDs for left-padded batch
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

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
            
            ctx = capturer if capturer else nullcontext()
            with ctx:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                        output_attentions=True if capturer else False
                    )
            
            if capturer:
                for idx, dp in enumerate(batch_datapoints):
                    if dp.should_capture_activations:
                        seq_len = input_ids.shape[1]
                        for t in range(seq_len):
                            # Skip padding tokens
                            if attention_mask[idx, t] == 0:
                                continue
                                
                            token_activations = {}
                            for layer_name, captured_data in capturer.captured_activations().items():
                                if len(captured_data) > 0:
                                    # Handle potential shape differences (e.g. attention matrices)
                                    data = captured_data[0]
                                    if data.shape[0] == current_batch_size:
                                        # Standard case: (batch, seq, ...)
                                        if t < data.shape[1]:
                                            token_activations[layer_name] = data[idx, t].clone()
                                    else:
                                        # Fallback or specific handling if needed
                                        pass
                            dp.activations.append(token_activations)
                        capturer.clean_captured_activations() # make sure to reset the capturer for next use 

            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            current_step = 0
            
            while current_step < self.experiment.model_generation_config.sampling_params.max_new_tokens:
                
                
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
                            temperature=self.experiment.model_generation_config.sampling_params.temperature,
                            top_k=self.experiment.model_generation_config.sampling_params.top_k,
                            top_p=self.experiment.model_generation_config.sampling_params.top_p,
                            take_dumb_max=self.experiment.model_generation_config.sampling_params.take_dumb_max
                        )
                        pred_token = pred_token.item()
                        history_plus_candidate = all_generated_ids[idx] + [pred_token]
                        current_seq_tokens = self.tokenizer.convert_ids_to_tokens(history_plus_candidate, skip_special_tokens=True)
                        if not state.has_injected and self.experiment.model_generation_config.should_stop_fn(current_seq_tokens):
                            injection_text = self.experiment.model_generation_config.get_injection_fn(current_seq_tokens)

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
                    current_seq_history = self.tokenizer.convert_ids_to_tokens(all_generated_ids[idx] + [final_next_input_ids[-1]], skip_special_tokens=True)
                    if self.experiment.model_generation_config.global_stop_fn(current_seq_history):
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
                
                # Compute position_ids for the new token (based on cumulative attention mask)
                next_position_ids = attention_mask.long().sum(dim=1, keepdim=True) - 1

                ctx = capturer if capturer else nullcontext()
                with ctx:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=next_input_tensor,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            position_ids=next_position_ids,
                            use_cache=True,
                            output_attentions=True if capturer else False
                        )
                
                if capturer:
                    for idx, dp in enumerate(batch_datapoints):
                        if dp.should_capture_activations and not batch_states[idx].is_finished:
                            token_activations = {}
                            for layer_name, captured_data in capturer.captured_activations().items():
                                if len(captured_data) > 0:
                                    data = captured_data[0]
                                    # During generation, seq_len is 1
                                    if data.shape[0] == current_batch_size:
                                        token_activations[layer_name] = data[idx, 0].clone()
                            dp.activations.append(token_activations)
                    capturer.clean_captured_activations() # make sure to reset the capturer for next use 

                
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                
                # update history
                for idx in range(current_batch_size):
                    if not batch_states[idx].is_finished:
                        all_generated_ids[idx].append(next_input_tensor[idx].item())
                
                current_step += 1
            
            for idx in range(current_batch_size):
                state = batch_states[idx]


                upto_injection_text = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(state.before_injection_ids, skip_special_tokens=True)]
                injection_text = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(state.injection_ids, skip_special_tokens=True)]
                after_injection_text = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(state.after_injection_ids, skip_special_tokens=True)]
                
                batch_datapoints[idx].model_cot_upto_injection = upto_injection_text
                batch_datapoints[idx].model_injection = injection_text
                batch_datapoints[idx].model_cot_after_injection = after_injection_text

            del past_key_values, inputs, outputs, next_token_logits
            torch.cuda.empty_cache()

        return self.experiment
