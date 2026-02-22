from pipeline import prompts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any, cast
from interface import ActivationCapturerV2, DataPoint as DataPoint, Experiment
from contextlib import nullcontext
from interface import GenerationMode
# from transformer_lens import HookedTransformer
import time
import gc

@dataclass
class BatchState:
    override_queue: List[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    generated_tokens_strings: list[str] = field(default_factory=list)
    injection_start_position: Optional[int] = None
    injection_end_position: Optional[int] = None
    first_pad_position: Optional[int] = None


def state_to_generation_mode(state: BatchState) -> GenerationMode:
    if state.injection_start_position is None:
        return GenerationMode.UPTO_INJECTION
    elif state.injection_end_position is None:
        return GenerationMode.INJECTION
    else:
        return GenerationMode.AFTER_INJECTION

class GenerateBatched:
    def __init__(self, experiment: Experiment, device: str = "cuda"):
        self.experiment = experiment
        self.config = experiment.model_generation_config
        self.device = device

        print(f"Loading model: {self.config.model_path} on {self.device}...")
        model_load_start = time.time()

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        kwargs = {}
        if experiment.activation_capturer is not None:
            kwargs["attn_implementation"] = "eager"
            kwargs["output_attentions"] = True


        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            torch_dtype=self.config.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **kwargs
        )

        seed = int(experiment.seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        if self.device == "cpu":
            self.model.to(self.device) # type:ignore

        self.model.eval()
        
        model_load_time = time.time() - model_load_start
        print(f"Model loaded in {model_load_time:.2f} seconds")

    def _print_activation_stats(self, batch_datapoints: List[DataPoint]):
        print(f"--- Activation Stats for {len(batch_datapoints)} datapoints ---")
        for idx, dp in enumerate(batch_datapoints):
            if not dp.should_capture_activations:
                continue
            print(f"DP {idx} ({dp.question_id}):")
            
            def print_stats(name, data):
                if data is None:
                    # print(f"  {name}: None")
                    return
                keys = list(data.keys())
                if not keys:
                    print(f"  {name}: Empty dict")
                    return
                print(f"  {name}: {len(keys)} layers")
                # Sample first layer
                first_key = keys[0]
                val = data[first_key]
                if isinstance(val, torch.Tensor):
                     print(f"    {first_key}: Tensor {val.shape}")
                elif isinstance(val, list):
                     print(f"    {first_key}: List[{len(val)}] | Item 0 type: {type(val[0]) if val else 'None'}")
                     if val and isinstance(val[0], torch.Tensor):
                         t = val[0]
                         print(f"      Item 0: Tensor {t.shape}")
                else:
                    print(f"    {first_key}: Type {type(val)}")

            print_stats("Question", dp.activations_question)
            print_stats("Upto Injection", dp.activations_upto_injection)
            print_stats("Injection", dp.activations_injection)
            print_stats("After Injection", dp.activations_after_injection)
        print("--------------------------------------------------")

    def _sample_token(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        config = self.experiment.model_generation_config.sampling_params

        if config.take_dumb_max or config.temperature == 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / config.temperature

        if config.top_k is not None and config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=self.gen).squeeze(-1)

    def unload_model(self):
        print("Memory in use before unloading model:", torch.cuda.memory_allocated() / 1e9, "GB")
        if hasattr(self, "model"):
            print("Deleting model...")
            del self.model
        if hasattr(self, "tokenizer"):
            print("Deleting tokenizer...")
            del self.tokenizer
        
        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            print("Emptying CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        print("Memory in use after unloading model and emptying cache:", torch.cuda.memory_allocated() / 1e9, "GB")

    def generate(
        self,
        batch_size: int,
        datapoints_callback: Optional[Callable[[List[DataPoint], int, int], None]] = None
    ):
        capturer: Optional[ActivationCapturerV2] = cast(ActivationCapturerV2, self.experiment.activation_capturer) if self.experiment.activation_capturer is not None else None
        print("capturer", capturer)
        
        generation_ctx = capturer.generation_context(self.model, experiment=self.experiment) if (capturer and hasattr(capturer, 'generation_context')) else nullcontext()
        with generation_ctx:
            total_prompts = len(self.experiment.datapoints)
            for i in range(0, total_prompts, batch_size):
                batch_start_time = time.time()
                print(f"Processing batch {i//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size} (datapoints {i} - {min(i+batch_size, total_prompts)})")
                batch_datapoints = self.experiment.datapoints[i : i + batch_size]

                self._generate_batch(batch_datapoints, capturer=capturer)
                

                batch_time = time.time() - batch_start_time
                print(f"Batch completed in {batch_time:.2f} seconds")
                
                torch.cuda.empty_cache()

                if datapoints_callback is not None:
                    datapoints_callback(batch_datapoints, i, i+len(batch_datapoints))
            
    def _generate_batch(self, batch_datapoints,  capturer: Optional[ActivationCapturerV2] = None):
        current_batch_size = len(batch_datapoints)
        batch_states = [BatchState() for _ in range(current_batch_size)]

        batch_prompts = [
            self.experiment.model_generation_config.question_prompt_template.format(
                datapoint.question_contents
            )
            for datapoint in batch_datapoints
        ]
        
        batch_prompts = [self.tokenizer.bos_token + prompt for prompt in batch_prompts]

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)


        num_pad_tokens = len(inputs.input_ids[0]) - inputs.attention_mask.long().sum(-1)

        for index, dp in enumerate(batch_datapoints):
            dp.pad_length = num_pad_tokens[index]
            ids_list = inputs.input_ids[index].tolist()
            dp.question_formatted_contents_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(ids_list[num_pad_tokens[index]:], skip_special_tokens=True)]


        max_new_tokens = self.experiment.model_generation_config.sampling_params.max_new_tokens
        input_len = inputs.input_ids.shape[1]

        # Preallocate attention mask to avoid O(N^2) dynamic allocation
        full_attention_mask = torch.zeros(
            (current_batch_size, input_len + max_new_tokens), 
            dtype=inputs.attention_mask.dtype, 
            device=self.device
        )
        full_attention_mask[:, :input_len] = inputs.attention_mask

        input_ids = inputs.input_ids[:,:-1] # all but the last token, which is the trigger token
        
        # Initial mask slices
        attention_mask = full_attention_mask[:, :input_len-1] # all but the last token
        
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        trigger_tokens = inputs.input_ids[:, -1] # the last token, which is the trigger token that prompts the model to start generating. we keep it separate because we want the computation column for it to be part of the generation activations
    
        # Use V2 context managers or old-style capturer
        with capturer.set_batch_context(datapoints=batch_datapoints, num_pad_tokens=num_pad_tokens) if capturer is not None else nullcontext():
            
            with torch.no_grad(), capturer.set_forward_context(modes=[GenerationMode.QUESTION_PREFILL for _ in batch_datapoints]) if capturer else nullcontext():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    output_attentions=True if capturer else False,
                )

            past_key_values = outputs.past_key_values
            current_step = 0
            current_position_ids = attention_mask.long().sum(dim=1)  # start from the position ID of the trigger token
            
            # Switch to mask including trigger token
            attention_mask = full_attention_mask[:, :input_len]
            
            next_token_logits : torch.Tensor = torch.empty((current_batch_size, self.model.config.vocab_size), device=self.device) # type:ignore
            while (
                current_step
                < max_new_tokens and any(state.first_pad_position is None for state in batch_states)
            ):
                with torch.no_grad(), capturer.set_forward_context(modes=[state_to_generation_mode(state) for state in batch_states]) if capturer else nullcontext():
                    outputs = self.model(
                        input_ids=trigger_tokens.unsqueeze(1),  # (batch, 1)
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        position_ids=current_position_ids.unsqueeze(1),  # (batch, 1)
                        use_cache=True,
                        output_attentions=True if capturer else False,
                    )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :] # shape (batch,seq_length,vocab_size)

                new_attention_mask_column = []
                for idx, state in enumerate(batch_states):
                
                    current_position_ids[idx] = current_position_ids[idx] + 1

                    # get the next trigger token
                    if state.first_pad_position is not None:
                        trigger_tokens[idx] = self.tokenizer.pad_token_id
                        new_attention_mask_column.append(0)
                    else:
                        new_attention_mask_column.append(1)
                        if len(state.override_queue)>0:
                            trigger_tokens[idx] = state.override_queue.pop(0)
                        else:
                            if state.injection_start_position is not None and state.injection_end_position is None:
                                state.injection_end_position = len(state.generated_tokens)
                            trigger_tokens[idx] = self._sample_token(next_token_logits[idx: idx + 1])

                    # update running totals
                    state.generated_tokens.append(trigger_tokens[idx].item())
                    state.generated_tokens_strings.append(self.tokenizer.convert_ids_to_tokens(trigger_tokens[idx].item(), skip_special_tokens=True).replace("Ġ", " ").replace("Ċ", "\n"))

                    # DEBUG: Check what's happening with period tokens
                    if state.generated_tokens_strings[-1].endswith('.') or state.generated_tokens_strings[-1].endswith('?') or state.generated_tokens_strings[-1].endswith('!'):
                        print(f"DEBUG idx={idx}: Sentence-end detected! Token: {repr(state.generated_tokens_strings[-1])}, total_tokens: {len(state.generated_tokens_strings)}, injection_start: {state.injection_start_position}, first_pad: {state.first_pad_position}")

                    # check if we need to inject or stop
                    if state.first_pad_position is None and self.experiment.model_generation_config.global_stop.should_stop(state.generated_tokens_strings):
                        state.first_pad_position = len(state.generated_tokens)
                        trigger_tokens[idx] = self.tokenizer.pad_token_id
                    elif state.injection_start_position is None and self.experiment.model_generation_config.should_stop.should_stop(state.generated_tokens_strings):
                        injection_text = self.experiment.model_generation_config.get_injection.get_injection(state.generated_tokens_strings)

                        injection_tokens = self.tokenizer(injection_text, add_special_tokens=False).input_ids

                        state.override_queue.extend(injection_tokens)
                        state.injection_start_position = len(state.generated_tokens)
                
                full_attention_mask[:, input_len + current_step] = torch.tensor(new_attention_mask_column, device=self.device, dtype=attention_mask.dtype)
                attention_mask = full_attention_mask[:, :input_len + current_step + 1]

                current_step += 1

            for idx in range(current_batch_size):
                state = batch_states[idx]

                start_pos = state.injection_start_position
                end_pos = state.injection_end_position
                batch_datapoints[idx].upto_injection_tokens = state.generated_tokens_strings[0:start_pos]


                batch_datapoints[idx].injection_tokens = (
                    state.generated_tokens_strings[start_pos:end_pos]
                    if start_pos is not None and end_pos is not None
                    else []
                )

                pad_pos = state.first_pad_position

                batch_datapoints[idx].after_injection_tokens = (
                    state.generated_tokens_strings[end_pos:pad_pos]
                    if end_pos is not None
                    else []
                )

                
                start_pos = state.injection_start_position
                end_pos = state.injection_end_position
                pad_pos = state.first_pad_position

                batch_datapoints[idx].upto_injection_string = self.tokenizer.decode(
                    state.generated_tokens[0:start_pos] if start_pos is not None else state.generated_tokens[0:pad_pos]
                )
                batch_datapoints[idx].injection = self.tokenizer.decode( 
                    state.generated_tokens[start_pos:end_pos] if start_pos is not None else []
                )
                batch_datapoints[idx].after_injection_string = self.tokenizer.decode(
                    state.generated_tokens[end_pos:pad_pos] if end_pos is not None else []
                )
            
        self._print_activation_stats(batch_datapoints)

        del past_key_values, inputs, outputs, next_token_logits 
