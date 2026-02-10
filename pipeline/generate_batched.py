from pipeline import prompts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
from interface import DataPoint as DataPoint, Experiment
from contextlib import nullcontext
from interface import GenerationMode
from transformer_lens import HookedTransformer
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
        self.model = HookedTransformer.from_pretrained(
            self.config.model_path,
            dtype=str(self.config.dtype),
            
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **kwargs,
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

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=self.gen).squeeze(-1)

    def unload_model(self):
        print("Memory in use before unloading model:", torch.cuda.memory_allocated() / 1e9, "GB")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        
        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Memory in use after unloading model and emptying cache:", torch.cuda.memory_allocated() / 1e9, "GB")

    def generate(
        self,
        batch_size: int,
        datapoints_callback: Optional[Callable[[List[DataPoint], int, int], None]] = None
    ):

        capturer = self.experiment.activation_capturer
        if capturer:
            capturer.bind(self.model)

        total_prompts = len(self.experiment.datapoints)
        for i in range(0, total_prompts, batch_size):
            batch_start_time = time.time()
            print(f"Processing batch {i//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size} (datapoints {i} - {min(i+batch_size, total_prompts)})")

            batch_datapoints = self.experiment.datapoints[i : i + batch_size]
            current_batch_size = len(batch_datapoints)
            batch_states = [BatchState() for _ in range(current_batch_size)]

            batch_prompts = [
                self.experiment.model_generation_config.question_prompt_template.format(
                    datapoint.question_contents
                )
                for datapoint in batch_datapoints
            ]
            for index, dp in enumerate(batch_datapoints):
                dp.question_contents = batch_prompts[index]
            
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
                ids_list = inputs.input_ids[index].tolist()
                dp.question_formatted_contents_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(ids_list[num_pad_tokens[index]:], skip_special_tokens=True)]


            input_ids = inputs.input_ids[:,:-1] # all but the last token, which is the trigger token
            attention_mask = inputs.attention_mask [:,:-1] # all but the last token
            
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            
            trigger_tokens = inputs.input_ids[:, -1] # the last token, which is the trigger token that prompts the model to start generating. we keep it separate because we want the computation column for it to be part of the generation activations
        
            with capturer.capturer(modes=[GenerationMode.QUESTION_PREFILL for _ in batch_datapoints], datapoints=batch_datapoints, num_pad_tokens=num_pad_tokens) if capturer else nullcontext():
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                        output_attentions=True if capturer else False,
                    )

            # if capturer: we do this in capturer from now on
            #     activations = capturer.captured_activations()
            #     for idx, dp in enumerate(batch_datapoints):
            #         if dp.should_capture_activations:
            #             padding_count = (attention_mask[idx] == 0).sum().item()
            #             dp.activations_question = {
            #                 layer_name: [activations_list[0][idx, :, padding_count:, padding_count:].clone()] if activations_list and activations_list[0] is not None else [None]
            #                 for layer_name, activations_list in activations.items()
            #             }
            #     capturer.kill_activations_array_reset_index()  # do NOT keeps indices

            past_key_values = outputs.past_key_values
            current_step = 0
            current_position_ids = attention_mask.long().sum(dim=1)  # start from the position ID of the trigger token
            
            # Update attention_mask to include the mask for the trigger_tokens (last token of prompt)
            attention_mask = torch.cat([attention_mask, inputs.attention_mask[:, -1].unsqueeze(1)], dim=1)
            next_token_logits : torch.Tensor = torch.empty((current_batch_size, self.model.config.vocab_size), device=self.device) # type:ignore
            while (
                current_step
                < self.experiment.model_generation_config.sampling_params.max_new_tokens
            ):
                with capturer.capturer(modes=[state_to_generation_mode(state) for state in batch_states], datapoints=batch_datapoints, num_pad_tokens=num_pad_tokens) if capturer else nullcontext():

                    with torch.no_grad():
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


                # if capturer: # we do this in capturer from now on
                #     for idx, datapoint in enumerate(batch_datapoints):
                #         if datapoint.should_capture_activations:
                #             token_activations = {}
                #             for layer_name, captured_data in capturer.captured_activations().items():
                #                 if len(captured_data) > 0:
                #                     data = captured_data[0] # removed after each forward pass, so always 0
                #                     token_activations[layer_name] = data[idx].clone() if data is not None else None

                #             target_dict = None
                #             if batch_states[idx].injection_start_position is None:
                #                 if datapoint.activations_upto_injection is None: datapoint.activations_upto_injection = {}
                #                 target_dict = datapoint.activations_upto_injection
                #             elif batch_states[idx].injection_end_position is None:
                #                 if datapoint.activations_injection is None: datapoint.activations_injection = {}    
                #                 target_dict = datapoint.activations_injection
                #             elif batch_states[idx].first_pad_position is None:
                #                 if datapoint.activations_after_injection is None: datapoint.activations_after_injection = {}
                #                 target_dict = datapoint.activations_after_injection

                #             if target_dict is not None:
                #                 for k, v in token_activations.items():
                #                     if k not in target_dict: target_dict[k] = []
                #                     target_dict[k].append(v)
                #     capturer.kill_activations_array_reset_index()

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
                    state.generated_tokens_strings.append(self.tokenizer.convert_ids_to_tokens(trigger_tokens[idx].item(), skip_special_tokens=True)[0].replace("Ġ", " ").replace("Ċ", "\n"))

                    # check if we need to inject or stop
                    if state.first_pad_position is None and self.experiment.model_generation_config.global_stop.should_stop(state.generated_tokens_strings):
                        state.first_pad_position = len(state.generated_tokens)
                        trigger_tokens[idx] = self.tokenizer.pad_token_id
                    elif state.injection_start_position is None and self.experiment.model_generation_config.should_stop.should_stop(state.generated_tokens_strings):
                        injection_text = self.experiment.model_generation_config.get_injection.get_injection(state.generated_tokens_strings)

                        injection_tokens = self.tokenizer(injection_text, add_special_tokens=False).input_ids

                        state.override_queue.extend(injection_tokens)
                        state.injection_start_position = len(state.generated_tokens)
                
                attention_mask = torch.cat([attention_mask, torch.tensor(new_attention_mask_column, device=self.device, dtype=attention_mask.dtype).unsqueeze(1)], dim=1)

                current_step += 1

            for idx in range(current_batch_size):
                state = batch_states[idx]

                batch_datapoints[idx].upto_injection_tokens = state.generated_tokens_strings[0:state.injection_start_position]
                batch_datapoints[idx].injection_tokens  = state.generated_tokens_strings[state.injection_start_position:state.injection_end_position]
                batch_datapoints[idx].after_injection_tokens = state.generated_tokens_strings[state.injection_end_position: state.first_pad_position]

                batch_datapoints[idx].upto_injection_string = self.tokenizer.decode(
                    state.generated_tokens[0:state.injection_start_position]
                )
                batch_datapoints[idx].injection = self.tokenizer.decode( # could also do this when populating the override queue, but the rest is here
                    state.generated_tokens[state.injection_start_position:state.injection_end_position]
                )
                batch_datapoints[idx].after_injection_string = self.tokenizer.decode(
                    state.generated_tokens[state.injection_end_position: state.first_pad_position]
                )
            
            self._print_activation_stats(batch_datapoints)
            
            del past_key_values, inputs, outputs, next_token_logits 

            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {batch_time:.2f} seconds")
            
            torch.cuda.empty_cache()
            if datapoints_callback is not None:
                datapoints_callback(batch_datapoints, i, i+current_batch_size)
            
            




#     final_next_input_ids = []

#     for idx in range(current_batch_size):
#         state = batch_states[idx]

#         # finished? pad
#         if state.is_finished:
#             final_next_input_ids.append(self.tokenizer.pad_token_id)
#             continue

#         # want to inject? force
#         if state.injection_queue and len(state.injection_queue) > 0:
#             forced_token = state.injection_queue.pop(0)
#             final_next_input_ids.append(forced_token)
#             state.injection_ids.append(forced_token)

#         # standard generation
#         else:
#             pred_token = self._sample_token(
#                 next_token_logits[idx : idx + 1],
#                 temperature=self.experiment.model_generation_config.sampling_params.temperature,
#                 top_k=self.experiment.model_generation_config.sampling_params.top_k,
#                 top_p=self.experiment.model_generation_config.sampling_params.top_p,
#                 take_dumb_max=self.experiment.model_generation_config.sampling_params.take_dumb_max,
#             )
#             pred_token = pred_token.item()
#             history_plus_candidate = all_generated_ids[idx] + [
#                 pred_token
#             ]
#             current_seq_tokens = self.tokenizer.convert_ids_to_tokens(
#                 history_plus_candidate, skip_special_tokens=True
#             )
#             if (
#                 not state.has_injected
#                 and self.experiment.model_generation_config.should_stop.should_stop(
#                     current_seq_tokens
#                 )
#             ):
#                 injection_text = self.experiment.model_generation_config.get_injection.get_injection(
#                     current_seq_tokens
#                 )

#                 # use add_special_tokens=False to avoid adding BOS/EOS inside sentence
#                 injection_tokens = self.tokenizer(
#                     injection_text, add_special_tokens=False
#                 ).input_ids

#                 if len(injection_tokens) <= 0:
#                     raise ValueError(
#                         "Injection text resulted in empty token list."
#                     )

#                 state.injection_queue.extend(injection_tokens)
#                 state.has_injected = True

#             final_next_input_ids.append(pred_token)

#             if state.has_injected:
#                 if len(state.injection_ids) > 0:
#                     state.after_injection_ids.append(pred_token)
#                 else:
#                     state.before_injection_ids.append(pred_token)
#             else:
#                 state.before_injection_ids.append(pred_token)

#         # check history + the token we just decided on
#         current_seq_history = self.tokenizer.convert_ids_to_tokens(
#             all_generated_ids[idx] + [final_next_input_ids[-1]],
#             skip_special_tokens=True,
#         )
#         if self.experiment.model_generation_config.global_stop_fn(
#             current_seq_history
#         ):
#             state.is_finished = True

#     # check if entire batch is finished
#     if all(s.is_finished for s in batch_states):
#         break

#     next_input_tensor = torch.tensor(
#         final_next_input_ids, device=self.device
#     ).unsqueeze(
#         1
#     )  # (batch, 1)

#     # update attention mask (append 1s for active, 0s for finished)
#     new_attention_column = torch.ones(
#         (current_batch_size, 1),
#         device=self.device,
#         dtype=attention_mask.dtype,
#     )
#     for idx in range(current_batch_size):
#         if batch_states[idx].is_finished:
#             new_attention_column[idx, 0] = 0

#     attention_mask = torch.cat(
#         [attention_mask, new_attention_column], dim=1
#     )

#     # Compute position_ids for the new token (based on cumulative attention mask)
#     next_position_ids = (
#         attention_mask.long().sum(dim=1, keepdim=True) - 1
#     )

#     ctx = capturer if capturer else nullcontext()
#     with ctx:
#         with torch.no_grad():
#             outputs = self.model(
#                 input_ids=next_input_tensor,
#                 past_key_values=past_key_values,
#                 attention_mask=attention_mask,
#                 position_ids=next_position_ids,
#                 use_cache=True,
#                 output_attentions=True if capturer else False,
#             )

#     if capturer:
#         for idx, dp in enumerate(batch_datapoints):
#             if (
#                 dp.should_capture_activations
#                 and not batch_states[idx].is_finished
#             ):
#                 token_activations = {}
#                 for (
#                     layer_name,
#                     captured_data,
#                 ) in capturer.captured_activations().items():
#                     if len(captured_data) > 0:
#                         data = captured_data[0]
#                         # During generation, seq_len is 1
#                         if data.shape[0] == current_batch_size:
#                             token_activations[layer_name] = data[
#                                 idx, 0
#                             ].clone()
#                         else:
#                             raise ValueError(
#                                 f"Unexpected captured data shape: {data.shape} when trying to populate datapoint activations."
#                             )
#                 dp.activations.append(token_activations)
#         capturer.clean_captured_activations()  # make sure to reset the capturer for next use. it DOESN'T remove indices from the arrays as we rely on them for accessing the current positions. only empties the tensors.

# past_key_values = outputs.past_key_values
# next_token_logits = outputs.logits[:, -1, :]

# # update history
# for idx in range(current_batch_size):
#     if not batch_states[idx].is_finished:
#         all_generated_ids[idx].append(next_input_tensor[idx].item())