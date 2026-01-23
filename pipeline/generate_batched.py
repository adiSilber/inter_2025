import os
import random
from numpy import ceil
import torch
import contextlib
import numpy as np
import sys
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.interface import (
    Experiment,
    ModelGenerationConfig,
    DataPoint,
    ActivationCapturer,
)


class GenerateBatched:
    def __init__(
        self, experiment: Experiment, device: str, deterministic: bool = False
    ):
        sys.stdout.reconfigure(line_buffering=True)
        self.experiment = experiment
        self.config = experiment.model_generation_config
        self.device = device
        if deterministic:
            print("Making generation deterministic. might mess up some kernels")
            self._make_deterministic()
        print(f"Loading model: {self.config.model_path} on {self.device}...")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        kwargs = {}
        if experiment.activation_capturer is not None:
            kwargs["attn_implementation"] = (
                "eager"  # Explicitly use eager to ensure weights can be captured
            )

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            dtype=self.config.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **kwargs,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    def _set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _make_deterministic(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def unload_model(self):
        print("Unloading model...")
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model unloaded and memory cleared.")

    def generate(self, batch_size: int = 8):
        self._set_seed(self.experiment.seed)
        total = len(self.experiment.datapoints)
        n_batches = int(ceil(total / batch_size))
        print(
            f"Starting batched generation for {total} datapoints in {n_batches} batches"
        )

        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        pad_token = self.tokenizer.pad_token_id 
        for i in range(n_batches):
            batch_datapoints = self.experiment.datapoints[
                i * batch_size : (i + 1) * batch_size
            ]
            print(
                f"Processing batch {i+1}/{n_batches} ({len(batch_datapoints)} datapoints)"
            )
            # Set up activation capturing context
            capturer = self.experiment.activation_capturer
            should_capture = capturer is not None
            ctx_manager = capturer if should_capture else contextlib.nullcontext()
            if should_capture:
                print("Binding activation capturer to model...")
                capturer.bind(self.model)

            formatted_prompts = [
                self.config.question_prompt_template(datapoint.question_contents)
                for datapoint in batch_datapoints
            ]
            formatted_prompts_input_id_list = [
                self.tokenizer(
                    formatted_prompt,
                    padding=False,
                    return_tensors="pt",
                )
                .input_ids.to("cpu")
                .squeeze(0)
                for formatted_prompt in formatted_prompts
            ]
            for datapoint, input_ids in zip(
                batch_datapoints, formatted_prompts_input_id_list
            ):
                datapoint.question_formatted_contents_tokenized = [
                    t.replace("Ġ", " ").replace("Ċ", "\n")
                    for t in self.tokenizer.convert_ids_to_tokens(
                        input_ids, skip_special_tokens=True
                    )
                ]
            print(
                f"  Prompts tokenized (avg {sum(len(ids) for ids in formatted_prompts_input_id_list) / len(formatted_prompts_input_id_list):.1f} tokens)"
            )

            should_inject = self.experiment.model_generation_config.should_stop_fn([])
            if should_inject:
                injections = [
                    self.experiment.model_generation_config.get_injection_fn(
                        datapoint.question_formatted_contents_tokenized
                    )
                    for datapoint in batch_datapoints
                ]

                injections_input_id_list = [
                    self.tokenizer(
                        injection,
                        padding=False,
                        truncation=True,
                        max_length=self.experiment.model_generation_config.sampling_params.max_new_tokens,
                        return_tensors="pt",
                    )
                    .input_ids.to("cpu")
                    .squeeze(0)
                    for injection in injections
                ]
                for datapoint, input_ids in zip(
                    batch_datapoints, injections_input_id_list
                ):
                    datapoint.injection_tokenized = [
                        t.replace("Ġ", " ").replace("Ċ", "\n")
                        for t in self.tokenizer.convert_ids_to_tokens(
                            input_ids, skip_special_tokens=True
                        )
                    ]
                print(
                    f"  Injections tokenized (avg {sum(len(ids) for ids in injections_input_id_list) / len(injections_input_id_list):.1f} tokens)"
                )
            merged_input_ids = [
                torch.concat([formatted_prompts_input_id, injection_input_id])
                for formatted_prompts_input_id, injection_input_id in zip(
                    formatted_prompts_input_id_list, injections_input_id_list
                )
            ] if should_inject else formatted_prompts_input_id_list
            # Manual left-padding
            input_max_len = max(len(ids) for ids in merged_input_ids)
            padded_input_ids_list = []
            num_left_pads = []
            for ids in merged_input_ids:
                pad_len = input_max_len - len(ids)
                if pad_len > 0:
                    padding = torch.full((pad_len,), pad_token, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([padding, ids])
                else:
                    padded_ids = ids
                num_left_pads.append(pad_len)

                padded_input_ids_list.append(padded_ids)
            
            merged_input_ids_tensor = torch.stack(padded_input_ids_list).to(self.device)


            print(f"  Starting model generation...")
            temperature = (
                self.experiment.model_generation_config.sampling_params.temperature
                if not self.experiment.model_generation_config.sampling_params.take_dumb_max
                else 0.0
            )

            prefill_inputs = merged_input_ids_tensor[:,:-1] # all but last token, last token is used for generating the first generated token
            trigger_tokens = merged_input_ids_tensor[:,-1:] # the last token in the input is the trigger token for generation
            
            attention_mask = (prefill_inputs != pad_token).long()
            do_sample = temperature > 0.0

            with ctx_manager, torch.no_grad():
                prefill = self.model(
                    prefill_inputs,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            print(f"  Prefill complete for batch {i+1}")

            if should_capture:
                activations_prefill = capturer.captured_activations()
            for index, datapoint in enumerate(batch_datapoints):
                if should_capture and datapoint.should_capture_activations:
                    datapoint.activations_question = {
                            k: v[0][index][num_left_pads[index]:].detach().cpu() if v[0] is not None else None # just one forward pass
                            
                            for k, v in activations_prefill.items()
                            }
            if should_capture:
                capturer.kill_activations_array_reset_index()



            attention_mask = (prefill_inputs != pad_token).long()

            trigger_mask_bit = torch.ones(
                (attention_mask.shape[0], 1), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )

            generation_attention_mask = torch.cat([attention_mask, trigger_mask_bit], dim=1)


            with ctx_manager, torch.no_grad():
                outputs = self.model.generate(
                    trigger_tokens,
                    attention_mask=generation_attention_mask,
                    max_new_tokens=self.experiment.model_generation_config.sampling_params.max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=self.experiment.model_generation_config.sampling_params.top_k,
                    top_p=self.experiment.model_generation_config.sampling_params.top_p,
                    pad_token_id=pad_token,
                    eos_token_id=self.tokenizer.eos_token_id,
                    past_key_values=prefill.past_key_values
                )
            
            batch_seq_len = generation_attention_mask.sum(dim=1) 
            trigger_position_ids = (batch_seq_len - 1).unsqueeze(-1) # Shape [Batch, 1]

            
            with ctx_manager, torch.no_grad():
                outputs = self.model.generate(
                    trigger_tokens,
                    attention_mask=generation_attention_mask,
                    position_ids=trigger_position_ids,
                    past_key_values=prefill.past_key_values,
                    max_new_tokens=self.experiment.model_generation_config.sampling_params.max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=self.experiment.model_generation_config.sampling_params.top_k,
                    top_p=self.experiment.model_generation_config.sampling_params.top_p,
                    pad_token_id=pad_token,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True 
                )
                
                
            print(f"  Generation complete for batch {i+1}")

            if should_capture:
                activations_generate = capturer.captured_activations()


            sequence_lengths = (outputs != self.tokenizer.pad_token_id).sum(dim=1)


            for index, datapoint in enumerate(batch_datapoints):
                # left pad
                generated_ids = outputs[index].tolist()[1:sequence_lengths[index]]
                datapoint.after_injection_tokens = [
                    t.replace("Ġ", " ").replace("Ċ", "\n")
                    for t in self.tokenizer.convert_ids_to_tokens(
                        generated_ids, skip_special_tokens=True
                    )
                ]
                print(f"  Datapoint {index}: generated {len(generated_ids)} tokens")

                if should_capture and datapoint.should_capture_activations:
                    num_non_padded_gen_steps = sequence_lengths[index] - 1
                    actvs = {
                        k: [
                            t[index].detach().cpu() if t is not None else None
                            for t in v[:num_non_padded_gen_steps]
                        ]
                        for k, v in activations_generate.items()
                    }
                    if should_inject:
                        datapoint.activations_after_injection = actvs
                    else:
                        datapoint.activations_upto_injection = actvs

            if should_capture:
                capturer.kill_activations_array_reset_index()


