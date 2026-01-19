import torch
import contextlib
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from play.interface import Experiment, ModelGenerationConfig, DataPoint, ActivationCapturer

# Assuming the previous dataclasses and Experiment class are defined above

class GenerateSimple:
    def __init__(self, experiment: Experiment,device:str):
        self.experiment = experiment
        self.config = experiment.model_generation_config
        self.device = device
        
        print(f"Loading model: {self.config.model_path} on {self.device}...")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )
        kwargs = {}
        if experiment.activation_capturer is not None:
            kwargs['attn_implementation']="eager" # Explicitly use eager to ensure weights can be captured
            
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            dtype=self.config.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **kwargs
        )
        
        if self.device == "cpu":
            self.model.to(self.device)
        
        self.model.eval()

    def _safe_model_call(self, **kwargs):
        try:
            return self.model(**kwargs)
        except RuntimeError as e:
            print('\n--- Model forward RuntimeError diagnostics ---')
            print('Error:', e)
            inp = kwargs.get('input_ids')
            print('input_ids:', type(inp), getattr(inp, 'shape', None))
            pkv = kwargs.get('past_key_values')
            if pkv is None:
                print('past_key_values: None')
            else:
                try:
                    print('past_key_values length:', len(pkv))
                    # print shapes for first two layers if possible
                    for i, layer_kv in enumerate(pkv[:2]):
                        shapes = []
                        try:
                            for arr in layer_kv:
                                shapes.append(getattr(arr, 'shape', str(type(arr))))
                        except Exception:
                            shapes = str(type(layer_kv))
                        print(f'  layer {i} kv shapes: {shapes}')
                except Exception as e2:
                    print('Could not introspect past_key_values:', e2)

            # Try to inspect first attention module parameters
            try:
                m = self.model
                layer = None
                if hasattr(m, 'model') and hasattr(m.model, 'layers'):
                    layer = m.model.layers[0]
                elif hasattr(m, 'layers'):
                    layer = m.layers[0]
                if layer is not None and hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    for name, param in attn.named_parameters():
                        print('attn param', name, getattr(param, 'shape', None))
            except Exception as e3:
                print('Failed to inspect model layers:', e3)

            print('--- End diagnostics ---\n')
            raise

    def unload_model(self):
        """
        Unload model weights and free GPU/CPU memory.
        """
        print("Unloading model...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("Model unloaded and memory cleared.")

    def generate(self):
        """
        Main execution loop. Iterates over all datapoints in the experiment.
        """
        total = len(self.experiment.datapoints)
        print(f"Starting generation for {total} datapoints.")
        
        for i, datapoint in enumerate(self.experiment.datapoints):
            if i % 10 == 0:
                print(f"Processing datapoint {i}/{total}...")
                
            self._generate_single_datapoint(datapoint)

    def _generate_single_datapoint(self, datapoint: DataPoint):
        """
        Generates response for a single datapoint token-by-token with injection logic.
        """

        # Set up activation capturing context
        capturer = self.experiment.activation_capturer
        should_capture = (capturer is not None) and datapoint.should_capture_activations
        ctx_manager = capturer if should_capture else contextlib.nullcontext()
        if should_capture:
            capturer.bind(self.model)




        formatted_prompt = self.config.question_prompt_template(datapoint.question_contents)
        
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        ids_list = input_ids[0].tolist()
        datapoint.question_formatted_contents_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(ids_list, skip_special_tokens=True)]

        # we need a_n-1 for the context, but the a_n is used to generate the first token, so we include it's activations in `upto_injection_activations`
        context_ids = input_ids[:, :-1] # All but last token
        trigger_token = input_ids[:, -1:] # first token for generation

        #our kv cache
        past_key_values = None
        
        # prefill, get kv cache
        with ctx_manager, torch.no_grad():
            outputs = self._safe_model_call(input_ids=context_ids, use_cache=True)
            past_key_values = outputs.past_key_values

        # save activations for question (prefill)
        if should_capture:
            # Create a deep copy of the activations structure and move to CPU/detach
            captured = capturer.captured_activations()
            datapoint.activations_question = {
                k: [t.detach().cpu() if t is not None else None for t in v] 
                for k, v in captured.items()
            }
            capturer.kill_activations_array_reset_index()

        #alias
        tokens_upto_injection = []
        # the last token from the prompt
        next_input_id = trigger_token

        # generate upto injection or end
        with ctx_manager, torch.no_grad():
            while (
                len(tokens_upto_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.should_stop_fn(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)) and # stop to inject (token strings)
                    not self.experiment.model_generation_config.global_stop_fn(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)) ): # global stop (token strings)
                    # generate next token
                    outputs = self._safe_model_call(
                        input_ids=next_input_id,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    # update cache
                    past_key_values = outputs.past_key_values

                    # sample
                    next_token = self._sample_token(outputs.logits[:, -1, :])
                    # save token id
                    tokens_upto_injection.append(next_token.item())

                    #update next token input
                    next_input_id = next_token.unsqueeze(-1)
        if should_capture:
            # capture all activations upto injection or end
            captured = capturer.captured_activations()
            datapoint.activations_upto_injection = {
                k: [t.detach().cpu() if t is not None else None for t in v] 
                for k, v in captured.items()
            }
            capturer.kill_activations_array_reset_index()
        
        datapoint.upto_injection_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(
            tokens_upto_injection, 
            skip_special_tokens=True
        )]

        if len(tokens_upto_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and not self.experiment.model_generation_config.global_stop_fn(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)): # we do need to inject, this is what broke the loop

            inject_text = self.experiment.model_generation_config.get_injection_fn(tokens_upto_injection)
            inject_tokens = self.tokenizer.encode(inject_text, return_tensors="pt").to(self.device)
            inject_tokens_list = inject_tokens[0].tolist()
            datapoint.injection_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(inject_tokens_list, skip_special_tokens=True)]

            
            context_ids = inject_tokens[:, :-1] # All but last token
            trigger_token = inject_tokens[:, -1:] # first token for generation
            with ctx_manager, torch.no_grad():
                outputs = self._safe_model_call(
                            input_ids=context_ids,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
            past_key_values = outputs.past_key_values

            if should_capture:
                captured = capturer.captured_activations()
                datapoint.activations_injection = {
                    k: [t.detach().cpu() if t is not None else None for t in v] 
                    for k, v in captured.items()
                }
                capturer.kill_activations_array_reset_index()


            tokens_after_injection = []

            next_input_id  = trigger_token
            with ctx_manager, torch.no_grad():
                while ( len(tokens_upto_injection+tokens_after_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.global_stop_fn(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection + tokens_after_injection, skip_special_tokens=False)) ): # global stop on combined sequence
                        outputs = self._safe_model_call(
                            input_ids=next_input_id,
                            past_key_values=past_key_values,
                            use_cache=True
                        )

                        past_key_values = outputs.past_key_values

                        next_token = self._sample_token(outputs.logits[:, -1, :])

                        tokens_after_injection.append(next_token.item())

                        next_input_id = next_token.unsqueeze(-1)
            if should_capture:
                captured = capturer.captured_activations()
                datapoint.activations_after_injection = {
                    k: [t.detach().cpu() if t is not None else None for t in v] 
                    for k, v in captured.items()
                }
                capturer.kill_activations_array_reset_index()


            
            datapoint.after_injection_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(
                tokens_after_injection, 
                skip_special_tokens=True
            )]


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
            logits[indices_to_remove] = float('-inf')
        
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
