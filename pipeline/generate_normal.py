import torch
import contextlib
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.interface import Experiment, ModelGenerationConfig, DataPoint, ActivationCapturer, GenerationMode

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
            kwargs['attn_implementation'] = "eager"
            kwargs['output_attentions'] = True # Ensure architecture supports returning weights
            
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            dtype=self.config.dtype,
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
            self.model.to(self.device)
        
        # # adisi check if its a good thing - cursor suggested
        # # Force attention output in config (some models need this)
        # if experiment.activation_capturer is not None:
        #     self.model.config.output_attentions = True
        # # adisi check if its a good thing - cursor suggested
        self.model.eval()

    def _safe_model_call(self, **kwargs):
        # # adisi check if its a good thing - cursor suggested
        # # Always pass output_attentions=True if we have a capturer
        # if self.experiment.activation_capturer is not None:
        #     kwargs['output_attentions'] = True
        # # adisi check if its a good thing - cursor suggested
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
        import gc
        print("Unloading model...")
        
        # Move model to CPU first to free GPU memory, then delete
        if hasattr(self, 'model') and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        if hasattr(self, 'gen'):
            del self.gen
            self.gen = None
        
        # Force garbage collection
        gc.collect()
        
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
        print(f"  Starting generation for datapoint: {datapoint.question_contents[:50]}...")

        # Set up activation capturing context
        capturer = self.experiment.activation_capturer
        should_capture = (capturer is not None) and datapoint.should_capture_activations
        ctx_manager = capturer if should_capture else contextlib.nullcontext()
        if should_capture:
            capturer.bind(self.model)

        formatted_prompt = self.config.question_prompt_template.format(datapoint.question_contents)
        
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        ids_list = input_ids[0].tolist()
        datapoint.question_formatted_contents_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(ids_list, skip_special_tokens=True)]
        print(f"    Prompt tokenized: {len(ids_list)} tokens")

        # we need a_n-1 for the context, but the a_n is used to generate the first token, so we include it's activations in `upto_injection_activations`
        context_ids = input_ids[:, :-1] # All but last token
        trigger_token = input_ids[:, -1:] # first token for generation

        #our kv cache
        past_key_values = None
        
        # prefill, get kv cache
        with ctx_manager.capturer(GenerationMode.QUESTION_PREFILL, [datapoint]), torch.no_grad():
            outputs = self._safe_model_call(input_ids=context_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        print(f"    Prefill complete, KV cache initialized")

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
        with ctx_manager.capturer(GenerationMode.UPTO_INJECTION,[datapoint]), torch.no_grad():
            while (
                len(tokens_upto_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.should_stop.should_stop(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)) and # stop to inject (token strings)
                    not self.experiment.model_generation_config.global_stop.should_stop(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)) ): # global stop (token strings)
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
                    
                    if len(tokens_upto_injection) % 20 == 0:
                        print(f"    Generated {len(tokens_upto_injection)} tokens so far...")

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
        datapoint.upto_injection_string = self.tokenizer.decode(tokens_upto_injection, skip_special_tokens=True)

        print(f"    Generated {len(tokens_upto_injection)} tokens before injection point")

        if len(tokens_upto_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and not self.experiment.model_generation_config.global_stop.should_stop(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False)): # we do need to inject, this is what broke the loop

            inject_text = self.experiment.model_generation_config.get_injection.get_injection(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection, skip_special_tokens=False), datapoint=datapoint)
            datapoint.injection_text = inject_text
            print(f"    Injecting text: '{inject_text[:100]}...'")
            inject_tokens = self.tokenizer.encode(inject_text, return_tensors="pt",add_special_tokens=False).to(self.device)
            inject_tokens_list = inject_tokens[0].tolist()
            datapoint.injection_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(inject_tokens_list, skip_special_tokens=True)]
            print(f"    Injection tokenized: {len(inject_tokens_list)} tokens")
            datapoint.injection = inject_text

            
            context_ids = inject_tokens[:, :-1] # All but last token
            trigger_token = inject_tokens[:, -1:] # first token for generation
            with ctx_manager.capturer(GenerationMode.INJECTION_PREFILL, [datapoint]), torch.no_grad():
                outputs = self._safe_model_call(
                            input_ids=context_ids,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
            past_key_values = outputs.past_key_values
            print(f"    Injection prefill complete")

            if should_capture:
                captured = capturer.captured_activations()
                datapoint.activations_injection = {
                    k: [t.detach().cpu() if t is not None else None for t in v] 
                    for k, v in captured.items()
                }
                capturer.kill_activations_array_reset_index()


            tokens_after_injection = []

            next_input_id  = trigger_token
            with ctx_manager.capturer(GenerationMode.AFTER_INJECTION,[datapoint], question_to_clip_indecies=range(len(datapoint.question_formatted_contents_tokenized))), torch.no_grad():
                while ( len(tokens_upto_injection+tokens_after_injection) < self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.global_stop.should_stop(self.tokenizer.convert_ids_to_tokens(tokens_upto_injection + tokens_after_injection, skip_special_tokens=False)) ): # global stop on combined sequence
                        outputs = self._safe_model_call(
                            input_ids=next_input_id,
                            past_key_values=past_key_values,
                            use_cache=True
                        )

                        past_key_values = outputs.past_key_values

                        next_token = self._sample_token(outputs.logits[:, -1, :])

                        tokens_after_injection.append(next_token.item())
                        
                        if len(tokens_after_injection) % 20 == 0:
                            print(f"    Generated {len(tokens_after_injection)} tokens after injection...")

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
            datapoint.after_injection_string = self.tokenizer.decode(tokens_after_injection, skip_special_tokens=True)
            print(f"    Generated {len(tokens_after_injection)} tokens after injection")
            print(f"    Total tokens generated: {len(tokens_upto_injection) + len(tokens_after_injection)}")


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
        return torch.multinomial(probs, num_samples=1,generator=self.gen).squeeze(-1)
