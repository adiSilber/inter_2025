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
        
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, 
            dtype=self.config.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self.model.to(self.device)
        
        self.model.eval()

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
        
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        datapoint.question_formatted_contents_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)]

        # we need a_n-1 for the context, but the a_n is used to generate the first token, so we include it's activations in `upto_injection_activations`
        context_ids = input_ids[:, :-1] # All but last token
        trigger_token = input_ids[:, -1:] # first token for generation

        #our kv cache
        past_key_values = None
        
        # prefill, get kv cache
        with ctx_manager, torch.no_grad():
            outputs = self.model(input_ids=context_ids, use_cache=True)
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
                len(tokens_upto_injection) <= self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.should_stop_fn(tokens_upto_injection) and # stop to inject
                    not self.experiment.model_generation_config.global_stop_fn(tokens_upto_injection) ): # global stop
                    # generate next token
                    outputs = self.model(
                        input_ids=next_input_id,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    # update cache
                    past_key_values = outputs.past_key_values
                    
                    # sample
                    next_token = self._sample_token(outputs.logits[:, -1, :])
                    # save token
                    tokens_upto_injection.append(next_token.item())
                    
                    #update next token input
                    next_input_id = next_token
        if should_capture:
            # capture all activations upto injection or end
            captured = capturer.captured_activations()
            datapoint.activations_upto_injection = {
                k: [t.detach().cpu() if t is not None else None for t in v] 
                for k, v in captured.items()
            }
            capturer.kill_activations_array_reset_index()
        
        datapoint.upto_injection_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(
            torch.tensor([tokens_upto_injection]), 
            skip_special_tokens=True
        )]

        if len(tokens_upto_injection) <= self.experiment.model_generation_config.sampling_params.max_new_tokens and not self.experiment.model_generation_config.global_stop_fn(tokens_upto_injection): # we do need to inject, this is what broke the loop

            inject_text = self.experiment.model_generation_config.get_injection_fn(tokens_upto_injection)
            inject_tokens = self.tokenizer.encode(inject_text, return_tensors="pt").input_ids.to(self.device)
            datapoint.injection_tokenized = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(inject_tokens, skip_special_tokens=True)]

            
            context_ids = inject_tokens[:, :-1] # All but last token
            trigger_token = inject_tokens[:, -1:] # first token for generation
            with ctx_manager, torch.no_grad():
                outputs = self.model(
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
                while ( len(tokens_upto_injection+tokens_after_injection) <= self.experiment.model_generation_config.sampling_params.max_new_tokens and # max tokens
                    not self.experiment.model_generation_config.global_stop_fn(tokens_after_injection) ): # global stop
                        outputs = self.model(
                            input_ids=next_input_id,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        
                        past_key_values = outputs.past_key_values
                        
                            
                        next_token = self._sample_token(outputs.logits[:, -1, :])
                        
                        tokens_after_injection.append(next_token.item())
                        
                        next_input_id = next_token
            if should_capture:
                captured = capturer.captured_activations()
                datapoint.activations_after_injection = {
                    k: [t.detach().cpu() if t is not None else None for t in v] 
                    for k, v in captured.items()
                }
                capturer.kill_activations_array_reset_index()


            
            datapoint.after_injection_tokens = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(
                torch.tensor([tokens_after_injection]), 
                skip_special_tokens=True
            )]


                

    def _save_activations_to_datapoint(self, datapoint: DataPoint, capturer: ActivationCapturer):


        """
        Extracts activations from the capturer, moves them to CPU, and stores them 
        in the datapoint before the capturer cleans them.
        """
        captured_data = capturer.captured_activations()
        
        # We need to construct the list of dicts structure: List[Dict[layer_name, Tensor]]
        # The capturer returns Dict[layer_name, List[Tensor]]
        
        # Get the length of the sequence captured
        if not captured_data:
            return
            
        seq_len = len(next(iter(captured_data.values())))
        
        activations_list = []
        for i in range(seq_len):
            step_activations = {}
            for layer_name, tensor_list in captured_data.items():
                tensor = tensor_list[i]
                if tensor is not None:
                    # Move to CPU to save GPU memory and detach from graph
                    step_activations[layer_name] = tensor.cpu().detach()
                else:
                    step_activations[layer_name] = None
            activations_list.append(step_activations)
            
        datapoint.activations = activations_list
        
        # Clean the capturer for the next pass
        capturer.clean_captured_activations()

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


def generate_with_manual_kv_cache(
    model_name: str, 
    prompt_text: str, 
    max_new_tokens: int = 10
):
    # 1. Setup
    generated_sequence = input_ids 

    # 3. Prefill Step
    # We pass the full prompt to get the initial KV cache.
    # use_cache=True is default for generation, but explicit here for clarity.
    

    print(f"Initial prompt: '{prompt_text}'")
    print("Generating...")

    # 4. Decoding Loop (Autoregressive)
    for i in range(max_new_tokens):
        with torch.no_grad():
            # STRICT REQUIREMENT:
            # When passing past_key_values, input_ids must ONLY be the last token.
            # Shape of next_token: [batch_size, 1]
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Update the cache for the next iteration
            past_key_values = outputs.past_key_values
            
            # Get logits. Since input was 1 token, logits shape is [batch, 1, vocab]
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decode
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to full sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
            
            # Optional: Decode and print current step for transparency
            decoded_token = tokenizer.decode(next_token[0])
            print(f"Step {i+1}: {decoded_token}")

    # 5. Final Output
    full_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    print("\n--- Final Generated Text ---")
    print(full_text)

# Run the function
if __name__ == "__main__":
    generate_with_manual_kv_cache("gpt2", "The logic of computer science is")