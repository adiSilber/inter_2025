import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pipeline.interface import Experiment, DataPoint, JudgeGenerationConfig
from typing import List, Optional
import re

class CorrectnessJudge:
    valid_answers = ['correct', 'incorrect', 'no_answer', 'irrelevant'] 
    def __init__(self, experiment: Experiment, device: str = "cuda"):
        """
        Initializes the judge based on the experiment's judge_generation_config.
        """
        self.experiment = experiment
        self.config = experiment.judge_generation_config
        self.device = device
        
        print(f"Loading judge model: {self.config.judge_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.judge_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.judge_model_path,
            torch_dtype=self.config.dtype,
            device_map=device
        )
        self.model.eval()

    def _extract_model_final_answer(self, response: str) -> Optional[str]:
        """
        Extracts only the final answer after the </think> tag.
        If no tag is found, returns None.
        """
        tag = "</think>"
        lower_response = response.lower()
        if tag in lower_response:
            pos = lower_response.find(tag)
            return response[pos + len(tag):].strip()
        return None

    def unload_model(self):
        """
        Unloads the model from memory.
        """
        del self.model
        torch.cuda.empty_cache()


    def _parse_judge_decision(self, judge_response: str) -> str:
        """
        Parses the judge's response to determine if the model's answer was correct.
        Looks for valid answer words after </think> tag.
        Returns the FIRST valid answer found, or None if undecided.
        """
        response_lower = judge_response.lower()
        
        # Look for </think> tag and take text AFTER it
        if "</think>" in response_lower:
            think_end_pos = response_lower.find("</think>")
            response_body = judge_response[think_end_pos + len("</think>"):].strip()
        else:
            response_body = judge_response.strip()
        
        # Find the FIRST valid answer in the response (by position)
        first_match = None
        first_pos = len(response_body)
        
        for valid_answer in self.valid_answers:
            match = re.search(r'\b' + re.escape(valid_answer) + r'\b', response_body, re.IGNORECASE)
            if match and match.start() < first_pos:
                first_pos = match.start()
                first_match = valid_answer
        
        return first_match

    def run(self, batch_size: int = 8, start_index: int = 0, end_index: Optional[int] = None):
        """
        Runs the judge on the experiment's datapoints within the specified range.
        Updates the datapoints in place with judge_response and judge_decision.
        """
        if not self.experiment.judge_generation_config:
            print("ERROR: No judge generation config found in the experiment.")
            return
        datapoints = self.experiment.datapoints
        if end_index is None:
            end_index = len(datapoints)
        
        target_datapoints = datapoints[start_index:end_index]
        
        for i in tqdm(range(0, len(target_datapoints), batch_size), desc="Running Judge"):
            batch = target_datapoints[i : i + batch_size]
            prompts = []
            
            for dp in batch:
                # We need to concanate in case there was no injection (generally more robust than selecting one of them)
                full_model_response = dp.upto_injection_string + dp.after_injection_string
                
                model_answer = self._extract_model_final_answer(full_model_response)
                
                prompt = self.config.judge_prompt.format(
                    question=dp.question_contents,
                    model_answer=model_answer,
                    correct_answer=dp.question_correct_answer
                )
                prompts.append(prompt)
                
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.sampling_params.max_new_tokens,
                    temperature=self.config.sampling_params.temperature,
                    top_p=self.config.sampling_params.top_p,
                    top_k=self.config.sampling_params.top_k,
                    do_sample=self.config.sampling_params.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract generated tokens (strip the prompt)
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_len:]
            
            # Decode and update datapoints
            for j, dp in enumerate(batch):
                judge_full_response = self.tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                
                # Record entire response as a list of string tokens
                dp.judge_response = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(generated_ids[j].tolist(), skip_special_tokens=True)]
                dp.judge_prompt = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in self.tokenizer.convert_ids_to_tokens(inputs.input_ids[j].tolist(), skip_special_tokens=True)]
                
                dp.judge_decision = self._parse_judge_decision(judge_full_response)
                
                print(f"  Datapoint {dp.question_id}: Decision = {dp.judge_decision}")
                print(f"  Judge thinking summary: {judge_full_response[:100]}...")


