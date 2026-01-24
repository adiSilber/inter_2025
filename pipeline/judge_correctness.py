import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pipeline.interface import Experiment, DataPoint, JudgeGenerationConfig
from typing import List, Optional
import re

class CorrectnessJudge:
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

    def _extract_final_answer(self, response: str) -> Optional[str]:
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

    def _parse_judge_decision(self, judge_full_response: str) -> Optional[bool]:
        """
        Looks for a 'yes' or 'no' answer in the judge's response,
        specifically after the judge has finished its own thinking process.
        """

        final_answer = self._extract_final_answer(judge_full_response)
        if final_answer is None:
            final_answer = judge_full_response
            
        decision_area = final_answer.lower()
            
        yes_matches = list(re.finditer(r'\byes\b', decision_area))
        no_matches = list(re.finditer(r'\bno\b', decision_area))
        
        all_matches = []
        for m in yes_matches:
            all_matches.append(('yes', m.start()))
        for m in no_matches:
            all_matches.append(('no', m.start()))
            
        if not all_matches:
            return None # Default to None if no clear yes/no found
            
        # Take the last occurrence of either 'yes' or 'no'
        all_matches.sort(key=lambda x: x[1])
        return all_matches[-1][0] == 'yes'
    def unload_model(self):
        """
        Unloads the model from memory.
        """
        del self.model
        torch.cuda.empty_cache()
    def run(self, batch_size: int = 8, start_index: int = 0, end_index: Optional[int] = None):
        """
        Runs the judge on the experiment's datapoints within the specified range.
        Updates the datapoints in place with judge_response and judge_decision.
        """
        datapoints = self.experiment.datapoints
        if end_index is None:
            end_index = len(datapoints)
        
        target_datapoints = datapoints[start_index:end_index]
        
        for i in tqdm(range(0, len(target_datapoints), batch_size), desc="Running Judge"):
            batch = target_datapoints[i : i + batch_size]
            prompts = []
            
            for dp in batch:
                # We need to concanate in case there was no injection (generally more robust than selecting one of them)
                full_model_response = "".join(dp.upto_injection_tokens) + "".join(dp.after_injection_tokens)
                
                model_answer = self._extract_final_answer(full_model_response)
                
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


