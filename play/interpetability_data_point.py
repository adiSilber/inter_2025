from dataclasses import dataclass


@dataclass
class DataPoint:
    question_dataset_name: str
    question_id: str
    question_contents: list[str]
    question_correct_answer: list[str]
    model_name: str
    model_prompt: list[str] # The prompt template used
    model_injection: list[str]   # The prompt injection content used (Can be empty)
    model_injection_methodology :str 
    model_injection_position: int = 0 # The index of the first token of the injection in the response
    model_cot_upto_injection: list[str] # Substring of model_response up to the injection point
    model_cot_after_injection: list[str]
    model_response: list[str] # Full text of the model's response
    judge_model_name: str
    judge_model_prompt: list[str]
    judge_response: list[str] # Full text of the judge model's response
    judge_decision: bool
    aha_moment_first_tokens: list[int]# The index of the first token of the aha moment in the response
    aha_moment_last_tokens: list[int]# The index of the last token of the aha moment in the response
    seed: int = 42 # The seed used for the random number generator