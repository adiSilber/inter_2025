from dataclasses import dataclass


@dataclass
class InterpretabilityDataPoint:
    dataset_name: str
    question_id: str
    question: list[str]
    correct_answer: list[str]
    model_name: str
    model_prompt: list[str] # The prompt template used
    model_injection_prompt: list[str] # The full prompt sent to the model (including model_injection placeholder)
    model_injection: list[str]   # The prompt injection content used (Can be empty)
    model_injection_position: int = 0 # The index of the first token of the injection in the response
    model_response: list[str]
    model_response_thinking: list[str] # Substring of model_response that contains the model's "thinking"
    judge_model_name: str
    judge_model_prompt: list[str]
    judge_response: list[str] # Full text of the judge model's response
    judge_decision: bool
    aha_moment_first_tokens: list[int]# The index of the first token of the aha moment in the response
    aha_moment_last_tokens: list[int]# The index of the last token of the aha moment in the response
    seed: int = 42 # The seed used for the random number generator