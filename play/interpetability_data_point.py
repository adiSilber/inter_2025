@dataclass
class InterpretabilityDataPoint:
    dataset_name: str
    question_id: str
    question: str
    correct_answer: str
    model_name: str
    model_prompt: str # The prompt template used
    model_injection_prompt: str # The full prompt sent to the model (including model_injection placeholder)
    model_injection: str   # The prompt injection content used (Can be empty)
    model_injection_position: int = 0 # The index of the first token of the injection in the response
    model_response: str
    model_response_thinking: str # Substring of model_response that contains the model's "thinking"
    judge_model_name: str
    judge_model_prompt: str
    judge_response: str # Full text of the judge model's response
    judge_decision: bool
    aha_moment_finder_model_name: str
    aha_moment_finder_model_prompt: str
    aha_moment_finder_response: str # Full text of the aha_moment_finder model's response
    aha_moment_finder_first_token: int # The index of the first token of the aha moment in the response
    aha_moment_finder_last_token: int # The index of the last token of the aha moment in the response
    
