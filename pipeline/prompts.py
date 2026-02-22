
from pipeline.interface import ModelPromptTemplate
from pipeline.interface import JudgePromptTemplate

class ShortAnswerPromptTemplate(ModelPromptTemplate):
    prompt_tokens_before_content = 7
    prompt_tokens_after_content = 3
    """Format question with instruction to answer shortly."""
    def format(self, question: str) -> str:
        return f'<｜begin▁of▁sentence｜>Answer the question in short<｜User｜>{question}<｜Assistant｜><think>\n'


class MathEvaluatorJudgePrompt(JudgePromptTemplate):
    """Format judge prompt for problem evaluation."""
    def format(self, question: str, model_answer: str, correct_answer: str) -> str:
        return (f"""
<｜begin▁of▁sentence｜>'
You are an expert problem evaluator. Compare the student's answer with the correct answer.<｜User｜>"
Question: \"\"\"{question} \"\"\"


Correct Answer: \"\"\"{correct_answer} \"\"\"


Student's Response:  \"\"\" {model_answer} \"\"\"


1. Does the student's response correctly answer the question?"
2. Do they try to answer the question but have an incorrect answer?"
3. Do they fail to provide an answer at all?"
4. Or do they talk about a totaly different subject?
Respond with ONLY with:
'correct' for option 1.
'incorrect' for option 2.
'no_answer' for option 3.
'irrelevant' for option 4.
<｜Assistant｜><think>"
"""
)


class SimpleEvaluatorJudgePrompt(JudgePromptTemplate):
    """Format judge prompt for simple correct/incorrect evaluation.

    Marks as incorrect if the answer contains extra irrelevant information,
    even if the correct answer is present.
    """
    def format(self, question: str, model_answer: str, correct_answer: str) -> str:
        return (f"""
<｜begin▁of▁sentence｜>'
You are an expert problem evaluator. Compare the student's answer with the correct answer.<｜User｜>"
Question: \"\"\"{question} \"\"\"


Correct Answer: \"\"\"{correct_answer} \"\"\"


Student's Response:  \"\"\" {model_answer} \"\"\"


1. Does the student's response correctly answer the question AND ONLY the question (no extra unrelated information)?
2. Is the answer incorrect, missing, contains extra irrelevant information, or talks about a different subject?
Respond with ONLY:
'correct' for option 1.
'incorrect' for option 2.
<｜Assistant｜><think>"
"""
)
