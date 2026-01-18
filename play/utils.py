from typing import List, Union
import re


def extract_final_answer(text: Union[str, List[str]]) -> str:
    """Extract text after </think> tag (the final answer portion).
    
    Args:
        text: The generated text that may contain <think>...</think> reasoning
        
    Returns:
        The text after the closing </think> tag, or the full text if no tag found.
    """
    if type(text) == list:
        text = "".join(text)

    # Look for the closing think tag (case-insensitive)
    think_close_patterns = r'</think>',
    
    # Find where thinking ends
    think_end_pos = text.lower().find(think_close_patterns)
    
    # Extract text after thinking
    if think_end_pos > 0:
        return text[think_end_pos:].strip()
    
    # No think tag found - return full text
    return text.strip()


def extract_judge_decision(text: Union[str, List[str]]) -> str:
    """Extract yes/no decision from judge's response.
    
    Looks for 'yes' or 'no' AFTER the </think> tag is generated.
    The judge should output yes/no after finishing their thinking process.
    """
    # Look for the closing think tag (case-insensitive)
    final_answer = extract_final_answer(text)

    answer_portion = final_answer[-500:].strip().lower()
    
    # Find all yes/no occurrences with word boundaries
    yes_matches = list(re.finditer(r'\byes\b', answer_portion))
    no_matches = list(re.finditer(r'\bno\b', answer_portion))

    # Collect all matches with their positions
    all_matches = []
    for match in yes_matches:
        all_matches.append(('yes', match.start()))
    for match in no_matches:
        all_matches.append(('no', match.start()))
    
    # Sort by position and take the last one
    if all_matches:
        all_matches.sort(key=lambda x: x[1])
        return all_matches[-1][0]
    
    # Fallback: check if the very last word is yes/no
    words = answer_portion.split()
    if words:
        last_word = words[-1].strip('.,!?;:\'"')
        if last_word in ['yes', 'no']:
            return last_word
    
    return "unclear"

