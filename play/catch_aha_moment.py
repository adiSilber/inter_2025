from play.interpetability_data_point import InterpretabilityDataPoint


def _get_aha_patterns() -> list[str]:
    """Return list of aha moment patterns to search for."""
    return [
        'doesn\'t make sense', 'not correct', 'won\'t work', 'not right',
        'i made a mistake', 'think differently', 'doesn\'t work', 'start over', 'rethink', \
        'reconsider', 'incorrect', 'mistake', 'not right', 'actually', 'wrong', 'wait'
    ]


def _find_all_occurrences(text: str, pattern: str) -> list[int]:
    """Find all character positions where pattern occurs in text."""
    occurrences = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + 1
    return occurrences


def _char_pos_to_token_indices(char_pos: int, pattern_length: int, token_texts: list[str]) -> tuple[int, int]:
    """Map character position to token indices (first and last token of pattern)."""
    char_pos_current = 0
    first_idx = None
    last_idx = None
    
    for i, token in enumerate(token_texts):
        token_start = char_pos_current
        token_end = char_pos_current + len(token)
        
        if token_start <= char_pos < token_end:
            first_idx = i
        
        if first_idx is not None and token_start < char_pos + pattern_length <= token_end:
            last_idx = i
            break
        
        char_pos_current = token_end
    
    return first_idx, last_idx


def catch_aha_moment(data_point: InterpretabilityDataPoint) -> None:
    """Detect all aha moments in the thinking process and set indices in data_point."""
    tokens = data_point.model_response_thinking
    text = ''.join(tokens).lower()
    token_texts = [t.lower() for t in tokens]
    
    first_tokens = []
    last_tokens = []
    
    for pattern in _get_aha_patterns():
        pattern_lower = pattern.lower()
        occurrences = _find_all_occurrences(text, pattern_lower)
        
        for char_pos in occurrences:
            first_idx, last_idx = _char_pos_to_token_indices(char_pos, len(pattern_lower), token_texts)
            
            if first_idx is not None:
                first_tokens.append(first_idx)
                last_tokens.append(last_idx if last_idx is not None else first_idx)
    
    data_point.aha_moment_first_tokens = first_tokens
    data_point.aha_moment_last_tokens = last_tokens