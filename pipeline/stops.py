from pipeline.interface import ShouldStop
from typing import Optional

class ImmediateStopCondition(ShouldStop):
    """Stop immediately so the injection is inserted before any model-generated tokens."""
    def should_stop(self, previous_tokens: Optional[list[str]] = None) -> bool:
        return True


class SentenceEndStopCondition(ShouldStop):
    """Stop when encountering sentence-ending punctuation after 20 tokens."""
    def should_stop(self, previous_tokens: Optional[list[str]] = None) -> bool:
        if previous_tokens is None or len(previous_tokens) < 20:
            return False
        if previous_tokens[-1] in {'.', '!', '?'} or previous_tokens[-1].endswith(('.', '!', '?')):
            return True
        return False


class EOSTokenStopCondition(ShouldStop):
    """Stop when encountering the end-of-sentence special token."""
    def should_stop(self, previous_tokens: Optional[list[str]] = None) -> bool:
        if previous_tokens == [] or previous_tokens is None:
            return False
        return previous_tokens[-1] == '<｜end▁of▁sentence｜>'