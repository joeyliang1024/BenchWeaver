from .utils import clean_text

def compute_syntax_complexity(text:str, lang) -> int:
    """Computes syntax complexity based on token count and sentence structure."""
    clean_tokens = clean_text(text, lang)
    return len(clean_tokens)  # Syntax complexity is measured by token count

