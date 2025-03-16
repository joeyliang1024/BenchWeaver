from .utils import tokenize_text, clean_text
from ...extras.lang_detect import detect_language

def compute_syntax_complexity(text:str) -> int:
    """Computes syntax complexity based on token count and sentence structure."""
    lang = detect_language(text)
    clean_txt = clean_text(text)
    tokens = tokenize_text(clean_txt, lang)
    
    return len(tokens)  # Syntax complexity is measured by token count

