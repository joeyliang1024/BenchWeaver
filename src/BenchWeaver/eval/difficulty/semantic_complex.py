from .utils import tokenize_text, clean_text
from ...extras.lang_detect import detect_language

def compute_semantic_complexity(text:str) -> int:
    """Computes semantic complexity based on unique token richness."""
    lang = detect_language(text)
    clean_txt = clean_text(text)
    return len(set(tokenize_text(clean_txt, lang)))  # Semantic complexity is measured by unique token count
