from .utils import clean_text

def compute_semantic_complexity(text:str, lang) -> int:
    """Computes semantic complexity based on unique token richness."""
    clean_tokens = clean_text(text, lang)
    return len(set(clean_tokens))  # Semantic complexity is measured by unique token count
