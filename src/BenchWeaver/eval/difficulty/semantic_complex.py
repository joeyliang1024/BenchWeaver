from .utils import clean_text

def compute_semantic_complexity(question:str, answer:str, lang) -> int:
    """Computes semantic complexity based on unique token richness."""
    clean_answers = clean_text(answer, lang)
    clean_question = clean_text(question, lang)
    return len(set(clean_answers)-set(clean_question))
