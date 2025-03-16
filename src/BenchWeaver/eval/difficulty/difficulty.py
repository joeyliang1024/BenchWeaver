from .syntax_complex import compute_syntax_complexity
from .semantic_complex import compute_semantic_complexity

def compute_difficulty(text:str) -> int:
    """Computes the overall difficulty by combining syntax and semantic complexity."""
    syntax_complexity = compute_syntax_complexity(text)
    semantic_complexity = compute_semantic_complexity(text)
    
    return syntax_complexity + semantic_complexity  # Overall difficulty is the sum of both complexities
