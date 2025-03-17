from .syntax_complex import compute_syntax_complexity
from .semantic_complex import compute_semantic_complexity
from ...extras.lang_detect import detect_language

def compute_difficulty(text:str, lang=None) -> int:
    """Computes the overall difficulty by combining syntax and semantic complexity."""
    if lang is None:
        lang = detect_language(text)
    syntax_complexity = compute_syntax_complexity(text, lang)
    semantic_complexity = compute_semantic_complexity(text, lang)
    
    return syntax_complexity + semantic_complexity  # Overall difficulty is the sum of both complexities

if __name__ ==  "__main__":
    english_text = "This is a sample sentence."
    difficulty = compute_difficulty(english_text)
    print(f"Difficulty of text: {difficulty}")
    
    korean_text = "이것은 샘플 문장입니다."
    difficulty = compute_difficulty(korean_text)
    print(f"Difficulty of Korean text: {difficulty}")
    
    chinese_text = "这是一个示例句子。"
    difficulty = compute_difficulty(chinese_text)
    print(f"Difficulty of Chinese text: {difficulty}")