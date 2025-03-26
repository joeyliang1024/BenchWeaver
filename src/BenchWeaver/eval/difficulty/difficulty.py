from typing import Dict
from .syntax_complex import compute_syntax_complexity
from .semantic_complex import compute_semantic_complexity
from ...extras.lang_detect import detect_language

def compute_difficulty(question:str, answer:str, lang=None, details: bool = False) -> Dict[str, int]|int:
    """
    Computes the overall difficulty of a question-answer pair by combining syntactic and semantic complexity.

    This function analyzes the difficulty of an answer by evaluating two key aspects:
    1. Syntax Complexity: The structural and grammatical complexity of the answer
    2. Semantic Complexity: The conceptual depth and meaning of the answer in relation to the question

    Parameters:
    
        question : str
        The original question being answered. Used to assess semantic relevance.
    
        answer : str
        The provided answer to the question. Used to compute both syntax and semantic complexity.
    
        lang : str, optional
        The language of the answer. If not provided, the language is automatically detected 
        using a language detection function. Defaults to None.
    
        details : bool, optional
        If True, returns a dictionary with individual complexity scores and total difficulty.
        If False, returns the total difficulty as an integer. Defaults to False.

    Returns:
        Union[Dict[str, int], int]
        - If details=False: Returns the total difficulty as an integer
        - If details=True: Returns a dictionary with the following keys:
            * 'syntax_complexity': Complexity score based on answer's syntax
            * 'semantic_complexity': Complexity score based on answer's semantic depth
            * 'difficulty': Total difficulty (sum of syntax and semantic complexities)
    """
    
    if lang is None:
        lang = detect_language(answer)
    syntax_complexity = compute_syntax_complexity(answer, lang)
    semantic_complexity = compute_semantic_complexity(question, answer, lang)
    if details:
        return {
            "syntax_complexity": syntax_complexity,
            "semantic_complexity": semantic_complexity,
            "difficulty": syntax_complexity + semantic_complexity
        }
    return syntax_complexity + semantic_complexity  # Overall difficulty is the sum of both complexities

if __name__ ==  "__main__":
    # Example usage
    english_question = "What is the capital of France?"
    english_answer = "The capital of France is Paris."
    difficulty = compute_difficulty(english_question, english_answer, lang="en")
    print(f"Difficulty of English text: {difficulty}")  
    
    korean_question = "프랑스의 수도는 무엇인가요?"
    korean_answer = "프랑스의 수도는 파리입니다."
    difficulty = compute_difficulty(korean_question, korean_answer, lang="ko")
    print(f"Difficulty of Korean text: {difficulty}")
    
    chinese_question = "法国的首都是什么？"
    chinese_answer = "法国的首都巴黎。"
    difficulty = compute_difficulty(chinese_question, chinese_answer, lang="zh")
    print(f"Difficulty of Chinese text: {difficulty}")
