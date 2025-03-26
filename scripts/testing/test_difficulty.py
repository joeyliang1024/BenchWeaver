from BenchWeaver.eval.difficulty import compute_difficulty

if __name__ ==  "__main__":
    english_question = "What is the capital of France?"
    english_answer = "The capital of France is Paris."
    difficulty = compute_difficulty(english_question, english_answer, lang="en", details=True)
    print(f"Difficulty of English text: {difficulty}")  
    
    korean_question = "프랑스의 수도는 무엇인가요?"
    korean_answer = "프랑스의 수도는 파리입니다."
    difficulty = compute_difficulty(korean_question, korean_answer, lang="ko", details=True)
    print(f"Difficulty of Korean text: {difficulty}")
    
    chinese_question = "法国的首都是什么？"
    chinese_answer = "法国的首都巴黎。"
    difficulty = compute_difficulty(chinese_question, chinese_answer, lang="zh", details=True)
    print(f"Difficulty of Chinese text: {difficulty}")