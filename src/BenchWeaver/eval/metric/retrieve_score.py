import re

def parse_numerical_score(s: str) -> float:
    score = -1.0
    m = re.search(r'score:\s*([\d.]+)', s, re.RegexFlag.I)
    if m is not None:
        score = float(m.group(1))
    else:
        t = s.replace('\n', '\\n')
        print(f'parse_comment(): {t}')
    return score

def parse_bool_score(s: str) -> str:
    text = text.lower()
    match = re.search(r'\b(true|false)\b', s)
    return match.group(0) if match else ""