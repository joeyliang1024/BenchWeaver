import re

def parse_numerical_score(text: str) -> float:
    score = -1.0
    regex_patterns = [
        r'(?:score|分數)\s*[:：]?\s*([\d.]+)',
        r'rating:\s*\[\[([\d.]+)\]\]$'
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                break
            except ValueError:
                continue  # Skip to next pattern if conversion fails
    else:
        # This block runs if no break occurred (i.e., no match)
        t = text.replace('\n', '\\n')
        print(f'parse_numerical_score(): {t}')
        
    return score


def parse_bool_score(text: str) -> str:
    '''
    Normally for MCQA checking. The answer is either true, false or unknown.
    '''
    text = text.lower()
    match = re.search(r'\b(true|false|unknown)\b', text)
    return match.group(0) if match else ""