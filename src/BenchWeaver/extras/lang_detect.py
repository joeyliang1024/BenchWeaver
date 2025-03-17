from langdetect import detect

def detect_language(text:str):
    """Detects the language of the given text."""
    try:
        return detect(text)
    except:
        return "unknown"