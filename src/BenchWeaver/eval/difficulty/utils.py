import re
import jieba
import nltk
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ...extras.lang_detect import detect_language

nltk.download('punkt')
nltk.download('stopwords')

okt = Okt()
lemmatizer = WordNetLemmatizer()

def tokenize_text(text):
    """Tokenizes text based on detected language."""
    lang = detect_language(text)
    if lang == "en":
        words = nltk.word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        return [word for word in words if word not in stopwords.words("english")]
    elif lang in ["zh-cn", "zh-tw"]:
        return list(jieba.cut(text))
    elif lang == "ko":
        return okt.morphs(text)
    else:
        return text.split()

def clean_text(text):
    """Cleans and tokenizes text based on language (English, Chinese, Korean)."""
    lang = detect_language(text)  # Detect the language of the text
    text = text.strip()

    # Remove HTML tags and URLs
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)

    if lang == "en":
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if not word.isdigit()]  # Remove numbers
        return words

    elif lang in ["zh-cn", "zh-tw"]:
        text = re.sub(r'[^\u4e00-\u9fff]', '', text)  # Keep only Chinese characters
        words = list(jieba.cut(text))  # Use Jieba for segmentation
        words = [word.strip() for word in words if word.strip()]  # Remove empty tokens
        return words

    elif lang == "ko":
        text = re.sub(r'[^가-힣\s]', '', text)  # Keep only Korean characters
        words = okt.morphs(text)  # Use Okt for tokenization
        words = [word.strip() for word in words if word.strip()]  # Remove empty tokens
        return words

    else:
        return text.split()  # Fallback: basic whitespace tokenization

