import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    stop_words = set(ENGLISH_STOP_WORDS)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
