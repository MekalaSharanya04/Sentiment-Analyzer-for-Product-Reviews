"""Text preprocessing utilities for the sentiment analyzer."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    """Normalize review text for TF-IDF."""
    text = "" if text is None else str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words)
