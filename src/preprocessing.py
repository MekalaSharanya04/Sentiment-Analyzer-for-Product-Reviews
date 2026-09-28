"""Text preprocessing utilities."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP_WORDS=set(ENGLISH_STOP_WORDS)
def clean_text(text):
    text='' if text is None else str(text)
    text=text.lower(); text=re.sub(r'http\S+|www\S+',' ',text); text=re.sub(r'[^a-zA-Z\s]',' ',text); text=re.sub(r'\s+',' ',text).strip()
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)
