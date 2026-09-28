"""Reusable prediction function."""
from pathlib import Path
import joblib
from preprocessing import clean_text
ROOT=Path(__file__).resolve().parents[1]; MODEL=ROOT/'models'/'sentiment_svm_model.pkl'; TFIDF=ROOT/'models'/'tfidf_vectorizer.pkl'
_model=None; _tfidf=None
def _load():
    global _model,_tfidf
    if _model is None: _model=joblib.load(MODEL); _tfidf=joblib.load(TFIDF)
    return _model,_tfidf
def predict_sentiment(review):
    model,tfidf=_load(); return model.predict(tfidf.transform([clean_text(review)]))[0]
if __name__=='__main__':
    review=input('Enter a product review: ').strip(); print('Predicted Sentiment:',predict_sentiment(review) if review else 'Please enter a review.')
