"""Reusable sentiment prediction functions."""

from pathlib import Path
import joblib

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "models" / "sentiment_svm_model.pkl"
)

TFIDF_PATH = (
    ROOT / "models" / "tfidf_vectorizer.pkl"
)

_model = None
_tfidf = None


def load_model():
    global _model, _tfidf

    if _model is None or _tfidf is None:
        _model = joblib.load(MODEL_PATH)
        _tfidf = joblib.load(TFIDF_PATH)

    return _model, _tfidf


def predict_sentiment(review):

    model, tfidf = load_model()

    cleaned_review = clean_text(review)

    vector = tfidf.transform(
        [cleaned_review]
    )

    return model.predict(vector)[0]


if __name__ == "__main__":

    review = input(
        "Enter a product review: "
    ).strip()

    if not review:

        print(
            "Please enter a review."
        )

    else:

        result = predict_sentiment(
            review
        )

        print(
            "Predicted Sentiment:",
            result
        )
