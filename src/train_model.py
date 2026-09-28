"""Train and save the TF-IDF + Linear SVM sentiment model."""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "clean_reviews_v2.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

required_columns = {"Clean_Review", "Sentiment"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

X_text = (
    df["Clean_Review"]
    .fillna("")
    .astype(str)
    .map(clean_text)
)

y = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
)

valid_rows = X_text.str.len().gt(0) & y.ne("")

X_text = X_text[valid_rows]
y = y[valid_rows]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")

print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    MODEL_DIR / "sentiment_svm_model.pkl"
)

joblib.dump(
    tfidf,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("Model saved successfully.")
print("TF-IDF vectorizer saved successfully.")
