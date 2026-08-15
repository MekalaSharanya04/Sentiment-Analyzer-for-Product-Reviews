import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


data_path = "dataset/clean_reviews_v2.csv"

df = pd.read_csv(data_path)

X_text = df["Clean_Review"].fillna("")
y = df["Sentiment"]

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/sentiment_svm_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("SVM model trained successfully!")
print("Model saved in models/")
