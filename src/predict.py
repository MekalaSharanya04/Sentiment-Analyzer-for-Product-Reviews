import joblib


model = joblib.load("models/sentiment_svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def predict_sentiment(review):

    vector = tfidf.transform([review])

    prediction = model.predict(vector)[0]

    return prediction


if __name__ == "__main__":

    review = input("Enter a product review: ")

    result = predict_sentiment(review)

    print("Predicted Sentiment:", result)
