import streamlit as st
import joblib


model = joblib.load("models/sentiment_svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


st.title("Sentiment Analyzer for Product Reviews")

st.write(
    "Enter a product review to predict its sentiment."
)

review = st.text_area("Enter your product review:")


if st.button("Analyze Sentiment"):

    if review.strip():

        vector = tfidf.transform([review])

        prediction = model.predict(vector)[0]

        st.success(f"Predicted Sentiment: {prediction}")

    else:

        st.warning("Please enter a review.")
