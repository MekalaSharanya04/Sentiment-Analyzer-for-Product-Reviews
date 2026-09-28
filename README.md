# 💬 Sentiment Analyzer for Product Reviews

A Machine Learning application that analyzes product reviews and classifies them as **Positive, Neutral, or Negative**.

## 🎯 Problem Statement

E-commerce platforms can receive thousands of written customer reviews. Manually reading every review is time-consuming and makes it difficult for businesses to quickly identify negative feedback and common customer concerns.

## 💡 Proposed Solution

This project uses Natural Language Processing and Machine Learning to automatically analyze written reviews.

The application provides:

- Single-review sentiment prediction
- Bulk CSV review analysis
- Positive/Neutral/Negative classification
- Sentiment distribution visualization
- Negative-review filtering
- Keyword search inside reviews
- Downloadable analysis results

## 🧠 Machine Learning Pipeline

Customer Reviews  
↓  
Text Preprocessing  
↓  
TF-IDF Feature Extraction  
↓  
Linear SVM Classifier  
↓  
Positive / Neutral / Negative  
↓  
Streamlit Dashboard  
↓  
Business Feedback Analysis

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Linear SVM
- Joblib
- Matplotlib
- Streamlit
- GitHub

## 📊 Model Evaluation

The current project evaluation achieved approximately **92.39% accuracy** on the held-out test data.

Because the dataset contains many more Positive reviews than Neutral and Negative reviews, accuracy should be considered together with precision, recall and F1-score.

## ⭐ Main Feature

The system does not only display:

```text
Positive: 88%
Negative: 10%
Neutral: 2%
