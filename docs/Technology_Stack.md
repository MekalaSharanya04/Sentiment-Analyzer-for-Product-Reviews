# Technology Stack

## 1. Programming Language

### Python

Python is the primary programming language used for the project.

It is used for:

* Data processing
* Text preprocessing
* Machine learning
* Model evaluation
* Prediction
* Application development

Python was selected because of its extensive libraries for Natural Language Processing, data analysis, and machine learning.

---

## 2. Data Processing

### Pandas

Pandas is used to:

* Read CSV datasets.
* Store review data in DataFrames.
* Clean and manipulate review records.
* Analyze sentiment distributions.
* Process bulk review files.

### NumPy

NumPy provides numerical operations required by several data-science and machine-learning libraries used in the project.

---

## 3. Natural Language Processing

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** is used to convert textual reviews into numerical feature vectors.

It helps represent the importance of words within individual reviews while considering their frequency across the dataset.

The TF-IDF representation is provided to the machine-learning classifier as input.

---

## 4. Machine Learning

### Scikit-learn

Scikit-learn is the main machine-learning library used in the project.

It is used for:

* Train-test splitting.
* TF-IDF vectorization.
* LinearSVC classification.
* Accuracy calculation.
* Precision calculation.
* Recall calculation.
* F1-score calculation.
* Confusion matrix generation.

### LinearSVC

The project uses **Linear Support Vector Classification (LinearSVC)** for sentiment prediction.

The model classifies reviews into:

* Positive
* Negative
* Neutral

The current evaluation achieved approximately **92.39% accuracy** on the held-out test data.

---

## 5. Model and Object Storage

### Joblib

Joblib is used to save and load the trained machine-learning components.

The project stores:

```text
sentiment_svm_model.pkl
tfidf_vectorizer.pkl
```

This allows the Streamlit application to reuse the trained model without retraining it every time the application starts.

---

## 6. Data Visualization

### Matplotlib

Matplotlib is used to create visualizations such as:

* Sentiment distribution.
* Confusion matrix.

These visualizations help evaluate and understand model results.

---

## 7. Web Application

### Streamlit

Streamlit is used to create the interactive web application.

The application supports:

* Individual review prediction.
* Bulk CSV upload.
* Sentiment classification.
* Sentiment summary metrics.
* Sentiment distribution visualization.
* Sentiment filtering.
* Review searching.
* Result downloading.

Streamlit was selected because it allows a Python-based machine-learning model to be converted into an interactive web interface without requiring a separate frontend framework.

---

## 8. Development Environment

The project can be developed and executed using:

* Windows
* Command Prompt
* Visual Studio Code / other Python IDEs
* Jupyter Notebook

---

## 9. Version Control and Repository

### GitHub

GitHub is used to store and document the project source code and project resources.

The repository contains:

* Source code.
* Documentation.
* Dataset-related files.
* Notebooks.
* Trained model files.
* Output files.
* Streamlit application.

---

## 10. Data Storage

### CSV

The current project uses CSV files for the prepared product-review dataset and bulk review input.

A database is **not part of the current implemented version**.

Database integration can be added in a future version to support continuous storage and retrieval of product reviews.

---

## 11. Overall Technology Pipeline

```text
Product Review Dataset
        ↓
      Pandas
        ↓
Text Preprocessing
        ↓
      TF-IDF
        ↓
    LinearSVC
        ↓
Sentiment Prediction
        ↓
Scikit-learn Evaluation
        ↓
   Joblib Model
        ↓
    Streamlit UI
        ↓
Charts / Filtering / Search / Download
```

## 12. Summary

| Component              | Technology                    |
| ---------------------- | ----------------------------- |
| Programming Language   | Python                        |
| Data Processing        | Pandas, NumPy                 |
| NLP Feature Extraction | TF-IDF                        |
| Machine Learning       | Scikit-learn                  |
| Classifier             | LinearSVC                     |
| Model Storage          | Joblib                        |
| Visualization          | Matplotlib, Streamlit         |
| Web Interface          | Streamlit                     |
| Dataset Format         | CSV                           |
| Development            | Jupyter Notebook / Python IDE |
| Version Control        | GitHub                        |
