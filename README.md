# 💬 Sentiment Analyzer for Product Reviews

A Python NLP and machine-learning application that classifies product reviews as **Positive, Neutral, or Negative** using **TF-IDF + Linear SVM**.

## Why this project?
Businesses can receive thousands of written reviews. Ratings show *how much* customers liked a product, while written reviews explain *why*. Manually reading every review is slow. This project automatically classifies large collections of feedback and lets a business inspect the actual negative, neutral, or positive reviews.

## Features
- Single-review sentiment prediction
- Bulk CSV sentiment analysis
- Positive/Neutral/Negative counts and percentages
- Interactive sentiment chart
- Filter by sentiment
- Search reviews by keyword
- View the actual reviews behind a sentiment count
- Download filtered analysis results
- Cached model loading for a more responsive Streamlit UI

## Pipeline
`Reviews → Cleaning → TF-IDF → Linear SVM → Sentiment → Dashboard`

## Technologies
Python, Pandas, Scikit-learn, TF-IDF, Linear SVM, Joblib, Streamlit, Matplotlib, Git and GitHub.

## Run
```bash
python -m pip install -r requirements.txt
python src/train_model.py
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8502
```
Open `http://127.0.0.1:8502`.

## CSV
Bulk mode recognizes: `Clean_Review`, `Review`, `review`, `Review_Text`, `review_text`, `Text`, or `text`.

## Evaluation
The existing evaluation achieved about **92.39% accuracy**. Because the dataset is strongly dominated by Positive reviews, precision, recall and F1-score should also be considered.

## Future scope
Aspect-based sentiment (battery/camera/delivery), multilingual support, database integration, product dashboards, time-based trends, authentication and cloud deployment.
