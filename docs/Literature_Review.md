# Literature Review

## 1. Introduction

Sentiment analysis is a Natural Language Processing (NLP) technique used to identify the opinion or emotional orientation expressed in textual data. It is commonly applied to customer reviews, social media posts, feedback forms, surveys, and other sources of user-generated content.

In product-review analysis, sentiment classification can help identify whether customers are generally satisfied, dissatisfied, or neutral about a product.

## 2. Traditional Sentiment Analysis

Early sentiment-analysis systems commonly used rule-based approaches. These systems depended on manually created lists of positive and negative words.

For example:

* "excellent"
* "amazing"
* "useful"

may indicate positive sentiment, while:

* "poor"
* "worst"
* "broken"

may indicate negative sentiment.

Although rule-based approaches are simple, they may have difficulty understanding context, sentence structure, negation, and different ways of expressing opinions.

## 3. Machine Learning-Based Sentiment Analysis

Machine learning approaches allow a model to learn sentiment patterns from previously labelled examples.

Common algorithms used for text classification include:

* Naive Bayes
* Logistic Regression
* Support Vector Machines
* Decision Trees
* Random Forests

The quality of a machine-learning model depends on factors such as the training dataset, text preprocessing, feature representation, and classification algorithm.

## 4. TF-IDF for Text Representation

Text cannot be directly provided to most traditional machine-learning classifiers. Therefore, textual information needs to be converted into numerical features.

TF-IDF is a commonly used feature-extraction technique for text classification.

It considers:

* How frequently a word appears in a document.
* How common or uncommon that word is across the complete collection of documents.

This allows important words in reviews to receive appropriate numerical representation.

In this project, TF-IDF is used to transform cleaned product reviews into numerical feature vectors.

## 5. Support Vector Machine

Support Vector Machines are supervised machine-learning algorithms that can be used for classification problems.

For text classification, a linear SVM is particularly useful because text data represented using TF-IDF can contain a large number of features.

This project uses **LinearSVC**, a linear Support Vector Classification algorithm provided by Scikit-learn.

The classifier learns patterns from labelled reviews and predicts one of the three sentiment categories:

* Positive
* Negative
* Neutral

## 6. Evaluation of Sentiment Classification

Accuracy alone may not provide a complete understanding of a classification model, particularly when the classes are not evenly distributed.

Therefore, this project evaluates the model using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The current model achieved approximately **92.39% accuracy** on the selected held-out test set.

The evaluation also shows that the dataset contains substantially more Positive reviews than Negative and Neutral reviews. Therefore, class-wise precision, recall, and F1-score are considered along with overall accuracy.

## 7. Interactive Sentiment Analysis Applications

Machine-learning results become more useful when users can interact with them through a simple interface.

The project uses **Streamlit** to provide a web-based interface. Users can enter an individual review or upload a CSV containing multiple reviews for bulk analysis.

The application displays sentiment counts and allows users to filter and search the analyzed reviews.

## 8. Research Gap and Project Contribution

Many sentiment-analysis implementations demonstrate only the classification of a single text input. In practical situations, businesses may need to analyze a collection of reviews and inspect specific categories of feedback.

This project extends the basic prediction workflow by providing:

* Single-review prediction.
* Bulk CSV analysis.
* Sentiment distribution.
* Sentiment-based filtering.
* Review searching.
* Downloadable analysis results.
* An interactive Streamlit interface.

The project therefore combines the machine-learning pipeline with a practical user interface for product-review analysis.

## 9. Future Research Directions

Future development can explore:

* Aspect-based sentiment analysis.
* Multilingual review classification.
* Transformer-based NLP models.
* Database integration.
* Automatic review collection through APIs.
* Product-specific sentiment analysis.
* Sentiment trends over time.

## Conclusion

The literature and existing approaches show that NLP and supervised machine learning can be effectively used for automated sentiment classification. Based on these approaches, this project implements a TF-IDF and LinearSVC-based sentiment-analysis pipeline and integrates it with a Streamlit application for practical product-review analysis.
