# System Design

## System Architecture

```text
                 CUSTOMER REVIEWS
                        |
                        v
              +-------------------+
              | Text Preprocessing|
              +---------+---------+
                        |
                        v
              +-------------------+
              | TF-IDF Vectorizer |
              +---------+---------+
                        |
                        v
              +-------------------+
              |   Linear SVM      |
              |  Classification   |
              +---------+---------+
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Positive   Neutral    Negative
             |          |          |
             +----------+----------+
                        |
                        v
              +-------------------+
              | Streamlit Dashboard|
              +---------+---------+
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Charts     Filtering   Download
