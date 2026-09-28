"""Project helper functions."""
def sentiment_counts(predictions):
    return {s:sum(x==s for x in predictions) for s in ['Positive','Neutral','Negative']}
def display_prediction(review,sentiment):
    print(f'Review: {review}\nPredicted Sentiment: {sentiment}')
