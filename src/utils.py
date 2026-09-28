"""Utility functions for sentiment analysis."""


def sentiment_counts(predictions):

    return {
        "Positive": sum(
            x == "Positive"
            for x in predictions
        ),

        "Neutral": sum(
            x == "Neutral"
            for x in predictions
        ),

        "Negative": sum(
            x == "Negative"
            for x in predictions
        )
    }


def display_prediction(
    review,
    sentiment
):

    print(
        f"Review: {review}"
    )

    print(
        f"Predicted Sentiment: {sentiment}"
    )
