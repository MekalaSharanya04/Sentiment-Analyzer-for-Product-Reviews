import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/clean_reviews_v2.csv")

# Count each sentiment
sentiment_counts = df["Sentiment"].value_counts()

# Display counts
print("Sentiment Distribution:")
print(sentiment_counts)

# Create bar chart
sentiment_counts.plot(kind="bar")

plt.title("Sentiment Distribution of Product Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

# Save chart
plt.savefig("outputs/sentiment_distribution.png")

# Display chart
plt.show()