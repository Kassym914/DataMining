import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Set matplotlib backend for non-interactive environments (use 'Agg' if you have issues with display)
matplotlib.use('Agg')

# Load Yelp JSON dataset (update path to actual dataset)
# The dataset contains reviews and business data
data = []
with open("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Convert the loaded data to a DataFrame
df = pd.DataFrame(data)

# Sample 10% of the data randomly
data_sample = df.sample(frac=0.5, random_state=42)

# Filter for restaurant businesses and group reviews by cuisine
# Example cuisines: ["Indian", "Italian", "Chinese", "Mexican", "Japanese"]
cuisines = ["Indian", "Italian", "Chinese", "Mexican", "Japanese"]

# Create a dictionary to store aggregated reviews for each cuisine
cuisine_reviews = {cuisine: [] for cuisine in cuisines}

# Example: if 'categories' are available in a business dataset linked to the 'business_id'
# For simplicity, let's assume we're directly categorizing based on keywords in the review text
for index, row in data_sample.iterrows():
    for cuisine in cuisines:
        if cuisine.lower() in row['text'].lower():  # This assumes cuisines are mentioned in the review text
            cuisine_reviews[cuisine].append(row['text'])

# Combine all reviews for each cuisine into a single string
for cuisine in cuisines:
    cuisine_reviews[cuisine] = " ".join(cuisine_reviews[cuisine])

# Vectorize the reviews using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(cuisine_reviews.values())

# Compute cosine similarity between cuisines
similarity_matrix = cosine_similarity(X)

# Visualize the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, xticklabels=cuisines, yticklabels=cuisines, cmap="Blues", cbar=True)
plt.title("Cuisine Similarity Matrx")
plt.xlabel("Cuisines")
plt.ylabel("Cuisines")

# Save the plot as a PNG file (adjust the path as needed)
plt.savefig("cuisine_similarity_matrix.png")

# Display the plot (this step is often used in interactive environments like Jupyter notebooks)
plt.show()
