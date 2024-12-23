import pandas as pd
import json
import nltk
import gensim
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Load Yelp review data from JSON file
file_path = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'

# Read the JSON file into a Pandas DataFrame
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    return df

df = load_data(file_path)

# Preprocessing: Tokenization and stopword removal
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# Apply preprocessing to all reviews
df['tokens'] = df['text'].apply(preprocess_text)

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# LDA Model
from gensim.models import LdaModel

# Train the LDA model with a fixed number of topics
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Visualize the topics using pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# Display the visualization
pyLDAvis.display(vis)

# Save the visualization as HTML (optional)
vis.save_html('lda_visualization.html')

# Task 1.2: Comparing Positive vs Negative Reviews

# Define subsets of positive and negative reviews
positive_reviews = df[df['stars'] >= 4]
negative_reviews = df[df['stars'] <= 2]

# Preprocess text for both subsets
positive_reviews['tokens'] = positive_reviews['text'].apply(preprocess_text)
negative_reviews['tokens'] = negative_reviews['text'].apply(preprocess_text)

# Create dictionaries and corpora for both subsets
positive_dictionary = corpora.Dictionary(positive_reviews['tokens'])
negative_dictionary = corpora.Dictionary(negative_reviews['tokens'])

positive_corpus = [positive_dictionary.doc2bow(tokens) for tokens in positive_reviews['tokens']]
negative_corpus = [negative_dictionary.doc2bow(tokens) for tokens in negative_reviews['tokens']]

# Train LDA models for both subsets
positive_lda_model = LdaModel(positive_corpus, num_topics=num_topics, id2word=positive_dictionary, passes=15)
negative_lda_model = LdaModel(negative_corpus, num_topics=num_topics, id2word=negative_dictionary, passes=15)

# Visualize the topics for positive reviews
positive_vis = pyLDAvis.gensim_models.prepare(positive_lda_model, positive_corpus, positive_dictionary)
pyLDAvis.display(positive_vis)

# Visualize the topics for negative reviews
negative_vis = pyLDAvis.gensim_models.prepare(negative_lda_model, negative_corpus, negative_dictionary)
pyLDAvis.display(negative_vis)

# Save the visualizations as HTML (optional)
positive_vis.save_html('positive_reviews_lda_visualization.html')
negative_vis.save_html('negative_reviews_lda_visualization.html')
