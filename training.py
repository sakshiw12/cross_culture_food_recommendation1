# training.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('data (2).csv')

# Drop missing values in essential fields
df.dropna(subset=['name', 'ingredients', 'cuisine'], inplace=True)

# Combine features for better representation
df['combined'] = df['ingredients'] + ' ' + df['cuisine'] + ' ' + df['diet']

# Vectorize the combined text features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Create model object
model_data = {
    'tfidf_matrix': tfidf_matrix,
    'tfidf': tfidf,
    'data': df
}

# Save to pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Model trained and saved successfully.")
