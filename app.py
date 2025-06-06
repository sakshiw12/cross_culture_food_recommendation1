# app.py

from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
df = model['data']
tfidf = model['tfidf']
tfidf_matrix = model['tfidf_matrix']

@app.route('/')
def home():
    return render_template('index.html', recipes=df['name'].tolist())

@app.route('/predict', methods=['POST'])
def predict():
    dish_name = request.form['dish_name']
    
    if dish_name not in df['name'].values:
        return render_template('index.html', recipes=df['name'].tolist(),
                               prediction_text="Dish not found. Please select a valid dish.")

    idx = df[df['name'] == dish_name].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = df.iloc[[i[0] for i in sim_scores]][['name', 'cuisine', 'diet', 'ingredients']]

    return render_template('index.html', recipes=df['name'].tolist(),
                           prediction_text="Recommended Dishes:", tables=[recommendations.to_html(classes='data', header="true", index=False)])

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
