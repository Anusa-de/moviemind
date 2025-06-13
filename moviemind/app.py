from flask import Flask, render_template, request, jsonify
from flask import request, redirect
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
import os

app = Flask(__name__)

# Load Data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('model/u.data', sep='\t', names=column_names)
movie_titles = pd.read_csv("model/Movie_Id_Titles")
df = pd.merge(df, movie_titles, on='item_id')

n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

# Train-test split
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')



OMDB_API_KEY = '4ac901ef'  # replace with your key

import requests
import re

def clean_title(title):
    # Remove year in brackets e.g., " (1995)"
    title = re.sub(r'\s+\(\d{4}\)', '', title)
    
    # Handle titles like "Shawshank Redemption, The" → "The Shawshank Redemption"
    if ', The' in title:
        title = 'The ' + title.replace(', The', '')
    elif ', A' in title:
        title = 'A ' + title.replace(', A', '')
    elif ', An' in title:
        title = 'An ' + title.replace(', An', '')
    
    return title.strip()

def get_movie_poster(title):
    cleaned_title = clean_title(title)
    api_key = "4ac901ef"  # your OMDb API key
    url = f"http://www.omdbapi.com/?t={cleaned_title}&apikey={api_key}"
    
    try:
        res = requests.get(url).json()
        if res.get('Response') == 'True':
            return res.get('Poster', None)
        else:
            print(f"OMDb Error for '{cleaned_title}':", res.get('Error'))
            return None
    except Exception as e:
        print("Poster fetch failed:", e)
        return None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend_by_movie')
def recommend_by_movie():
    movie_query = request.args.get('movie', '').lower().strip()

    # Clean up titles for flexible matching
    def clean_title(title):
        title = title.lower()
        if '(' in title:
            title = title[:title.rfind('(')].strip()
        return title

    # Create a mapping from cleaned title to actual title
    pivot_table = df.pivot_table(index='user_id', columns='title', values='rating')
    cleaned_to_original = {clean_title(title): title for title in pivot_table.columns}

    # Find the best match
    matched_title = None
    for clean, original in cleaned_to_original.items():
        if movie_query in clean:
            matched_title = original
            break

    if not matched_title:
        return jsonify({'error': 'Movie not found in dataset.'})

    # Proceed with recommendation using the matched title
    movie_ratings = pivot_table[matched_title]
    similar_movies = pivot_table.corrwith(movie_ratings)
    corr_movies = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_movies.dropna(inplace=True)

    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['num of ratings'] = df.groupby('title')['rating'].count()
    corr_movies = corr_movies.join(ratings['num of ratings'])

    recommendations = corr_movies[corr_movies['num of ratings'] > 100].sort_values('Correlation', ascending=False).head(10)

    result = []
    for index, row in recommendations.iterrows():
        title = index
        correlation = row['Correlation']
        poster = get_movie_poster(title)
        result.append({
            'title': title,
            'Correlation': correlation,
            'poster': poster
        })

    return jsonify(result)





@app.route('/popular_movies')
def popular_movies():
    ratings = df.groupby('title')['rating'].mean()
    counts = df.groupby('title')['rating'].count()
    popular = ratings[counts > 100].sort_values(ascending=False).head(10)

    result = []
    for title in popular.index:
        rate = popular[title]
        count = counts[title]
        poster_url = get_movie_poster(title)
        result.append({
            'title': title,
            'rating': rate,
            'poster': poster_url,
            'num of ratings': int(count)
        })

    
    return jsonify(result)

    

@app.route('/user_based_cf')
def user_based_cf():
    mse = rmse(user_prediction, test_data_matrix)
    return f"User-based CF RMSE: {mse:.4f}"

@app.route('/item_based_cf')
def item_based_cf():
    mse = rmse(item_prediction, test_data_matrix)
    return f"Item-based CF RMSE: {mse:.4f}"



@app.route('/rate5')
def rate_five():
    return render_template('a.html')

@app.route('/all_movies')
def all_movies():
    unique_titles = df['title'].unique().tolist()
    return jsonify(sorted(unique_titles))

# @app.route('/user_recommendation', methods=['POST'])
# def user_recommendation():
#     data = request.get_json()
#     selected_movies = data.get('movies', [])

#     if len(selected_movies) != 5:
#         return jsonify({'error': 'Please select exactly 5 movies'}), 400

#     # Build user-item matrix
#     ratings_matrix = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
#     movie_list = ratings_matrix.columns.tolist()

#     # Ensure all selected movies exist
#     for movie in selected_movies:
#         if movie not in movie_list:
#             return jsonify({'error': f'Movie \"{movie}\" not found in dataset'}), 400

#     # Step 1: Create a pseudo-user vector with 5-star ratings
#     pseudo_user = np.zeros(len(movie_list))
#     for movie in selected_movies:
#         pseudo_user[movie_list.index(movie)] = 5

#     # Step 2: Add this pseudo user to the ratings matrix
#     ratings_matrix_np = ratings_matrix.to_numpy()
#     ratings_with_pseudo = np.vstack([ratings_matrix_np, pseudo_user])

#     # Step 3: Compute user-user cosine similarity
#     user_similarity = 1 - pairwise_distances(ratings_with_pseudo, metric='cosine')

#     # Step 4: Get similarity of pseudo user with all real users (excluding itself)
#     sim_vector = user_similarity[-1][:-1]  # shape: (num_real_users,)

#     # Step 5: Compute mean ratings for each real user
#     mean_user_rating = ratings_matrix_np.mean(axis=1)

#     # Step 6: Compute rating deviation matrix
#     ratings_diff = ratings_matrix_np - mean_user_rating[:, np.newaxis]

#     # Step 7: Predict pseudo user’s ratings for all items
#     # result shape: (num_items,)
#     pred = sim_vector.dot(ratings_diff) / np.abs(sim_vector).sum()
#     pred = pred + ratings_matrix_np.mean(axis=0)  # Add back item-wise global average

#     # Step 8: Filter out already rated movies
#     already_rated = set(selected_movies)
#     recommendations = []
#     for i, movie in enumerate(movie_list):
#         if movie not in already_rated:
#             recommendations.append((movie, pred[i]))

#     # Step 9: Sort and return top 10
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#     top_movies = [{'title': title} for title, _ in recommendations[:10]]

#     return jsonify(top_movies)
import requests
import re

def clean_omdb_title(title):
    title = re.sub(r'\s+\(\d{4}\)', '', title)
    if ', The' in title:
        title = 'The ' + title.replace(', The', '')
    elif ', A' in title:
        title = 'A ' + title.replace(', A', '')
    elif ', An' in title:
        title = 'An ' + title.replace(', An', '')
    return title.strip()

def get_movie_poster(title):
    cleaned_title = clean_omdb_title(title)
    url = f"http://www.omdbapi.com/?t={cleaned_title}&apikey=4ac901ef"  # use your actual API key
    try:
        res = requests.get(url)
        data = res.json()
        if data.get("Response") == "True":
            return data.get("Poster")
    except:
        pass
    return None

@app.route('/user_recommendation', methods=['POST'])
def user_recommendation():
    data = request.get_json()
    selected_movies = data.get('movies', [])

    if len(selected_movies) != 5:
        return jsonify({'error': 'Please select exactly 5 movies'}), 400

    ratings_matrix = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)
    movie_list = ratings_matrix.columns.tolist()

    for movie in selected_movies:
        if movie not in movie_list:
            return jsonify({'error': f'Movie \"{movie}\" not found in dataset'}), 400

    pseudo_user = np.zeros(len(movie_list))
    for movie in selected_movies:
        pseudo_user[movie_list.index(movie)] = 5

    ratings_matrix_np = ratings_matrix.to_numpy()
    ratings_with_pseudo = np.vstack([ratings_matrix_np, pseudo_user])

    user_similarity = 1 - pairwise_distances(ratings_with_pseudo, metric='cosine')
    sim_vector = user_similarity[-1][:-1]
    mean_user_rating = ratings_matrix_np.mean(axis=1)
    ratings_diff = ratings_matrix_np - mean_user_rating[:, np.newaxis]
    pred = sim_vector.dot(ratings_diff) / np.abs(sim_vector).sum()
    pred = pred + ratings_matrix_np.mean(axis=0)

    already_rated = set(selected_movies)
    recommendations = []
    for i, movie in enumerate(movie_list):
        if movie not in already_rated:
            recommendations.append((movie, pred[i]))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_movies = []
    for title, score in recommendations[:10]:
        poster = get_movie_poster(title)
        top_movies.append({
            'title': title,
            'poster': poster
        })

    return jsonify(top_movies)



if __name__ == '__main__':
    app.run(debug=True)
