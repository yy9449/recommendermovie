import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from content_based import find_similar_titles
import os

@st.cache_data
def load_user_ratings():
    """
    Load user ratings. Priority:
    1. From the user-uploaded file (via session_state).
    2. From a local file path.
    3. Generate synthetic data as a fallback.
    """
    # Priority 1: Check session_state for the uploaded dataframe
    if 'user_ratings_df' in st.session_state:
        st.success("✅ Using user ratings from uploaded file.")
        return st.session_state['user_ratings_df']

    # Priority 2: If not in session state, try to find a local file
    for path in ["user_movie_rating.csv", "data/user_movie_rating.csv"]:
        if os.path.exists(path):
            try:
                user_ratings_df = pd.read_csv(path)
                st.success(f"✅ Loaded local user_movie_rating.csv from: {path}")
                return user_ratings_df
            except Exception as e:
                st.error(f"Error reading local file at {path}: {e}")

    # Priority 3: Fallback to synthetic data
    st.warning("📋 user_movie_rating.csv not found. Using synthetic data.")
    num_users = 100
    num_movies = 1000
    num_ratings = 5000
    user_ids = np.random.randint(1, num_users + 1, num_ratings)
    movie_ids = np.random.randint(1, num_movies + 1, num_ratings)
    ratings = np.random.randint(1, 11, num_ratings)
    synthetic_df = pd.DataFrame({'User_ID': user_ids, 'Movie_ID': movie_ids, 'Rating': ratings})
    return synthetic_df

# ... (the rest of your collaborative.py file remains unchanged)

def get_user_item_matrix(user_ratings_df, merged_df):
    """Create user-item matrix from ratings and movie data."""
    if user_ratings_df is None or user_ratings_df.empty:
        return None
    
    if 'Movie_ID' not in user_ratings_df.columns or 'Movie_ID' not in merged_df.columns:
        st.warning("⚠️ 'Movie_ID' column not found for creating user-item matrix.")
        return None
        
    valid_movie_ids = merged_df['Movie_ID'].unique()
    user_ratings_df = user_ratings_df[user_ratings_df['Movie_ID'].isin(valid_movie_ids)]

    if user_ratings_df.empty:
        st.warning("⚠️ No user ratings correspond to the movies in the loaded dataset.")
        return None

    user_item_matrix = user_ratings_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating').fillna(0)
    return user_item_matrix

@st.cache_data
def collaborative_filtering_enhanced(merged_df, target_movie, top_n=10):
    """Main collaborative filtering function using KNN."""
    user_ratings_df = load_user_ratings()
    
    if user_ratings_df is None or user_ratings_df.empty:
        st.error("Cannot perform collaborative filtering without user ratings data.")
        return None

    user_item_matrix = get_user_item_matrix(user_ratings_df, merged_df)

    if user_item_matrix is None:
        return None
        
    titles_list = merged_df['Series_Title'].tolist()
    similar_titles = find_similar_titles(target_movie, titles_list)
    
    if not similar_titles:
        st.warning(f"⚠️ Movie '{target_movie}' not found.")
        return None
    
    target_title_matched = similar_titles[0]
    
    try:
        target_movie_id = merged_df[merged_df['Series_Title'] == target_title_matched]['Movie_ID'].iloc[0]
    except IndexError:
        st.warning(f"⚠️ Could not find a Movie ID for '{target_title_matched}'.")
        return None

    users_who_rated = user_ratings_df[user_ratings_df['Movie_ID'] == target_movie_id]
    if users_who_rated.empty:
        st.info(f"ℹ️ No user ratings found for '{target_title_matched}'. Cannot find similar users.")
        return None

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
    model_knn.fit(user_item_matrix.values)
    
    similar_users = set()
    for user_id in users_who_rated['User_ID']:
        if user_id in user_item_matrix.index:
            query_index = user_item_matrix.index.get_loc(user_id)
            distances, indices = model_knn.kneighbors(user_item_matrix.iloc[query_index, :].values.reshape(1, -1))
            
            for i in range(1, len(distances.flatten())):
                similar_users.add(user_item_matrix.index[indices.flatten()[i]])

    if not similar_users:
        st.info("ℹ️ No similar users found for this movie.")
        return None

    recommended_movies = user_ratings_df[user_ratings_df['User_ID'].isin(similar_users)]
    
    movie_scores = recommended_movies.groupby('Movie_ID')['Rating'].mean().sort_values(ascending=False)
    
    movie_scores = movie_scores.drop(target_movie_id, errors='ignore')
    top_movie_ids = movie_scores.head(top_n * 2).index.tolist()
    
    result_df = merged_df[merged_df['Movie_ID'].isin(top_movie_ids)]
    
    rating_col = 'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating'
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'

    result_df = result_df.sort_values(by=rating_col, ascending=False)
    
    return result_df[['Series_Title', genre_col, rating_col]].head(top_n)

