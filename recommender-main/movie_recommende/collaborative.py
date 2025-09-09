import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from difflib import get_close_matches
import os

def find_similar_titles(target_title, all_titles, n_matches=3, cutoff=0.6):
    """Find similar movie titles using fuzzy string matching"""
    if not target_title or not all_titles:
        return []
    
    # Use difflib for fuzzy matching
    matches = get_close_matches(target_title, all_titles, n=n_matches, cutoff=cutoff)
    return matches

@st.cache_data
def load_user_ratings():
    """Load real user ratings from CSV file or session state"""
    try:
        # First check if user uploaded the file (stored in session state)
        if 'user_ratings_df' in st.session_state:
            user_ratings_df = st.session_state['user_ratings_df']
            st.success("✅ Using uploaded user_movie_rating.csv from session")
            
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                st.warning(f"⚠️ Uploaded user_movie_rating.csv missing required columns: {required_cols}")
                return None
        
        # If not in session state, try to find it in local filesystem
        user_ratings_df = None
        for path in ["user_movie_rating.csv", "./user_movie_rating.csv", "data/user_movie_rating.csv", "../user_movie_rating.csv"]:
            if os.path.exists(path):
                user_ratings_df = pd.read_csv(path)
                st.success(f"✅ Found user_movie_rating.csv at: {path}")
                break
        
        if user_ratings_df is not None:
            # Validate required columns
            required_cols = ['User_ID', 'Movie_ID', 'Rating']
            if all(col in user_ratings_df.columns for col in required_cols):
                return user_ratings_df
            else:
                st.warning(f"⚠️ user_movie_rating.csv missing required columns: {required_cols}")
                return None
        else:
            st.info("📋 user_movie_rating.csv not found in local directory, using synthetic data")
            return None
            
    except Exception as e:
        st.error(f"Error loading user ratings: {str(e)}")
        return None

@st.cache_data
def create_user_item_matrix_from_real_data(merged_df, user_ratings_df):
    """Create user-item matrix from real user rating data - FIXED VERSION"""
    if user_ratings_df is None:
        return create_user_item_matrix_synthetic(merged_df)
    
    try:
        st.info(f"📊 Processing user ratings: {len(user_ratings_df)} ratings from {user_ratings_df['User_ID'].nunique()} users")
        
        # CRITICAL FIX: Remove duplicate ratings before creating pivot table
        # Keep only the latest rating if a user rated the same movie multiple times
        user_ratings_clean = user_ratings_df.drop_duplicates(subset=['User_ID', 'Movie_ID'], keep='last')
        
        duplicate_count = len(user_ratings_df) - len(user_ratings_clean)
        if duplicate_count > 0:
            st.info(f"🔄 Removed {duplicate_count} duplicate user-movie ratings")
        
        # Create a mapping between Movie_ID and our merged_df
        movie_id_to_index = {}
        
        # Try different approaches to map Movie_ID to our dataset
        if 'Movie_ID' in merged_df.columns:
            # Direct Movie_ID mapping
            movie_id_to_index = dict(zip(merged_df['Movie_ID'], merged_df.index))
        else:
            # Use index as Movie_ID (common approach)
            movie_id_to_index = dict(zip(range(len(merged_df)), merged_df.index))
        
        # Filter user ratings to only include movies in our dataset
        valid_movie_ids = set(movie_id_to_index.keys())
        filtered_ratings = user_ratings_clean[user_ratings_clean['Movie_ID'].isin(valid_movie_ids)].copy()
        
        if filtered_ratings.empty:
            st.warning("⚠️ No matching movies found between user ratings and movie dataset")
            st.info("💡 Make sure Movie_ID in user_movie_rating.csv corresponds to movie indices in your dataset")
            return create_user_item_matrix_synthetic(merged_df)
        
        st.success(f"✅ Successfully matched {len(filtered_ratings)} ratings with movies in dataset")
        
        # Map Movie_ID to our dataset indices
        filtered_ratings['dataset_index'] = filtered_ratings['Movie_ID'].map(movie_id_to_index)
        
        # Create user-item matrix using dataset indices - NOW WITHOUT DUPLICATES
        user_movie_matrix = filtered_ratings.pivot(
            index='User_ID', 
            columns='dataset_index', 
            values='Rating'
        ).fillna(0)
        
        # Ensure the matrix covers all movies in our dataset
        all_indices = list(range(len(merged_df)))
        missing_movies = set(all_indices) - set(user_movie_matrix.columns)
        
        for movie_idx in missing_movies:
            user_movie_matrix[movie_idx] = 0
        
        # Reorder columns to match merged_df order
        user_movie_matrix = user_movie_matrix.reindex(columns=all_indices, fill_value=0)
        
        rating_matrix = user_movie_matrix.values
        user_names = [f"User_{uid}" for uid in user_movie_matrix.index]
        
        st.success(f"🎯 Created user-item matrix: {len(user_names)} users × {len(all_indices)} movies")
        
        # Show some statistics
        non_zero_ratings = np.count_nonzero(rating_matrix)
        total_possible = rating_matrix.size
        sparsity = (1 - (non_zero_ratings / total_possible)) * 100
        
        st.info(f"📈 Matrix Statistics: {non_zero_ratings:,} ratings ({sparsity:.1f}% sparse)")
        
        return rating_matrix, user_names
        
    except Exception as e:
        st.error(f"Error processing user ratings: {str(e)}")
        st.exception(e)  # Show full error traceback for debugging
        return create_user_item_matrix_synthetic(merged_df)

@st.cache_data
def create_user_item_matrix_synthetic(merged_df):
    """Create a synthetic user-item matrix based on movie characteristics (fallback)"""
    np.random.seed(42)
    
    user_types = {
        'action_lover': {'Action': 5, 'Adventure': 4, 'Thriller': 4, 'Drama': 2, 'Comedy': 2, 'Romance': 1},
        'drama_fan': {'Drama': 5, 'Romance': 4, 'Biography': 4, 'Action': 2, 'Comedy': 3, 'Thriller': 2},
        'comedy_fan': {'Comedy': 5, 'Romance': 4, 'Family': 4, 'Action': 2, 'Drama': 3, 'Horror': 1},
        'thriller_fan': {'Thriller': 5, 'Mystery': 4, 'Crime': 4, 'Horror': 3, 'Action': 4, 'Comedy': 2},
        'classic_lover': {'Drama': 4, 'Romance': 4, 'Biography': 5, 'History': 5, 'War': 4, 'Comedy': 3},
        'sci_fi_fan': {'Sci-Fi': 5, 'Fantasy': 4, 'Action': 4, 'Adventure': 3, 'Thriller': 3, 'Drama': 2},
        'horror_fan': {'Horror': 5, 'Thriller': 4, 'Mystery': 4, 'Sci-Fi': 3, 'Action': 3, 'Comedy': 1},
        'family_viewer': {'Family': 5, 'Animation': 5, 'Comedy': 4, 'Adventure': 4, 'Fantasy': 3, 'Drama': 2}
    }
    
    user_movie_ratings = {}
    genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'
    
    for user_type, preferences in user_types.items():
        user_ratings = []
        for _, movie in merged_df.iterrows():
            rating = 0
            if pd.notna(movie[genre_col]):
                genres = [g.strip() for g in movie[genre_col].split(',')]
                genre_scores = [preferences.get(genre, 0) for genre in genres]
                if genre_scores:
                    base_rating = np.mean(genre_scores)
                    rating = max(1, min(5, base_rating + np.random.normal(0, 0.5)))
                    if np.random.random() < 0.3:
                        rating = 0
            
            user_ratings.append(rating)
        user_movie_ratings[user_type] = user_ratings
    
    rating_matrix = np.array(list(user_movie_ratings.values()))
    user_names = list(user_movie_ratings.keys())
    
    st.info("📊 Using synthetic user data for collaborative filtering")
    return rating_matrix, user_names

def collaborative_filtering_enhanced(merged_df, movie_title, top_n=8):
    """Enhanced Collaborative Filtering with real user data support"""
    try:
        # Load real user ratings
        user_ratings_df = load_user_ratings()
        
        # Create user-item matrix (will use real data if available, synthetic as fallback)
        rating_matrix, user_names = create_user_item_matrix_from_real_data(merged_df, user_ratings_df)
        
        # Find the movie index
        movie_matches = merged_df[merged_df['Series_Title'].str.contains(movie_title, case=False, na=False)]
        
        if movie_matches.empty:
            # Try fuzzy matching
            all_titles = merged_df['Series_Title'].tolist()
            similar_titles = find_similar_titles(movie_title, all_titles, n_matches=1, cutoff=0.3)
            if similar_titles:
                movie_title = similar_titles[0]
                movie_matches = merged_df[merged_df['Series_Title'] == movie_title]
            
            if movie_matches.empty:
                st.warning(f"Movie '{movie_title}' not found in dataset")
                return pd.DataFrame()
        
        movie_idx = movie_matches.index[0]
        actual_movie_title = movie_matches.iloc[0]['Series_Title']
        
        # Check if any users have rated this movie
        users_who_rated = np.where(rating_matrix[:, movie_idx] > 0)[0]
        
        if len(users_who_rated) == 0:
            st.warning(f"No users have rated '{actual_movie_title}'. Using item-based approach.")
            return item_based_collaborative_filtering(merged_df, movie_idx, rating_matrix, top_n)
        
        st.info(f"🎯 Found {len(users_who_rated)} users who rated '{actual_movie_title}'")
        
        # Use user-based collaborative filtering
        return user_based_collaborative_filtering(merged_df, users_who_rated, rating_matrix, movie_idx, top_n)
        
    except Exception as e:
        st.error(f"Error in collaborative filtering: {str(e)}")
        st.exception(e)
        return pd.DataFrame()

def user_based_collaborative_filtering(merged_df, users_who_rated, rating_matrix, movie_idx, top_n):
    """User-based collaborative filtering using KNN"""
    try:
        # Get ratings from users who rated the target movie
        similar_users_ratings = rating_matrix[users_who_rated]
        
        # Find movies that these users also rated highly (rating >= 4)
        recommendations = {}
        
        for user_idx in users_who_rated:
            user_ratings = rating_matrix[user_idx]
            # Find movies this user rated highly (4 or 5 stars)
            highly_rated_movies = np.where(user_ratings >= 4)[0]
            
            for rec_movie_idx in highly_rated_movies:
                if rec_movie_idx != movie_idx:  # Don't recommend the same movie
                    if rec_movie_idx not in recommendations:
                        recommendations[rec_movie_idx] = []
                    recommendations[rec_movie_idx].append(user_ratings[rec_movie_idx])
        
        # Calculate average ratings for each recommended movie
        movie_scores = {}
        for movie_idx_rec, ratings_list in recommendations.items():
            avg_rating = np.mean(ratings_list)
            movie_scores[movie_idx_rec] = avg_rating
        
        # Sort by average rating and get top N
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create results DataFrame
        results = []
        for movie_idx_rec, avg_rating in sorted_movies:
            movie_info = merged_df.iloc[movie_idx_rec]
            results.append({
                'Series_Title': movie_info['Series_Title'],
                'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre': 
                    movie_info.get('Genre_y', movie_info.get('Genre', 'N/A')),
                'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating':
                    movie_info.get('IMDB_Rating', movie_info.get('Rating', 0)),
                'Predicted_Rating': avg_rating
            })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            st.success(f"✅ Generated {len(results_df)} recommendations based on {len(users_who_rated)} similar users")
        
        return results_df
        
    except Exception as e:
        st.error(f"Error in user-based collaborative filtering: {str(e)}")
        return pd.DataFrame()

def item_based_collaborative_filtering(merged_df, movie_idx, rating_matrix, top_n):
    """Item-based collaborative filtering as fallback"""
    try:
        # Use KNN to find similar movies based on user ratings
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=min(top_n + 1, len(rating_matrix[0])))
        knn_model.fit(rating_matrix.T)  # Transpose for item-based
        
        # Find similar movies
        distances, indices = knn_model.kneighbors(rating_matrix[:, movie_idx].reshape(1, -1), n_neighbors=min(top_n + 1, len(rating_matrix[0])))
        
        # Get recommendations (excluding the input movie)
        similar_movie_indices = indices[0][1:]  # Exclude the first one (itself)
        
        results = []
        for idx in similar_movie_indices[:top_n]:
            movie_info = merged_df.iloc[idx]
            results.append({
                'Series_Title': movie_info['Series_Title'],
                'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre': 
                    movie_info.get('Genre_y', movie_info.get('Genre', 'N/A')),
                'IMDB_Rating' if 'IMDB_Rating' in merged_df.columns else 'Rating':
                    movie_info.get('IMDB_Rating', movie_info.get('Rating', 0))
            })
        
        results_df = pd.DataFrame(results)
        st.info(f"📊 Used item-based collaborative filtering")
        
        return results_df
        
    except Exception as e:
        st.error(f"Error in item-based collaborative filtering: {str(e)}")
        return pd.DataFrame()
