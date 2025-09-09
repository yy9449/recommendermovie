import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from content_based import content_based_filtering_enhanced
from collaborative import collaborative_filtering_enhanced, load_user_ratings
from hybrid import smart_hybrid_recommendation

warnings.filterwarnings('ignore')

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# =========================
# Data Loading with Error Handling
# =========================
@st.cache_data
def load_and_prepare_data():
    """Load CSVs and prepare data for recommendation algorithms"""
    try:
        # Try different possible file paths
        movies_df = None
        imdb_df = None

        # Check for movies.csv
        for path in ["movies.csv", "./movies.csv", "data/movies.csv", "../movies.csv"]:
            if os.path.exists(path):
                movies_df = pd.read_csv(path)
                st.success(f"✅ Found movies.csv at: {path}")
                break

        # Check for imdb_top_1000.csv
        for path in ["imdb_top_1000.csv", "./imdb_top_1000.csv", "data/imdb_top_1000.csv", "../imdb_top_1000.csv"]:
            if os.path.exists(path):
                imdb_df = pd.read_csv(path)
                st.success(f"✅ Found imdb_top_1000.csv at: {path}")
                break

        if movies_df is None or imdb_df is None:
            st.error("🚨 Critical Error: Could not find 'movies.csv' or 'imdb_top_1000.csv'. Please make sure the files are in the correct directory.")
            return None

        # Data merging and cleaning
        imdb_df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        movies_df['Movie_ID'] = movies_df.index
        merged_df = pd.merge(movies_df, imdb_df, left_on='Series_Title', right_on='Title', how='inner')
        
        # Ensure correct data types
        rating_col = 'IMDB_Rating'
        if rating_col in merged_df.columns:
            merged_df[rating_col] = pd.to_numeric(merged_df[rating_col], errors='coerce')
        
        st.success("✅ Data loaded successfully!")
        return merged_df

    except FileNotFoundError as e:
        st.error(f"🚨 File not found: {e}. Please ensure the necessary CSV files are in the root directory.")
        return None
    except Exception as e:
        st.error(f"🚨 An error occurred during data loading: {e}")
        return None

merged_df = load_and_prepare_data()

# =========================
# Sidebar Configuration
# =========================
st.sidebar.header("⚙️ Configuration")

# ✨ NEW: File uploader for user ratings
uploaded_file = st.sidebar.file_uploader("📤 Upload user_movie_rating.csv", type=["csv"])

recommendation_mode = st.sidebar.selectbox(
    "Select Recommendation Mode",
    ["Hybrid (Content + Collaborative)", "Content-Based", "Collaborative Filtering"]
)

if merged_df is not None:
    # Use 'Series_Title' as it's the common column after the merge
    movie_list = merged_df['Series_Title'].unique().tolist()
    
    selected_movie = st.sidebar.selectbox(
        "🎬 Select a Movie You Like",
        options=movie_list,
        index=0  # Default to the first movie
    )
    
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About:** This app provides movie recommendations based on different filtering techniques.")

# =========================
# Main Panel Display
# =========================
if merged_df is not None:
    st.header("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Movies:** {len(merged_df)}")
        
        rating_col = 'IMDB_Rating'
        genre_col = 'Genre_y' if 'Genre_y' in merged_df.columns else 'Genre'

        if rating_col in merged_df.columns:
            avg_rating = merged_df[rating_col].mean()
            max_rating = merged_df[rating_col].max()
            min_rating = merged_df[rating_col].min()
            st.write(f"**Average Rating:** {avg_rating:.1f}⭐")
            st.write(f"**Rating Range:** {min_rating:.1f} - {max_rating:.1f}")

        # ✨ UPDATED: Logic to handle uploaded file
        # Pass the uploaded file object to the loading function
        user_ratings_df = load_user_ratings(uploaded_file)
        
        if user_ratings_df is not None and not user_ratings_df.empty:
            st.write(f"**User Ratings Available:** ✅")
            st.write(f"**Total User Ratings:** {len(user_ratings_df)}")
            st.write(f"**Unique Users:** {user_ratings_df['User_ID'].nunique()}")
        else:
            st.write(f"**User Ratings Available:** ❌ (Using synthetic data)")
    
    with col2:
        # Top genres
        all_genres = []
        for genre_str in merged_df[genre_col].dropna():
            if isinstance(genre_str, str):
                all_genres.extend([g.strip() for g in genre_str.split(',')])
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            st.bar_chart(genre_counts.head(10))

    st.markdown("---")
    st.header(f"✨ Recommendations for '{selected_movie}'")

    if st.button("Get Recommendations"):
        with st.spinner('Finding recommendations...'):
            recommendations = None
            if recommendation_mode == "Content-Based":
                recommendations = content_based_filtering_enhanced(merged_df, target_title=selected_movie, top_n=top_n)
            elif recommendation_mode == "Collaborative Filtering":
                recommendations = collaborative_filtering_enhanced(merged_df, target_movie=selected_movie, top_n=top_n)
            else: # Hybrid
                recommendations = smart_hybrid_recommendation(merged_df, target_movie=selected_movie, top_n=top_n)
            
            if recommendations is not None and not recommendations.empty:
                st.success(f"Here are your top {top_n} recommendations:")
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.warning("Could not generate recommendations. Try a different movie or mode.")
else:
    st.info("Please wait for the data to load, or check the file paths if an error occurs.")
