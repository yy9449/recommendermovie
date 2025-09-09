import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from content_based import content_based_filtering_enhanced, create_content_features, find_rating_column, find_genre_column
from collaborative import collaborative_knn, load_user_ratings
import warnings
warnings.filterwarnings('ignore')


class FixedLinearHybridRecommender:
    def __init__(self, merged_df):
        self.merged_df = merged_df
        self.rating_col = find_rating_column(merged_df)
        self.genre_col = find_genre_column(merged_df)
        self.user_ratings_df = load_user_ratings()
        
        # Updated weights as requested
        self.alpha = 0.4  # Content-based weight
        self.beta = 0.4   # Collaborative weight  
        self.gamma = 0.1  # Popularity weight
        self.delta = 0.1  # Recency weight

    def _normalize_scores(self, scores_dict):
        """Normalize scores to 0-1 range"""
        if not scores_dict:
            return {}
        
        values = list(scores_dict.values())
        if not values:
            return scores_dict
            
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores_dict.keys()}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

    def _content_scores(self, target_movie, genre, top_n):
        """Get content-based similarity scores"""
        scores = {}
        
        if target_movie:
            try:
                content_features = create_content_features(self.merged_df)
                
                # Find target movie
                if target_movie in self.merged_df['Series_Title'].values:
                    target_idx = self.merged_df[self.merged_df['Series_Title'] == target_movie].index[0]
                else:
                    match_series = self.merged_df[self.merged_df['Series_Title'].str.lower() == str(target_movie).lower()]
                    if match_series.empty:
                        return scores
                    target_idx = match_series.index[0]
                
                target_loc = self.merged_df.index.get_loc(target_idx)
                target_vec = content_features[target_loc].reshape(1, -1)
                sims = cosine_similarity(target_vec, content_features).flatten()
                
                # Get all movies except target
                for i, sim_score in enumerate(sims):
                    if i != target_loc:
                        title = self.merged_df.iloc[i]['Series_Title']
                        scores[title] = float(sim_score)
                        
            except Exception as e:
                st.warning(f"Content scoring error: {e}")
                
        elif genre:
            try:
                # Genre-based content scoring
                genre_corpus = self.merged_df[self.genre_col].fillna('').tolist()
                tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = tfidf.fit_transform(genre_corpus)
                query_vector = tfidf.transform([genre])
                sims = cosine_similarity(query_vector, tfidf_matrix).flatten()
                
                for i, sim_score in enumerate(sims):
                    title = self.merged_df.iloc[i]['Series_Title']
                    scores[title] = float(sim_score)
                    
            except Exception as e:
                st.warning(f"Genre content scoring error: {e}")
        
        return self._normalize_scores(scores)

    def _collab_scores(self, target_movie, top_n):
        """Get collaborative filtering scores - FIXED to properly extract data"""
        scores = {}
        
        if target_movie and self.user_ratings_df is not None:
            try:
                # Get collaborative filtering results with more candidates
                collab_results = collaborative_knn(self.merged_df, target_movie, top_n=top_n * 5, k_neighbors=50)
                
                if collab_results is not None and not collab_results.empty:
                    # Extract similarity scores from collaborative results
                    if 'Similarity' in collab_results.columns:
                        for _, row in collab_results.iterrows():
                            title = row['Series_Title']
                            similarity = row.get('Similarity', 0.0)
                            if pd.notna(similarity):
                                scores[title] = float(similarity)
                    
                    # Also consider average user ratings as collaborative signal
                    if 'Avg_User_Rating' in collab_results.columns:
                        # Normalize avg user ratings (assuming 1-5 scale) and combine with similarity
                        for _, row in collab_results.iterrows():
                            title = row['Series_Title']
                            avg_rating = row.get('Avg_User_Rating', 3.0)
                            similarity = row.get('Similarity', 0.0)
                            
                            if pd.notna(avg_rating):
                                # Normalize rating to 0-1 scale (assuming 1-5 rating scale)
                                normalized_rating = (float(avg_rating) - 1.0) / 4.0
                                
                                # Combine similarity and rating (60% similarity, 40% rating)
                                if pd.notna(similarity):
                                    combined_score = 0.6 * float(similarity) + 0.4 * normalized_rating
                                else:
                                    combined_score = normalized_rating
                                
                                scores[title] = combined_score
                            elif pd.notna(similarity):
                                scores[title] = float(similarity)
                                
            except Exception as e:
                st.warning(f"Collaborative scoring error: {e}")
        
        return self._normalize_scores(scores)

    def _popularity_scores(self):
        """Calculate popularity scores using IMDB_Rating and vote count"""
        scores = {}
        
        try:
            for _, movie in self.merged_df.iterrows():
                title = movie['Series_Title']
                
                # Get IMDB rating
                imdb_rating = movie.get(self.rating_col, 7.0)
                if pd.isna(imdb_rating):
                    imdb_rating = 7.0
                
                # Get vote count
                votes_col = 'No_of_Votes' if 'No_of_Votes' in self.merged_df.columns else 'Votes'
                votes = movie.get(votes_col, 1000)
                
                # Clean vote count (remove commas, handle strings)
                try:
                    if isinstance(votes, str):
                        votes_val = float(votes.replace(',', '').replace(' ', ''))
                    else:
                        votes_val = float(votes) if pd.notna(votes) else 1000.0
                except:
                    votes_val = 1000.0
                
                # Calculate weighted popularity: rating * log(votes) 
                # Normalize IMDB rating to 0-1 (assuming 0-10 scale)
                normalized_rating = float(imdb_rating) / 10.0
                
                # Log transform votes for better distribution
                log_votes = np.log10(max(votes_val, 1.0))
                
                # Combine rating and popularity (weighted average)
                # Higher rated movies with more votes get higher scores
                popularity = (normalized_rating * 0.7) + (min(log_votes / 6.0, 1.0) * 0.3)
                
                scores[title] = float(np.clip(popularity, 0.0, 1.0))
                
        except Exception as e:
            st.warning(f"Popularity scoring error: {e}")
            # Fallback: use just ratings
            for _, movie in self.merged_df.iterrows():
                title = movie['Series_Title']
                rating = movie.get(self.rating_col, 7.0)
                if pd.isna(rating):
                    rating = 7.0
                scores[title] = float(rating) / 10.0
        
        return scores

    def _recency_scores(self):
        """Calculate recency scores using Released_Year"""
        scores = {}
        
        try:
            current_year = pd.Timestamp.now().year
            year_col = 'Released_Year' if 'Released_Year' in self.merged_df.columns else 'Year'
            
            for _, movie in self.merged_df.iterrows():
                title = movie['Series_Title']
                year = movie.get(year_col, 2000)
                
                # Extract year from string if needed
                try:
                    if isinstance(year, str):
                        # Handle formats like "2020 (I)" or "2020"
                        year_val = int(year.split()[0].strip('()'))
                    else:
                        year_val = int(year) if pd.notna(year) else 2000
                except:
                    year_val = 2000
                
                # Calculate recency - more recent movies get higher scores
                years_ago = max(0, current_year - year_val)
                
                # Exponential decay: more recent = higher score
                # Movies from current decade get scores > 0.5
                recency = np.exp(-years_ago / 15.0)  # 15-year half-life
                
                scores[title] = float(np.clip(recency, 0.0, 1.0))
                
        except Exception as e:
            st.warning(f"Recency scoring error: {e}")
            # Fallback: uniform scores
            for _, movie in self.merged_df.iterrows():
                scores[movie['Series_Title']] = 0.5
        
        return scores

    def recommend(self, target_movie=None, genre=None, top_n=8):
        """Generate hybrid recommendations using the fixed weighted formula"""
        
        # Get component scores
        content_scores = self._content_scores(target_movie, genre, top_n)
        collab_scores = self._collab_scores(target_movie, top_n)  
        popularity_scores = self._popularity_scores()
        recency_scores = self._recency_scores()
        
        # Debug info
        st.write(f"Debug - Content candidates: {len(content_scores)}")
        st.write(f"Debug - Collaborative candidates: {len(collab_scores)}")
        st.write(f"Debug - Popularity candidates: {len(popularity_scores)}")
        st.write(f"Debug - Recency candidates: {len(recency_scores)}")
        
        # Combine all candidates
        all_candidates = set()
        all_candidates.update(content_scores.keys())
        all_candidates.update(collab_scores.keys())
        all_candidates.update(popularity_scores.keys())
        all_candidates.update(recency_scores.keys())
        
        # Remove target movie from candidates
        if target_movie and target_movie in all_candidates:
            all_candidates.remove(target_movie)
        
        # Calculate final hybrid scores using the requested formula
        final_scores = {}
        
        for title in all_candidates:
            content_score = content_scores.get(title, 0.0)
            collab_score = collab_scores.get(title, 0.0) 
            popularity_score = popularity_scores.get(title, 0.0)
            recency_score = recency_scores.get(title, 0.0)
            
            # Apply the requested formula:
            # FinalScore = 0.4×Content + 0.4×Collaborative + 0.1×Popularity + 0.1×Recency
            final_score = (self.alpha * content_score + 
                          self.beta * collab_score + 
                          self.gamma * popularity_score + 
                          self.delta * recency_score)
            
            final_scores[title] = float(final_score)
        
        # Sort by final score and get top recommendations
        sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_titles = [title for title, score in sorted_recommendations[:top_n]]
        
        if not top_titles:
            return None
        
        # Create result dataframe with proper ordering
        result_rows = []
        for title in top_titles:
            movie_data = self.merged_df[self.merged_df['Series_Title'] == title]
            if not movie_data.empty:
                result_rows.append(movie_data.iloc[0])
        
        if not result_rows:
            return None
            
        result_df = pd.DataFrame(result_rows)
        
        # Add score breakdown for debugging (optional)
        if st.sidebar.checkbox("Show Score Breakdown", value=False):
            score_breakdown = []
            for title in top_titles:
                breakdown = {
                    'Movie': title,
                    'Content': f"{content_scores.get(title, 0.0):.3f}",
                    'Collaborative': f"{collab_scores.get(title, 0.0):.3f}",
                    'Popularity': f"{popularity_scores.get(title, 0.0):.3f}",
                    'Recency': f"{recency_scores.get(title, 0.0):.3f}",
                    'Final Score': f"{final_scores.get(title, 0.0):.3f}"
                }
                score_breakdown.append(breakdown)
            
            st.subheader("Score Breakdown")
            st.dataframe(pd.DataFrame(score_breakdown), use_container_width=True)
        
        return result_df[['Series_Title', self.genre_col, self.rating_col]]


@st.cache_data
def smart_hybrid_recommendation(merged_df, target_movie=None, genre=None, top_n=8, show_debug=False):
    """Updated hybrid recommendation function"""
    recommender = FixedLinearHybridRecommender(merged_df)
    return recommender.recommend(target_movie, genre, top_n, show_debug)
