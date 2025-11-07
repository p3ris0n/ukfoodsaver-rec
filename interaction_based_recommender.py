"""
UKFoodSaver Interaction-Based Recommendation System
Migrated from rating-based to interaction-based approach
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from surprise import Dataset, Reader, SVD
from scipy.sparse import csr_matrix
from collections import defaultdict
import json

# ============================================================================
# INTERACTION WEIGHTS CONFIGURATION
# ============================================================================

INTERACTION_WEIGHTS = {
    'view': 1.0,           # User viewed the item
    'add_to_cart': 3.0,    # Strong signal - user wants it
    'purchase': 5.0         # Strongest signal - user bought it
}

# For cold start - popularity calculation
POPULARITY_DECAY_DAYS = 2  # Items lose popularity after 2 days (food expiration)


# ============================================================================
# DATA PREPROCESSING: Convert Interactions to Implicit Ratings
# ============================================================================

def load_interaction_data(csv_path=None, df=None):
    """
    Load interaction data and convert to implicit ratings.
    
    Expected columns: user_id, item_id, interaction_type, timestamp
    If still using old format (user_id, item_id, rating), it will handle it.
    """
    if df is None:
        df = pd.read_csv("data/UKFS_testdata.csv")
    
    # Check if we're using new interaction format or old rating format
    if 'interaction_type' in df.columns:
        # NEW FORMAT: Convert interactions to implicit ratings
        df['implicit_rating'] = df['interaction_type'].map(INTERACTION_WEIGHTS)
        
        # Handle missing interaction types
        df['implicit_rating'] = df['implicit_rating'].fillna(INTERACTION_WEIGHTS['view'])
        
    elif 'rating' in df.columns:
        # OLD FORMAT: Use ratings directly (backward compatibility)
        df['implicit_rating'] = df['rating']
        
    else:
        raise ValueError("DataFrame must have either 'interaction_type' or 'rating' column")
    
    # Aggregate multiple interactions per user-item pair
    # A user might view, add to cart, AND purchase the same item
    aggregated = df.groupby(['user_id', 'item_id']).agg({
        'implicit_rating': 'sum',  # Sum all interaction weights
        'timestamp': 'max' if 'timestamp' in df.columns else 'first'
    }).reset_index()
    
    # Cap ratings at reasonable maximum (e.g., view + cart + purchase = 9.0)
    aggregated['implicit_rating'] = aggregated['implicit_rating'].clip(upper=10.0)
    
    print(f"Loaded {len(df)} interactions → {len(aggregated)} unique user-item pairs")
    print(f"Rating distribution:\n{aggregated['implicit_rating'].describe()}")
    
    return aggregated


def create_interaction_matrix(interactions_df):
    """
    Create sparse user-item interaction matrix.
    Useful for quick lookups and matrix operations.
    """
    user_ids = interactions_df['user_id'].unique()
    item_ids = interactions_df['item_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
    
    rows = [user_to_idx[user] for user in interactions_df['user_id']]
    cols = [item_to_idx[item] for item in interactions_df['item_id']]
    data = interactions_df['implicit_rating'].values
    
    interaction_matrix = csr_matrix((data, (rows, cols)), 
                                   shape=(len(user_ids), len(item_ids)))
    
    return interaction_matrix, user_to_idx, item_to_idx


# ============================================================================
# POSTAL CODE & KEYWORD FILTERING
# ============================================================================

class ItemMetadata:
    """
    Manages item metadata for filtering (postal code, keywords, freshness).
    This would typically come from your backend database.
    """
    
    def __init__(self):
        self.items = {}  # item_id -> {postal_code, keywords, created_at, store_id}
    
    def add_item(self, item_id, postal_code, keywords, store_id, created_at=None):
        """Add or update item metadata."""
        self.items[item_id] = {
            'postal_code': postal_code,
            'keywords': set(kw.lower() for kw in keywords) if keywords else set(),
            'store_id': store_id,
            'created_at': created_at or datetime.now()
        }
    
    def filter_by_postal_code(self, item_ids, postal_code):
        """Filter items by postal code."""
        return [
            item_id for item_id in item_ids 
            if self.items.get(item_id, {}).get('postal_code') == postal_code
        ]
    
    def filter_by_keyword(self, item_ids, keyword):
        """Filter items by keyword match."""
        keyword = keyword.lower()
        return [
            item_id for item_id in item_ids
            if keyword in self.items.get(item_id, {}).get('keywords', set()) or
               keyword in item_id.lower()  # Also search in item_id itself
        ]
    
    def filter_by_freshness(self, item_ids, max_age_hours=24):
        """Filter items that are still fresh (not expired)."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        return [
            item_id for item_id in item_ids
            if self.items.get(item_id, {}).get('created_at', datetime.now()) > cutoff
        ]
    
    def get_different_stores(self, item_ids, exclude_store_id):
        """Get items from different stores (for diversity)."""
        return [
            item_id for item_id in item_ids
            if self.items.get(item_id, {}).get('store_id') != exclude_store_id
        ]


# ============================================================================
# POPULARITY BASELINE (For Cold Start)
# ============================================================================

def build_popularity_baseline(interactions_df, item_metadata=None, 
                              recency_weight=0.3):
    """
    Build popularity baseline considering both interaction count and recency.
    Perfect for "Available Food" section for new users.
    """
    # Count interactions per item
    item_popularity = interactions_df.groupby('item_id').agg({
        'implicit_rating': ['sum', 'count', 'mean'],
        'timestamp': 'max' if 'timestamp' in interactions_df.columns else 'first'
    }).reset_index()
    
    item_popularity.columns = ['item_id', 'total_rating', 'interaction_count', 
                               'avg_rating', 'last_interaction']
    
    # Normalize interaction count
    max_interactions = item_popularity['interaction_count'].max()
    item_popularity['norm_interactions'] = (
        item_popularity['interaction_count'] / max_interactions
        if max_interactions > 0 else 0
    )
    
    # Add recency score (newer items get boost)
    if 'last_interaction' in item_popularity.columns and item_metadata:
        now = datetime.now()
        item_popularity['recency_score'] = item_popularity['item_id'].apply(
            lambda x: 1.0 if (now - item_metadata.items.get(x, {}).get(
                'created_at', now)).days < POPULARITY_DECAY_DAYS else 0.5
        )
    else:
        item_popularity['recency_score'] = 1.0
    
    # Combined popularity score
    item_popularity['popularity_score'] = (
        (1 - recency_weight) * item_popularity['norm_interactions'] +
        recency_weight * item_popularity['recency_score']
    )
    
    return item_popularity.sort_values('popularity_score', ascending=False)


def get_available_food_recommendations(popularity_df, item_metadata, 
                                      postal_code=None, keyword=None, 
                                      max_age_hours=24, n=20):
    """
    Get "Available Food" recommendations for cold start users.
    Applies postal code, keyword, and freshness filters.
    """
    # Start with all items from popularity baseline
    item_ids = popularity_df['item_id'].tolist()
    
    # Apply filters
    if item_metadata:
        # Filter by freshness (food hasn't expired)
        item_ids = item_metadata.filter_by_freshness(item_ids, max_age_hours)
        
        # Filter by postal code if provided
        if postal_code:
            item_ids = item_metadata.filter_by_postal_code(item_ids, postal_code)
        
        # Filter by keyword if provided
        if keyword:
            item_ids = item_metadata.filter_by_keyword(item_ids, keyword)
    
    # Get top N items that passed all filters
    filtered_popularity = popularity_df[popularity_df['item_id'].isin(item_ids)]
    
    return filtered_popularity.head(n)[['item_id', 'popularity_score']].values.tolist()


# ============================================================================
# COLLABORATIVE FILTERING (For "For You" Page)
# ============================================================================

def train_interaction_model(interactions_df):
    """
    Train collaborative filtering model on interaction data.
    Uses same SVD approach but with implicit ratings.
    """
    # Prepare data for Surprise library
    reader = Reader(rating_scale=(0, 10))  # Adjusted for implicit ratings
    dataset = Dataset.load_from_df(
        interactions_df[['user_id', 'item_id', 'implicit_rating']], 
        reader
    )
    trainset = dataset.build_full_trainset()
    
    # Train SVD model
    model = SVD(n_factors=20, random_state=42)
    model.fit(trainset)
    
    print(f"✓ Model trained on {len(interactions_df)} interactions")
    
    return model, trainset


def get_for_you_recommendations(user_id, trainset, model, interactions_df,
                               item_metadata, popularity_df,
                               postal_code=None, keyword=None,
                               max_age_hours=24, n=10,
                               min_interactions=3):
    """
    Get personalized "For You" recommendations.
    
    This combines:
    1. Collaborative filtering for users with history
    2. Popularity for cold start users
    3. Postal code & keyword filtering
    4. Freshness filtering (food expiration)
    5. Store diversity (recommend from different stores)
    """
    
    # Check user interaction history
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
        user_interaction_count = len(trainset.ur[user_inner_id])
    except ValueError:
        # User not in training set - cold start
        user_interaction_count = 0
    
    # COLD START: Use popularity baseline
    if user_interaction_count < min_interactions:
        print(f"Cold start user {user_id} - using popularity baseline")
        return get_available_food_recommendations(
            popularity_df, item_metadata, postal_code, keyword, max_age_hours, n
        )
    
    # WARM USER: Use collaborative filtering
    all_items = trainset.all_items()
    user_inner_id = trainset.to_inner_uid(user_id)
    
    # Get items user has already interacted with
    user_items = set([
        trainset.to_raw_iid(item_id) 
        for item_id, _ in trainset.ur[user_inner_id]
    ])
    
    # Get user's previous purchases to determine preferred stores
    user_history = interactions_df[interactions_df['user_id'] == user_id]
    user_stores = set()
    if item_metadata:
        for item_id in user_items:
            store_id = item_metadata.items.get(item_id, {}).get('store_id')
            if store_id:
                user_stores.add(store_id)
    
    # Predict ratings for all unseen items
    predictions = []
    for item_id in all_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        
        if raw_item_id not in user_items:
            pred = model.predict(user_id, raw_item_id)
            predictions.append((raw_item_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just item IDs for filtering
    candidate_items = [item_id for item_id, _ in predictions]
    
    # Apply filters
    if item_metadata:
        # Freshness filter
        candidate_items = item_metadata.filter_by_freshness(
            candidate_items, max_age_hours
        )
        
        # Postal code filter
        if postal_code:
            candidate_items = item_metadata.filter_by_postal_code(
                candidate_items, postal_code
            )
        
        # Keyword filter
        if keyword:
            candidate_items = item_metadata.filter_by_keyword(
                candidate_items, keyword
            )
    
    # Reconstruct predictions with filtered items
    filtered_predictions = [
        (item_id, score) for item_id, score in predictions 
        if item_id in candidate_items
    ]
    
    # Add diversity: prioritize items from different stores
    if item_metadata and user_stores:
        diverse_recs = []
        same_store_recs = []
        
        for item_id, score in filtered_predictions:
            store_id = item_metadata.items.get(item_id, {}).get('store_id')
            if store_id and store_id not in user_stores:
                diverse_recs.append((item_id, score))
            else:
                same_store_recs.append((item_id, score))
        
        # Mix: 70% from different stores, 30% from same stores
        diverse_count = int(n * 0.7)
        final_recs = diverse_recs[:diverse_count] + same_store_recs[:n - diverse_count]
        
        return final_recs[:n]
    
    return filtered_predictions[:n]


# ============================================================================
# COMPLEMENTARY ITEMS (Butter → Bread logic)
# ============================================================================

def find_complementary_items(item_id, interactions_df, item_metadata, 
                            trainset, n=5):
    """
    Find items frequently purchased together (complementary items).
    E.g., if user bought butter, recommend bread.
    """
    # Find users who interacted with this item
    users_with_item = interactions_df[
        interactions_df['item_id'] == item_id
    ]['user_id'].unique()
    
    if len(users_with_item) == 0:
        return []
    
    # Find what else these users interacted with
    co_occurrences = interactions_df[
        interactions_df['user_id'].isin(users_with_item) &
        (interactions_df['item_id'] != item_id)
    ]
    
    # Count co-occurrences
    complementary = co_occurrences.groupby('item_id').size().reset_index(
        name='co_occurrence_count'
    )
    complementary = complementary.sort_values(
        'co_occurrence_count', ascending=False
    )
    
    # Filter for freshness
    if item_metadata:
        fresh_items = item_metadata.filter_by_freshness(
            complementary['item_id'].tolist()
        )
        complementary = complementary[
            complementary['item_id'].isin(fresh_items)
        ]
    
    return complementary.head(n)[['item_id', 'co_occurrence_count']].values.tolist()


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

class UKFoodSaverRecommender:
    """
    Production-ready recommender system.
    Wraps all functionality for easy API integration.
    """
    
    def __init__(self):
        self.model = None
        self.trainset = None
        self.interactions_df = None
        self.popularity_df = None
        self.item_metadata = ItemMetadata()
        self.last_train_time = None
    
    def train(self, interactions_df):
        """Train the model on interaction data."""
        self.interactions_df = interactions_df
        self.model, self.trainset = train_interaction_model(interactions_df)
        self.popularity_df = build_popularity_baseline(
            interactions_df, self.item_metadata
        )
        self.last_train_time = datetime.now()
        print(f"✓ Model trained at {self.last_train_time}")
    
    def get_recommendations(self, user_id=None, postal_code=None, 
                          keyword=None, n=10):
        """
        Main recommendation endpoint.
        
        Returns:
        - "Available Food" for new users (no user_id or cold start)
        - "For You" personalized recs for users with history
        """
        if user_id is None:
            # No user ID - show popular items
            return {
                'type': 'available_food',
                'recommendations': get_available_food_recommendations(
                    self.popularity_df, self.item_metadata,
                    postal_code, keyword, n=n
                )
            }
        
        # Get personalized recommendations
        recs = get_for_you_recommendations(
            user_id, self.trainset, self.model,
            self.interactions_df, self.item_metadata,
            self.popularity_df, postal_code, keyword, n=n
        )
        
        return {
            'type': 'for_you',
            'user_id': user_id,
            'recommendations': recs
        }
    
    def get_complementary_items(self, item_id, n=5):
        """Get items frequently bought together with given item."""
        return find_complementary_items(
            item_id, self.interactions_df, self.item_metadata,
            self.trainset, n
        )
    
    def log_interaction(self, user_id, item_id, interaction_type, timestamp=None):
        """
        Log a new interaction (for real-time updates).
        In production, this would append to database.
        """
        new_interaction = pd.DataFrame([{
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'timestamp': timestamp or datetime.now(),
            'implicit_rating': INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        }])
        
        self.interactions_df = pd.concat(
            [self.interactions_df, new_interaction], 
            ignore_index=True
        )
        
        # Retrain periodically (e.g., every 100 interactions or daily)
        if len(self.interactions_df) % 100 == 0:
            print("Retraining model with new interactions...")
            self.train(self.interactions_df)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("UKFoodSaver Interaction-Based Recommender System")
    print("=" * 70)
    
    # Example: Load data (adapt to your CSV format)
    # If you have old rating-based data, it will still work
    example_data = pd.DataFrame([
        {'user_id': 'user1', 'item_id': 'butter_store1', 'interaction_type': 'purchase', 'timestamp': datetime.now()},
        {'user_id': 'user1', 'item_id': 'bread_store2', 'interaction_type': 'purchase', 'timestamp': datetime.now()},
        {'user_id': 'user2', 'item_id': 'butter_store1', 'interaction_type': 'add_to_cart', 'timestamp': datetime.now()},
    ])
    
    # Initialize recommender
    recommender = UKFoodSaverRecommender()
    
    # Add item metadata (would come from your database)
    recommender.item_metadata.add_item(
        'butter_store1', 'AB1 2CD', ['butter', 'dairy'], 'store1'
    )
    recommender.item_metadata.add_item(
        'bread_store2', 'AB1 2CD', ['bread', 'bakery'], 'store2'
    )
    
    # Convert interactions and train
    interactions = load_interaction_data(df=example_data)
    recommender.train(interactions)
    
    # Get recommendations
    recs = recommender.get_recommendations(user_id='user1', postal_code='AB1 2CD')
    print("\nRecommendations:", recs)