"""
UKFoodSaver Interaction-Based Recommendation System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from surprise import Dataset, Reader, SVD
from scipy.sparse import csr_matrix
from collections import defaultdict
import json

# INTERACTION WEIGHTS CONFIGURATION
 
INTERACTION_WEIGHTS = {
    'view': 1.0,
    'add_to_cart': 3.0,
    'purchase': 5.0
}

POPULARITY_DECAY_DAYS = 2

# DATA LOADING WITH METADATA
 
def load_interaction_data(csv_path=None, df=None):
    """
    Load interaction data and convert to implicit ratings.
    NOW ALSO EXTRACTS ITEM METADATA (store_id, postal_code, location)
    """
    if df is None:
        df = pd.read_csv(csv_path or "data/UKFS_testdata.csv")
    
    print(f"# Loaded raw data: {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Check format and convert
    if 'interaction_type' in df.columns:
        df['implicit_rating'] = df['interaction_type'].map(INTERACTION_WEIGHTS)
        df['implicit_rating'] = df['implicit_rating'].fillna(INTERACTION_WEIGHTS['view'])
    elif 'rating' in df.columns:
        df['implicit_rating'] = df['rating']
    else:
        raise ValueError("DataFrame must have either 'interaction_type' or 'rating' column")
    
    # Add timestamp if missing
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.Timestamp.now()
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Aggregate interactions
    aggregated = df.groupby(['user_id', 'item_id']).agg({
        'implicit_rating': 'sum',
        'timestamp': 'max'
    }).reset_index()
    
    aggregated['implicit_rating'] = aggregated['implicit_rating'].clip(upper=10.0)
    
    print(f"✓ Processed {len(df)} interactions → {len(aggregated)} unique user-item pairs")
    print(f"  Users: {aggregated['user_id'].nunique()}")
    print(f"  Items: {aggregated['item_id'].nunique()}")
    
    return aggregated, df  # Return both aggregated and raw data


def extract_item_metadata(raw_df):
    """
    Extract item metadata from raw dataframe.
    Creates mapping of item_id -> {store_id, postal_code, city, state, country}
    """
    # Get unique items with their metadata
    metadata_cols = ['item_id', 'store_id', 'postal_code', 'city', 'state', 'country']
    available_cols = [col for col in metadata_cols if col in raw_df.columns]
    
    if len(available_cols) < 2:
        print("⚠️  No metadata columns found in CSV")
        return {}
    
    # Get first occurrence of each item (assuming metadata doesn't change)
    item_metadata_df = raw_df[available_cols].drop_duplicates(subset=['item_id'])
    
    metadata_dict = {}
    for _, row in item_metadata_df.iterrows():
        item_id = row['item_id']
        metadata_dict[item_id] = {
            'store_id': row.get('store_id', ''),
            'postal_code': row.get('postal_code', ''),
            'city': row.get('city', ''),
            'state': row.get('state', ''),
            'country': row.get('country', ''),
            'created_at': datetime.now()  # Can be updated with real creation time
        }
    
    print(f"✓ Extracted metadata for {len(metadata_dict)} items")
    print(f"  Sample item: {list(metadata_dict.keys())[0]} -> {metadata_dict[list(metadata_dict.keys())[0]]}")
    
    return metadata_dict


 # ITEM METADATA CLASS (ENHANCED)
 
class ItemMetadata:
    """Manages item metadata for filtering and recommendations."""
    
    def __init__(self):
        self.items = {}
    
    def load_from_dict(self, metadata_dict):
        """Load metadata from dictionary extracted from CSV."""
        self.items = metadata_dict
        print(f"✓ ItemMetadata loaded {len(self.items)} items")
    
    def add_item(self, item_id, store_id, postal_code, city='', state='', country='', keywords=None, created_at=None):
        """Add or update item metadata."""
        self.items[item_id] = {
            'store_id': store_id,
            'postal_code': postal_code,
            'city': city,
            'state': state,
            'country': country,
            'keywords': set(kw.lower() for kw in (keywords or [])),
            'created_at': created_at or datetime.now()
        }
    
    def get_item_info(self, item_id):
        """Get all metadata for an item."""
        return self.items.get(item_id, {})
    
    def filter_by_postal_code(self, item_ids, postal_code):
        """Filter items by postal code."""
        if not postal_code:
            return item_ids
        
        postal_code_upper = postal_code.upper() if isinstance(postal_code, str) else ''
        return [
            item_id for item_id in item_ids 
            if str(self.items.get(item_id, {}).get('postal_code', '')).upper() == postal_code_upper
        ]
    
    def filter_by_keyword(self, item_ids, keyword):
        """Filter items by keyword match."""
        if not keyword:
            return item_ids
        
        keyword_lower = keyword.lower()
        return [
            item_id for item_id in item_ids
            if (keyword_lower in str(item_id).lower() or any (keyword_lower in str(value).lower()
                for value in self.items.get(item_id, {}).values()))
        ]
    
    def filter_by_freshness(self, item_ids, max_age_hours=24):
        """Filter items that are still fresh (not expired)."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        return [
            item_id for item_id in item_ids
            if self.items.get(item_id, {}).get('created_at', datetime.now()) > cutoff
        ]
    
    def get_store_id(self, item_id):
        """Get store ID for an item."""
        return self.items.get(item_id, {}).get('store_id', '')


 # POPULARITY BASELINE
 
def build_popularity_baseline(interactions_df, item_metadata=None, recency_weight=0.3):
    """Build popularity baseline with item metadata."""
    item_popularity = interactions_df.groupby('item_id').agg({
        'implicit_rating': ['sum', 'count', 'mean'],
        'timestamp': 'max'
    }).reset_index()
    
    item_popularity.columns = ['item_id', 'total_rating', 'interaction_count', 
                               'avg_rating', 'last_interaction']
    
    max_interactions = item_popularity['interaction_count'].max()
    item_popularity['norm_interactions'] = (
        item_popularity['interaction_count'] / max_interactions
        if max_interactions > 0 else 0
    )
    
    item_popularity['recency_score'] = 1.0
    
    item_popularity['popularity_score'] = (
        (1 - recency_weight) * item_popularity['norm_interactions'] +
        recency_weight * item_popularity['recency_score']
    )
    
    return item_popularity.sort_values('popularity_score', ascending=False)


 # RECOMMENDATION FUNCTIONS
 
def get_available_food_recommendations(popularity_df, item_metadata, 
                                      postal_code=None, keyword=None, 
                                      max_age_hours=24, n=20):
    """
    Get "Available Food" recommendations with full metadata.
    Returns: [(item_id, score, metadata), ...]
    """
    item_ids = popularity_df['item_id'].tolist()
    
    # Apply filters
    if item_metadata:
        if postal_code:
            item_ids = item_metadata.filter_by_postal_code(item_ids, postal_code)
        if keyword:
            item_ids = item_metadata.filter_by_keyword(item_ids, keyword)
    
    # Get filtered items with scores
    filtered_popularity = popularity_df[popularity_df['item_id'].isin(item_ids)]
    
    # Add metadata to each recommendation
    results = []
    for _, row in filtered_popularity.head(n).iterrows():
        item_id = row['item_id']
        score = row['popularity_score']
        metadata = item_metadata.get_item_info(item_id) if item_metadata else {}
        
        results.append({
            'item_id': item_id,
            'score': float(score),
            'store_id': metadata.get('store_id', ''),
            'postal_code': metadata.get('postal_code', ''),
            'city': metadata.get('city', ''),
            'state': metadata.get('state', ''),
            'country': metadata.get('country', '')
        })
    
    return results


def train_interaction_model(interactions_df):
    """Train collaborative filtering model."""
    reader = Reader(rating_scale=(0, 10))
    dataset = Dataset.load_from_df(
        interactions_df[['user_id', 'item_id', 'implicit_rating']], 
        reader
    )
    trainset = dataset.build_full_trainset()
    
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
    Get personalized "For You" recommendations with full metadata.
    """
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
        user_interaction_count = len(trainset.ur[user_inner_id])
    except ValueError:
        user_interaction_count = 0
    
    # COLD START
    if user_interaction_count < min_interactions:
        return get_available_food_recommendations(
            popularity_df, item_metadata, postal_code, keyword, max_age_hours, n
        )
    
    # WARM USER
    all_items = trainset.all_items()
    user_inner_id = trainset.to_inner_uid(user_id)
    
    user_items = set([
        trainset.to_raw_iid(item_id) 
        for item_id, _ in trainset.ur[user_inner_id]
    ])
    
    # Get user's preferred stores for diversity
    user_stores = set()
    if item_metadata:
        for item_id in user_items:
            store_id = item_metadata.get_store_id(item_id)
            if store_id:
                user_stores.add(store_id)
    
    # Predict ratings
    predictions = []
    for item_id in all_items:
        raw_item_id = trainset.to_raw_iid(item_id)
        if raw_item_id not in user_items:
            pred = model.predict(user_id, raw_item_id)
            predictions.append((raw_item_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Apply filters
    candidate_items = [item_id for item_id, _ in predictions]
    
    if item_metadata:
        if postal_code:
            candidate_items = item_metadata.filter_by_postal_code(candidate_items, postal_code)
        if keyword:
            candidate_items = item_metadata.filter_by_keyword(candidate_items, keyword)
    
    # Reconstruct with metadata
    filtered_predictions = [
        (item_id, score) for item_id, score in predictions 
        if item_id in candidate_items
    ]
    
    # Add store diversity
    if item_metadata and user_stores:
        diverse_recs = []
        same_store_recs = []
        
        for item_id, score in filtered_predictions:
            store_id = item_metadata.get_store_id(item_id)
            if store_id and store_id not in user_stores:
                diverse_recs.append((item_id, score))
            else:
                same_store_recs.append((item_id, score))
        
        diverse_count = int(n * 0.7)
        final_predictions = diverse_recs[:diverse_count] + same_store_recs[:n - diverse_count]
    else:
        final_predictions = filtered_predictions
    
    # Format with metadata
    results = []
    for item_id, score in final_predictions[:n]:
        metadata = item_metadata.get_item_info(item_id) if item_metadata else {}
        results.append({
            'item_id': item_id,
            'score': float(score),
            'store_id': metadata.get('store_id', ''),
            'postal_code': metadata.get('postal_code', ''),
            'city': metadata.get('city', ''),
            'state': metadata.get('state', ''),
            'country': metadata.get('country', '')
        })
    
    return results


def find_complementary_items(item_id, interactions_df, item_metadata, trainset, n=5):
    """Find complementary items with metadata."""
    users_with_item = interactions_df[
        interactions_df['item_id'] == item_id
    ]['user_id'].unique()
    
    if len(users_with_item) == 0:
        return []
    
    co_occurrences = interactions_df[
        (interactions_df['user_id'].isin(users_with_item)) &
        (interactions_df['item_id'] != item_id)
    ]
    
    complementary = co_occurrences.groupby('item_id').size().reset_index(
        name='co_occurrence_count'
    )
    complementary = complementary.sort_values('co_occurrence_count', ascending=False)
    
    # Add metadata
    results = []
    for _, row in complementary.head(n).iterrows():
        comp_item_id = row['item_id']
        count = row['co_occurrence_count']
        metadata = item_metadata.get_item_info(comp_item_id) if item_metadata else {}
        
        results.append({
            'item_id': comp_item_id,
            'co_occurrence_count': int(count),
            'store_id': metadata.get('store_id', ''),
            'postal_code': metadata.get('postal_code', ''),
            'city': metadata.get('city', ''),
            'state': metadata.get('state', ''),
            'country': metadata.get('country', '')
        })
    
    return results


# Recommender Class
 
class UKFoodSaverRecommender:
    """Production-ready recommender with full metadata support."""
    
    def __init__(self):
        self.model = None
        self.trainset = None
        self.interactions_df = None
        self.popularity_df = None
        self.item_metadata = ItemMetadata()
        self.last_train_time = None
    
    def train(self, interactions_df, raw_df=None):
        """
        Train the model on interaction data.
        If raw_df provided, also extract and load item metadata.
        """
        self.interactions_df = interactions_df
        
        # Extract and load metadata if raw_df provided
        if raw_df is not None:
            metadata_dict = extract_item_metadata(raw_df)
            self.item_metadata.load_from_dict(metadata_dict)
        
        # Train model
        self.model, self.trainset = train_interaction_model(interactions_df)
        self.popularity_df = build_popularity_baseline(interactions_df, self.item_metadata)
        self.last_train_time = datetime.now()
        
        print(f"✓ Training completed at {self.last_train_time}")
        print(f"  {len(self.item_metadata.items)} items with metadata")
    
    def get_recommendations(self, user_id=None, postal_code=None, keyword=None, n=10):
        """Main recommendation endpoint with metadata."""
        if user_id is None:
            return {
                'type': 'available_food',
                'recommendations': get_available_food_recommendations(
                    self.popularity_df, self.item_metadata,
                    postal_code, keyword, n=n
                )
            }
        
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
        """Get complementary items with metadata."""
        return find_complementary_items(
            item_id, self.interactions_df, self.item_metadata,
            self.trainset, n
        )
    
    def log_interaction(self, user_id, item_id, interaction_type, timestamp=None):
        """Log a new interaction."""
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
        
        print(f"✓ Logged: {user_id} → {item_id} ({interaction_type})")


if __name__ == "__main__":
    print("UKFoodSaver Interaction-Based Recommender System")
    print("=" * 70)
    
    # Test with real data
    import os
    if os.path.exists('data/UKFS_testdata.csv'):
        print("\n1. Loading data...")
        interactions, raw_df = load_interaction_data('data/UKFS_testdata.csv')
        
        print("\n2. Initializing recommender...")
        recommender = UKFoodSaverRecommender()
        
        print("\n3. Training model...")
        recommender.train(interactions, raw_df)
        
        print("\n4. Testing recommendations...")
        
        # Test available food
        print("\n--- Available Food (with metadata) ---")
        cold_recs = recommender.get_recommendations(n=3)
        for i, rec in enumerate(cold_recs['recommendations'][:3], 1):
            print(f"{i}. {rec['item_id']}")
            print(f"   Store: {rec['store_id']}")
            print(f"   Location: {rec['city']}, {rec['state']} {rec['postal_code']}")
            print(f"   Score: {rec['score']:.3f}\n")
        
        # Test personalized
        test_user = interactions['user_id'].iloc[0]
        print(f"\n--- For You ({test_user}) ---")
        personal_recs = recommender.get_recommendations(test_user, n=3)
        for i, rec in enumerate(personal_recs['recommendations'][:3], 1):
            print(f"{i}. {rec['item_id']}")
            print(f"   Store: {rec['store_id']}")
            print(f"   Location: {rec['city']}, {rec['state']} {rec['postal_code']}")
            print(f"   Score: {rec['score']:.3f}\n")
        
        print("=" * 70)
        print("✓ System ready with full metadata integration!")