# render_api.py - FIXED VERSION
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Optional, List
import os

app = FastAPI(
    title="UKFoodSaver Recommendations API",
    description="Food recommendation system for UKFoodSaver platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data on startup
def load_data():
    """Load the UKFS test data"""
    try:
        # Try different possible file paths
        possible_paths = [
            'data/UKFS_testdata.csv',
            './data/UKFS_testdata.csv',
            'UKFS_testdata.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"✓ Data loaded from {path}: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find UKFS_testdata.csv in any expected location")
            
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Don't return sample data - raise the error so we know it failed
        raise e

# Global data - will fail if data can't be loaded
try:
    data = load_data()
    print("✓ Data successfully loaded!")
    print(f"Sample items: {list(data['item_id'].unique())[:10]}")
except Exception as e:
    print(f"❌ CRITICAL: Could not load data: {e}")
    # Exit if data can't be loaded
    import sys
    sys.exit(1)

# Build popularity model
def build_popularity_model(df):
    """Build popularity-based recommendations using REAL cuisine names"""
    item_popularity = df.groupby('item_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    item_popularity.columns = ['item_id', 'interaction_count', 'avg_rating']
    
    # Normalize scores
    max_count = item_popularity['interaction_count'].max()
    max_rating = item_popularity['avg_rating'].max() if item_popularity['avg_rating'].max() > 0 else 1
    
    item_popularity['popularity_score'] = (
        0.7 * (item_popularity['interaction_count'] / max_count) +
        0.3 * (item_popularity['avg_rating'] / max_rating)
    )
    
    return item_popularity.sort_values('popularity_score', ascending=False)

popularity_df = build_popularity_model(data)

@app.get("/")
async def root():
    return {
        "message": "UKFoodSaver Recommendations API",
        "status": "active",
        "data_loaded": True,
        "total_items": len(popularity_df),
        "endpoints": {
            "health": "/health",
            "available_food": "/available-food",
            "for_you": "/for-you/{user_id}",
            "search": "/search?query=your_query",
            "items": "/items"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": True,
        "total_interactions": len(data),
        "total_users": data['user_id'].nunique(),
        "total_items": data['item_id'].nunique(),
        "top_cuisines": list(popularity_df.head(5)['item_id'].tolist())
    }

@app.get("/available-food")
async def available_food(
    n: int = Query(20, ge=1, le=50, description="Number of recommendations")
):
    """Get popular food items (cold start recommendations)"""
    recommendations = popularity_df.head(n)[['item_id', 'popularity_score']]
    
    return {
        "type": "available_food",
        "recommendations": [
            {
                "item_id": row['item_id'],
                "score": float(row['popularity_score']),
                "rank": idx + 1
            }
            for idx, row in recommendations.iterrows()
        ],
        "count": len(recommendations)
    }

@app.get("/for-you/{user_id}")
async def for_you_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """Get personalized recommendations for a user"""
    # Check if user exists
    if user_id not in data['user_id'].values:
        # Return popular items for new users
        return await available_food(n)
    
    # Get user's rated items
    user_ratings = data[data['user_id'] == user_id]
    rated_items = set(user_ratings['item_id'])
    
    # Simple collaborative filtering: find similar users
    try:
        # Get users who rated similar items
        user_items = set(user_ratings['item_id'])
        similar_users = data[
            (data['item_id'].isin(user_items)) & 
            (data['user_id'] != user_id)
        ]['user_id'].unique()
        
        # Get items liked by similar users that current user hasn't rated
        if len(similar_users) > 0:
            similar_user_ratings = data[
                (data['user_id'].isin(similar_users)) & 
                (data['rating'] >= 1.5)  # Only consider positive ratings
            ]
            
            # Filter out items user already rated
            recommendations = similar_user_ratings[
                ~similar_user_ratings['item_id'].isin(rated_items)
            ]
            
            # Score by frequency and average rating
            item_scores = recommendations.groupby('item_id').agg({
                'user_id': 'count',
                'rating': 'mean'
            }).reset_index()
            
            item_scores.columns = ['item_id', 'frequency', 'avg_rating']
            item_scores['score'] = (
                0.6 * (item_scores['frequency'] / item_scores['frequency'].max()) +
                0.4 * (item_scores['avg_rating'] / 2.0)  # Normalize to 0-2 scale
            )
            
            recommendations = item_scores.nlargest(n, 'score')
            
            return {
                "type": "for_you",
                "user_id": user_id,
                "recommendations": [
                    {
                        "item_id": row['item_id'],
                        "score": float(row['score']),
                        "rank": idx + 1
                    }
                    for idx, row in recommendations.iterrows()
                ],
                "count": len(recommendations)
            }
    
    except Exception as e:
        print(f"Error in personalized recommendations: {e}")
    
    # Fallback to popular items
    fallback_recs = popularity_df[~popularity_df['item_id'].isin(rated_items)].head(n)
    
    return {
        "type": "for_you",
        "user_id": user_id,
        "recommendations": [
            {
                "item_id": row['item_id'],
                "score": float(row['popularity_score']),
                "rank": idx + 1
            }
            for idx, row in fallback_recs.iterrows()
        ],
        "count": len(fallback_recs)
    }

@app.get("/search")
async def search_items(
    query: str = Query(..., description="Search query"),
    n: int = Query(20, ge=1, le=50, description="Number of results")
):
    """Search for food items by keyword"""
    query_lower = query.lower()
    
    # Filter items by keyword match
    matching_items = popularity_df[
        popularity_df['item_id'].str.lower().str.contains(query_lower, na=False)
    ].head(n)
    
    return {
        "type": "search_results",
        "query": query,
        "recommendations": [
            {
                "item_id": row['item_id'],
                "score": float(row['popularity_score']),
                "rank": idx + 1
            }
            for idx, row in matching_items.iterrows()
        ],
        "count": len(matching_items)
    }

@app.get("/users")
async def get_users():
    """Get list of all users in the system"""
    users = data['user_id'].unique().tolist()
    return {
        "users": users,
        "count": len(users)
    }

@app.get("/items")
async def get_items():
    """Get list of all food items in the system"""
    items = data['item_id'].unique().tolist()
    return {
        "items": items,
        "count": len(items),
        "sample_items": items[:20]  # Show first 20 items
    }

@app.get("/debug/data")
async def debug_data():
    """Debug endpoint to check what data is loaded"""
    return {
        "data_loaded": True,
        "total_rows": len(data),
        "columns": list(data.columns),
        "sample_data": data.head(10).to_dict('records'),
        "unique_items_count": data['item_id'].nunique(),
        "unique_users_count": data['user_id'].nunique(),
        "top_5_items": popularity_df.head(5)[['item_id', 'popularity_score']].to_dict('records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)