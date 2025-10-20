# render_api.py - Fixed version for Render deployment
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
        df = pd.read_csv('data/UKFS_testdata.csv')
        print(f"✓ Data loaded: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
        return df
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        # Return sample data if file not found
        """return pd.DataFrame({
            'user_id': ['U1001', 'U1002', 'U1003'],
            'item_id': ['Mexican', 'American', 'Italian'],
            'rating': [2, 1, 2]
        })"""
        raise Exception(f"Could not load data file: {e}")

# Global data
data = load_data()

# Build popularity model
def build_popularity_model(df):
    """Build popularity-based recommendations"""
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
        "endpoints": {
            "health": "/health",
            "available_food": "/available-food",
            "for_you": "/for-you/{user_id}",
            "search": "/search?query=your_query"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_interactions": len(data),
        "total_users": data['user_id'].nunique(),
        "total_items": data['item_id'].nunique(),
        "model_ready": True
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
        "count": len(items)
    }

@app.get("/test")
async def test_endpoint():
    return {"message": "API is working", "status": "ok"}

@app.get("/debug/files")
async def debug_files():
    """Check what files exist in the deployment"""
    import os
    files = {}
    
    # Check common directories
    paths_to_check = ['.', './data']
    
    for path in paths_to_check:
        try:
            if os.path.exists(path):
                files[path] = os.listdir(path)
            else:
                files[path] = "PATH_NOT_FOUND"
        except Exception as e:
            files[path] = f"ERROR: {e}"
    
    return {
        "current_working_dir": os.getcwd(),
        "files_in_directories": files,
        "data_file_exists": os.path.exists('data/UKFS_testdata.csv')
    }

@app.get("/debug/data-source")
async def debug_data_source():
    """Check where the data is coming from"""
    sample_data_users = ['U1001', 'U1002', 'U1003']
    sample_data_items = ['item_1', 'item_2', 'item_3']
    
    # Check if we're using sample data
    is_sample_data = (data['user_id'].isin(sample_data_users).any() and 
                     data['item_id'].isin(sample_data_items).any())
    
    return {
        "is_using_sample_data": is_sample_data,
        "data_source": "sample_data" if is_sample_data else "real_data",
        "total_rows": len(data),
        "unique_items": list(data['item_id'].unique())[:10],  # Show first 10 items
        "data_file_found": os.path.exists('data/UKFS_testdata.csv')
    }

@app.get("/debug/files")
async def debug_files():
    """Check what files exist in the deployment"""
    import os
    files = {}
    
    # Check common directories
    paths_to_check = [
        '.',
        './data',
        '/opt/render/project/src',
        '/opt/render/project/src/data'
    ]
    
    for path in paths_to_check:
        try:
            if os.path.exists(path):
                files[path] = os.listdir(path)
            else:
                files[path] = "PATH_NOT_FOUND"
        except Exception as e:
            files[path] = f"ERROR: {e}"
    
    return {
        "current_working_dir": os.getcwd(),
        "files_in_directories": files,
        "data_file_exists": os.path.exists('data/UKFS_testdata.csv')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)