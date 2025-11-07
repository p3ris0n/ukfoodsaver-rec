"""
FastAPI Server for UKFoodSaver Recommendation System
Deploy this on Render and share endpoints with Bro Johnson
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime
from contextlib import asynccontextmanager
import pandas as pd
import uvicorn

# Import your recommender system
from interaction_based_recommender import (
    UKFoodSaverRecommender,
    load_interaction_data,
    INTERACTION_WEIGHTS
)

# ============================================================================
# LIFESPAN EVENT HANDLER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    print("=" * 70)
    print("UKFoodSaver Recommendations API Starting...")
    print("=" * 70)
    
    try:
        import os
        
        # Check for data files
        data_files = [
            'data/interactions.csv',
            'data/UKFS_testdata.csv',
            './interactions.csv',
            './UKFS_testdata.csv'
        ]
        
        data_file = None
        for file_path in data_files:
            # Check file exists AND is not empty
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                data_file = file_path
                break
        
        if data_file:
            print(f"Loading data from: {data_file}")
            
            if 'UKFS_testdata' in data_file:
                df = pd.read_csv(data_file)
                if 'rating' in df.columns and 'interaction_type' not in df.columns:
                    df['interaction_type'] = df['rating'].apply(
                        lambda x: 'purchase' if x >= 1.5 else 'view'
                    )
                    df['timestamp'] = pd.Timestamp.now()
                interactions_df = load_interaction_data(df=df)
            else:
                interactions_df = load_interaction_data("data/UKFS_testdata.csv")
            
            recommender.train(interactions_df)
            print(f"✓ Model trained on {len(interactions_df)} interactions")
        else:
            print("⚠️  No valid data files found. Model starting untrained.")
            print("   Use POST /train to initialize the model")
            
    except Exception as e:
        print(f"⚠️  Startup training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Model starting untrained.")
    
    print("=" * 70)
    print("✓ API Ready!")
    print("=" * 70)
    
    yield
    
    # Shutdown code (if needed in the future)
    pass

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="UKFoodSaver Recommendations API",
    description="Recommendation system for food marketplace platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender = UKFoodSaverRecommender()

# ============================================================================
# PYDANTIC MODELS (Request/Response schemas)
# ============================================================================

class InteractionLog(BaseModel):
    user_id: str
    item_id: str
    interaction_type: Literal['view', 'add_to_cart', 'purchase']
    timestamp: Optional[datetime] = None

class ItemMetadataInput(BaseModel):
    item_id: str
    postal_code: str
    keywords: List[str]
    store_id: str
    created_at: Optional[datetime] = None

class RecommendationResponse(BaseModel):
    type: str  # 'available_food' or 'for_you'
    user_id: Optional[str] = None
    recommendations: List[dict]
    postal_code: Optional[str] = None
    keyword: Optional[str] = None
    count: int

class ComplementaryResponse(BaseModel):
    item_id: str
    complementary_items: List[dict]
    count: int

class HealthResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    status: str
    model_trained: bool
    last_train_time: Optional[str]
    total_interactions: int
    total_users: int
    total_items: int

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API welcome endpoint"""
    return {
        "message": "UKFoodSaver Recommendations API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API and model health"""
    if recommender.interactions_df is None:
        return HealthResponse(
            status="not_trained",
            model_trained=False,
            last_train_time=None,
            total_interactions=0,
            total_users=0,
            total_items=0
        )
    
    return HealthResponse(
        status="healthy",
        model_trained=recommender.model is not None,
        last_train_time=recommender.last_train_time.isoformat() if recommender.last_train_time else None,
        total_interactions=len(recommender.interactions_df),
        total_users=recommender.interactions_df['user_id'].nunique(),
        total_items=recommender.interactions_df['item_id'].nunique()
    )

@app.post("/train", tags=["Admin"])
async def train_model(csv_path: str = "data/interactions.csv"):
    """
    Train/retrain the recommendation model.
    In production, this would be called periodically or triggered by data updates.
    """
    try:
        interactions_df = load_interaction_data(csv_path)
        recommender.train(interactions_df)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "interactions_count": len(interactions_df),
            "train_time": recommender.last_train_time.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: Optional[str] = Query(None, description="User ID (optional for cold start)"),
    postal_code: Optional[str] = Query(None, description="Filter by postal code"),
    keyword: Optional[str] = Query(None, description="Search by keyword"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get recommendations for a user.
    
    - **No user_id**: Returns "Available Food" (popular items)
    - **With user_id**: Returns "For You" personalized recommendations
    - **postal_code**: Filter results by location
    - **keyword**: Search for specific food items (e.g., "meat pie", "rice")
    """
    if recommender.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not trained. Call /train endpoint first."
        )
    
    try:
        result = recommender.get_recommendations(
            user_id=user_id,
            postal_code=postal_code,
            keyword=keyword,
            n=n
        )
        
        # Format recommendations for response
        formatted_recs = [
            {
                "item_id": item_id,
                "score": float(score),
                "rank": idx + 1
            }
            for idx, (item_id, score) in enumerate(result['recommendations'])
        ]
        
        return RecommendationResponse(
            type=result['type'],
            user_id=result.get('user_id'),
            recommendations=formatted_recs,
            postal_code=postal_code,
            keyword=keyword,
            count=len(formatted_recs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/available-food", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_available_food(
    postal_code: Optional[str] = Query(None, description="Filter by postal code"),
    keyword: Optional[str] = Query(None, description="Search by keyword"),
    n: int = Query(20, ge=1, le=50, description="Number of items")
):
    """
    Get "Available Food" for cold start users or homepage.
    Returns popular items, optionally filtered by location and keyword.
    """
    return await get_recommendations(
        user_id=None,
        postal_code=postal_code,
        keyword=keyword,
        n=n
    )

@app.get("/for-you/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_for_you(
    user_id: str,
    postal_code: Optional[str] = Query(None, description="Filter by postal code"),
    keyword: Optional[str] = Query(None, description="Search by keyword"),
    n: int = Query(10, ge=1, le=50, description="Number of recommendations")
):
    """
    Get "For You" personalized recommendations for a specific user.
    Based on their interaction history (purchases, add-to-cart).
    """
    return await get_recommendations(
        user_id=user_id,
        postal_code=postal_code,
        keyword=keyword,
        n=n
    )

@app.get("/complementary/{item_id}", response_model=ComplementaryResponse, tags=["Recommendations"])
async def get_complementary_items(
    item_id: str,
    n: int = Query(5, ge=1, le=20, description="Number of complementary items")
):
    """
    Get complementary items (frequently bought together).
    E.g., user bought butter → recommend bread
    """
    if recommender.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not trained. Call /train endpoint first."
        )
    
    try:
        complementary = recommender.get_complementary_items(item_id, n=n)
        
        formatted = [
            {
                "item_id": item_id,
                "co_occurrence_count": int(count),
                "rank": idx + 1
            }
            for idx, (item_id, count) in enumerate(complementary)
        ]
        
        return ComplementaryResponse(
            item_id=item_id,
            complementary_items=formatted,
            count=len(formatted)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@app.post("/interaction", tags=["Tracking"])
async def log_interaction(interaction: InteractionLog):
    """
    Log a user interaction (view, add_to_cart, purchase).
    Frontend should call this endpoint whenever user interacts with items.
    """
    if recommender.interactions_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized. Call /train endpoint first."
        )
    
    try:
        recommender.log_interaction(
            user_id=interaction.user_id,
            item_id=interaction.item_id,
            interaction_type=interaction.interaction_type,
            timestamp=interaction.timestamp
        )
        
        return {
            "status": "success",
            "message": "Interaction logged",
            "user_id": interaction.user_id,
            "item_id": interaction.item_id,
            "interaction_type": interaction.interaction_type,
            "weight": INTERACTION_WEIGHTS[interaction.interaction_type]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log: {str(e)}")

@app.post("/item/metadata", tags=["Admin"])
async def add_item_metadata(item: ItemMetadataInput):
    """
    Add or update item metadata (postal code, keywords, store info).
    Call this when a business owner lists a new food item.
    """
    try:
        recommender.item_metadata.add_item(
            item_id=item.item_id,
            postal_code=item.postal_code,
            keywords=item.keywords,
            store_id=item.store_id,
            created_at=item.created_at
        )
        
        return {
            "status": "success",
            "message": "Item metadata added",
            "item_id": item.item_id,
            "postal_code": item.postal_code,
            "keywords": item.keywords,
            "store_id": item.store_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@app.get("/search", response_model=RecommendationResponse, tags=["Search"])
async def search_items(
    query: str = Query(..., description="Search query (keyword or postal code)"),
    search_type: Literal['keyword', 'postal_code', 'auto'] = Query(
        'auto', 
        description="Search type (auto-detects if not specified)"
    ),
    n: int = Query(20, ge=1, le=50, description="Number of results")
):
    """
    Unified search endpoint supporting both keyword and postal code search.
    
    - **keyword**: Searches item names/descriptions (e.g., "meat pie", "rice")
    - **postal_code**: Searches by location (e.g., "AB1 2CD")
    - **auto**: Automatically detects search type based on query format
    """
    # Auto-detect search type
    if search_type == 'auto':
        # Simple heuristic: if query looks like postal code format, treat as postal code
        if len(query) <= 8 and any(c.isdigit() for c in query):
            search_type = 'postal_code'
        else:
            search_type = 'keyword'
    
    # Route to appropriate search
    if search_type == 'postal_code':
        return await get_available_food(postal_code=query, n=n)
    else:
        return await get_available_food(keyword=query, n=n)

@app.get("/user/{user_id}/history", tags=["User"])
async def get_user_history(
    user_id: str,
    limit: int = Query(20, ge=1, le=100, description="Number of recent interactions")
):
    """
    Get user's interaction history.
    Useful for displaying "Recently Viewed" or "Purchase History".
    """
    if recommender.interactions_df is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        user_history = recommender.interactions_df[
            recommender.interactions_df['user_id'] == user_id
        ].sort_values('timestamp', ascending=False).head(limit)
        
        if len(user_history) == 0:
            return {
                "user_id": user_id,
                "interaction_count": 0,
                "history": []
            }
        
        history = [
            {
                "item_id": row['item_id'],
                "interaction_type": row.get('interaction_type', 'unknown'),
                "implicit_rating": float(row['implicit_rating']),
                "timestamp": row['timestamp'].isoformat() if pd.notna(row.get('timestamp')) else None
            }
            for _, row in user_history.iterrows()
        ]
        
        return {
            "user_id": user_id,
            "interaction_count": len(user_history),
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

@app.get("/stats", tags=["Analytics"])
async def get_statistics():
    """
    Get system statistics and analytics.
    Useful for admin dashboard.
    """
    if recommender.interactions_df is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        df = recommender.interactions_df
        
        # Calculate statistics
        total_interactions = len(df)
        total_users = df['user_id'].nunique()
        total_items = df['item_id'].nunique()
        
        # User engagement
        interactions_per_user = df.groupby('user_id').size()
        avg_interactions_per_user = float(interactions_per_user.mean())
        
        # Item popularity
        interactions_per_item = df.groupby('item_id').size()
        avg_interactions_per_item = float(interactions_per_item.mean())
        
        # Interaction type breakdown
        if 'interaction_type' in df.columns:
            interaction_breakdown = df['interaction_type'].value_counts().to_dict()
        else:
            interaction_breakdown = {}
        
        # Top items
        top_items = recommender.popularity_df.head(10)[['item_id', 'popularity_score']].to_dict('records')
        
        return {
            "total_interactions": total_interactions,
            "total_users": total_users,
            "total_items": total_items,
            "avg_interactions_per_user": avg_interactions_per_user,
            "avg_interactions_per_item": avg_interactions_per_item,
            "interaction_breakdown": interaction_breakdown,
            "top_items": top_items,
            "model_status": "trained" if recommender.model else "not_trained",
            "last_train_time": recommender.last_train_time.isoformat() if recommender.last_train_time else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

# run server.

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "api_server:app",
        host="localhost",
        port=8000,
        reload=True  # auto-reload on code changes
    )