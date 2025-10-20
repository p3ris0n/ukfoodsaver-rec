import pandas as pd
from fastapi import FastAPI, Query

app = FastAPI()
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

# ... (keep imports and app setup)
def load_data_correctly():
    # This is a placeholder. In a real application, you'd load your data here.
    # For example, from a CSV file, a database, or an API.
    # Ensure the returned DataFrame has 'user_id', 'item_id', and 'rating' columns.
    sample_data = {
        'user_id': ['user_a', 'user_a', 'user_b', 'user_b', 'user_c'],
        'item_id': ['item_1', 'item_2', 'item_1', 'item_3', 'item_2'],
        'rating': [5, 4, 3, 5, 2]
    }
    return pd.DataFrame(sample_data)
# Load model on startup
print("Loading model...")
data = load_data_correctly()

# Create user-item matrix
user_ids = data['user_id'].unique()
item_ids = data['item_id'].unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {item: i for i, item in enumerate(item_ids)}

rows = [user_to_idx[u] for u in data['user_id']]
cols = [item_to_idx[item] for item in data['item_id']]
values = data['rating'].values

interaction_matrix = csr_matrix((values, (rows, cols)))

# Train NMF model
model = NMF(n_components=20, random_state=42)
user_features = model.fit_transform(interaction_matrix)
item_features = model.components_

# Build popularity baseline
popularity_df = data.groupby('item_id').agg({
    'rating': ['count', 'mean']
}).reset_index()
popularity_df.columns = ['item_id', 'interaction_count', 'avg_rating']
popularity_df['popularity_score'] = (
    0.7 * (popularity_df['interaction_count'] / popularity_df['interaction_count'].max()) +
    0.3 * (popularity_df['avg_rating'] / popularity_df['avg_rating'].max())
)
popularity_df = popularity_df.sort_values('popularity_score', ascending=False)

print("✓ Model ready!")

@app.get("/")
def root():
    return {"message": "UKFoodSaver Recommendations API", "status": "ready"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "total_interactions": len(data),
        "total_users": len(user_ids),
        "total_items": len(item_ids)
    }

@app.get("/available-food")
def available_food(n: int = Query(20, ge=1, le=50)):
    recs = popularity_df.head(n)[['item_id', 'popularity_score']].values.tolist()
    return {
        "type": "available_food",
        "recommendations": [
            {"item_id": item, "score": float(score), "rank": i+1}
            for i, (item, score) in enumerate(recs)
        ],
        "count": len(recs)
    }

@app.get("/for-you/{user_id}")
def for_you(user_id: str, n: int = Query(10, ge=1, le=50)):
    try:
        if user_id not in user_to_idx:
            # Cold start - return popular items
            return available_food(n)
        
        # Get user's feature vector
        user_idx = user_to_idx[user_id]
        user_vec = user_features[user_idx]
        
        # Predict ratings for all items
        predicted_ratings = user_vec.dot(item_features)
        
        # Get items user hasn't interacted with
        user_items = set(data[data['user_id'] == user_id]['item_id'])
        
        # Sort and filter
        item_scores = [
            (item_ids[i], predicted_ratings[i]) 
            for i in range(len(item_ids))
            if item_ids[i] not in user_items
        ]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "type": "for_you",
            "user_id": user_id,
            "recommendations": [
                {"item_id": item, "score": float(score), "rank": i+1}
                for i, (item, score) in enumerate(item_scores[:n])
            ],
            "count": min(n, len(item_scores))
        }
    except Exception as e:
        print(f"Error: {e}")
        return available_food(n)

@app.get("/search")
def search(query: str = Query(...), n: int = Query(20)):
    recs = popularity_df.head(50)[['item_id', 'popularity_score']].values.tolist()
    query_lower = query.lower()
    recs = [(item, score) for item, score in recs if query_lower in item.lower()]
    
    return {
        "type": "search_results",
        "query": query,
        "recommendations": [
            {"item_id": item, "score": float(score), "rank": i+1}
            for i, (item, score) in enumerate(recs[:n])
        ],
        "count": len(recs[:n])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)