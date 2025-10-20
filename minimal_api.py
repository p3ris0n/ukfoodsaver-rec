from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from collaborative_filter import *  # Import YOUR existing code
import pandas as pd
from surprise import Dataset, Reader, SVD

app = FastAPI(title="UKFoodSaver Recommendations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your existing model
print("Loading model...")
data = load_data_correctly()
reader = Reader(rating_scale=(0, 2))
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
trainset = dataset.build_full_trainset()
model = SVD()
model.fit(trainset)
popularity_df = build_popularity_baseline(data)
print("✓ Model ready!")

@app.get("/")
def root():
    return {"message": "UKFoodSaver Recommendations", "status": "ready"}

@app.get("/available-food")
def available_food(n: int = Query(20)):
    recs = get_popularity_recommendations(popularity_df, n=n)
    return {
        "type": "available_food",
        "recommendations": [
            {"item_id": item, "score": float(score), "rank": i+1}
            for i, (item, score) in enumerate(recs)
        ]
    }

@app.get("/for-you/{user_id}")
def for_you(user_id: str, n: int = Query(10)):
    try:
        recs = get_hybrid_recommendations(
            user_id, trainset, model, popularity_df, n=n
        )
        return {
            "type": "for_you",
            "user_id": user_id,
            "recommendations": [
                {"item_id": item, "score": float(score), "rank": i+1}
                for i, (item, score) in enumerate(recs)
            ]
        }
    except:
        return available_food(n)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)