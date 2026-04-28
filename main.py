"""
Movie Recommender FastAPI Backend
==================================
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
import threading

app = FastAPI(title="Movie Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH   = "svd_model.pkl"
MOVIES_PATH  = "movies.csv"
RATINGS_PATH = "ratings.csv"

svd        = None
movies_df  = None
ratings_df = None

# In-memory store for session ratings: { (user_id, movie_id): rating }
user_ratings: dict = {}

# Lock to prevent concurrent retraining conflicts
retrain_lock = threading.Lock()

# ── Startup ────────────────────────────────────────────────────────────────────
def load_resources():
    global svd, movies_df, ratings_df
    try:
        from surprise import dump as sdump
        _, svd = sdump.load(MODEL_PATH)
        print("SVD model loaded")
    except Exception as e:
        print(f"Could not load model ({e}). /recommend will return demo data.")

    try:
        movies_df = pd.read_csv(MOVIES_PATH)
        movies_df["title"] = (
            movies_df["title"]
            .str.replace(r"\(\d{4}\)", "", regex=True)
            .str.strip()
        )
        print(f"Movies loaded: {len(movies_df)} rows")
    except Exception as e:
        print(f"Could not load movies.csv ({e}).")

    try:
        ratings_df = pd.read_csv(RATINGS_PATH)
        print(f"Ratings loaded: {len(ratings_df)} rows")
    except Exception as e:
        print(f"Could not load ratings.csv ({e}).")

load_resources()

# ── Helpers ────────────────────────────────────────────────────────────────────
def generate_session_user_id() -> int:
    """Generate a unique user ID that won't clash with existing dataset IDs."""
    base = (ratings_df["userId"].max() + 1) if ratings_df is not None else 10000
    return base + random.randint(0, 99999)

def retrain_model():
    """Merge in-memory ratings with original dataset and retrain SVD in place."""
    global svd
    if svd is None or ratings_df is None:
        return
    with retrain_lock:
        try:
            from surprise import Dataset, Reader, SVD as SurpriseSVD
            new_rows = [
                {"userId": uid, "movieId": mid, "rating": rat}
                for (uid, mid), rat in user_ratings.items()
            ]
            combined = pd.concat([ratings_df, pd.DataFrame(new_rows)], ignore_index=True)
            reader   = Reader(rating_scale=(0.5, 5.0))
            data     = Dataset.load_from_df(combined[["userId", "movieId", "rating"]], reader)
            trainset = data.build_full_trainset()
            svd.fit(trainset)
            print(f"Model retrained with {len(new_rows)} new rating(s)")
        except Exception as e:
            print(f"Retrain failed: {e}")

def get_top_n(user_id: int, n: int = 10) -> List[dict]:
    if svd is None or movies_df is None:
        demo = [
            {"movieId": 318,  "title": "Shawshank Redemption, The",      "genres": "Crime|Drama",                 "predicted_rating": 4.95},
            {"movieId": 527,  "title": "Schindler's List",                "genres": "Drama|War",                   "predicted_rating": 4.88},
            {"movieId": 296,  "title": "Pulp Fiction",                    "genres": "Comedy|Crime|Drama|Thriller", "predicted_rating": 4.82},
            {"movieId": 356,  "title": "Forrest Gump",                    "genres": "Comedy|Drama|Romance|War",    "predicted_rating": 4.75},
            {"movieId": 593,  "title": "Silence of the Lambs, The",       "genres": "Crime|Horror|Thriller",       "predicted_rating": 4.71},
            {"movieId": 858,  "title": "Godfather, The",                  "genres": "Crime|Drama",                 "predicted_rating": 4.68},
            {"movieId": 50,   "title": "Usual Suspects, The",             "genres": "Crime|Mystery|Thriller",      "predicted_rating": 4.60},
            {"movieId": 608,  "title": "Fargo",                           "genres": "Comedy|Crime|Drama|Thriller", "predicted_rating": 4.55},
            {"movieId": 1193, "title": "One Flew Over the Cuckoo's Nest", "genres": "Drama",                       "predicted_rating": 4.50},
            {"movieId": 260,  "title": "Star Wars: Episode IV",           "genres": "Action|Adventure|Sci-Fi",     "predicted_rating": 4.45},
        ]
        return demo[:n]

    # Exclude movies already rated (original dataset + in-memory)
    original_rated = set()
    if ratings_df is not None:
        original_rated = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())
    memory_rated = {mid for (uid, mid) in user_ratings if uid == user_id}
    rated_ids    = original_rated | memory_rated

    all_ids = movies_df["movieId"].tolist()
    unrated = [mid for mid in all_ids if mid not in rated_ids]

    preds = [svd.predict(user_id, mid) for mid in unrated]
    top   = sorted(preds, key=lambda x: x.est, reverse=True)[:n]

    results = []
    for p in top:
        row = movies_df[movies_df["movieId"] == p.iid]
        if not row.empty:
            r = row.iloc[0]
            results.append({
                "movieId":          int(p.iid),
                "title":            r["title"],
                "genres":           r["genres"],
                "predicted_rating": round(float(p.est), 2),
            })
    return results

# ── Pydantic models ────────────────────────────────────────────────────────────
class MovieRec(BaseModel):
    movieId: int
    title: str
    genres: str
    predicted_rating: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRec]

class SearchResult(BaseModel):
    movieId: int
    title: str
    genres: str

class RateRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float

class RateResponse(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    message: str

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/api/session")
def new_session():
    """Generate a fresh unique user ID for a new session."""
    return {"user_id": generate_session_user_id()}

@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int = Query(...),
    n: int       = Query(10, ge=1, le=50),
):
    retrain_model()
    return {"user_id": user_id, "recommendations": get_top_n(user_id, n)}

@app.get("/api/search", response_model=List[SearchResult])
def search_movies(q: str = Query(..., min_length=1)):
    if movies_df is None:
        raise HTTPException(status_code=503, detail="movies.csv not loaded")
    mask = movies_df["title"].str.contains(q, case=False, na=False)
    return movies_df[mask].head(20)[["movieId", "title", "genres"]].to_dict(orient="records")

@app.post("/api/rate", response_model=RateResponse)
def rate_movie(body: RateRequest):
    if not (0.5 <= body.rating <= 5.0):
        raise HTTPException(status_code=422, detail="Rating must be between 0.5 and 5.0")
    user_ratings[(body.user_id, body.movie_id)] = body.rating
    return {
        "user_id":  body.user_id,
        "movie_id": body.movie_id,
        "rating":   body.rating,
        "message":  "Rating saved and model updated.",
    }

@app.get("/api/my-ratings")
def get_my_ratings(user_id: int = Query(...)):
    rows = [
        {"movieId": mid, "rating": rat}
        for (uid, mid), rat in user_ratings.items()
        if uid == user_id
    ]
    if movies_df is not None and rows:
        id_to_title = movies_df.set_index("movieId")["title"].to_dict()
        id_to_genre = movies_df.set_index("movieId")["genres"].to_dict()
        for r in rows:
            r["title"]  = id_to_title.get(r["movieId"], "Unknown")
            r["genres"] = id_to_genre.get(r["movieId"], "")
    return {"user_id": user_id, "ratings": rows}

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_loaded":   svd is not None,
        "movies_loaded":  movies_df is not None,
        "ratings_loaded": ratings_df is not None,
        "memory_ratings": len(user_ratings),
    }