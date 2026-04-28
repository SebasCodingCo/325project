"""
Movie Recommender FastAPI Backend
==================================
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
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

# Build a lookup dict { movieId: set_of_genres } once at load time for fast filtering
movie_genres_lookup: dict = {}

retrain_lock  = threading.Lock()
needs_retrain = False

# ── Startup ────────────────────────────────────────────────────────────────────
def load_resources():
    global svd, movies_df, ratings_df, movie_genres_lookup
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
        # Build genre lookup for fast filtering
        movie_genres_lookup = {
            int(row["movieId"]): set(str(row["genres"]).split("|"))
            for _, row in movies_df.iterrows()
        }
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
    existing = set(ratings_df["userId"].tolist()) if ratings_df is not None else set()
    base = max(existing) + 1 if existing else 10000
    candidate = base + random.randint(0, 99999)
    while candidate in existing:
        candidate = base + random.randint(0, 99999)
    return int(candidate)

def session_exists(user_id: int) -> bool:
    if ratings_df is None:
        return False
    return int(user_id) in ratings_df["userId"].values

def save_rating(user_id: int, movie_id: int, rating: float):
    global ratings_df, needs_retrain
    new_row = pd.DataFrame([{
        "userId":    user_id,
        "movieId":   movie_id,
        "rating":    rating,
        "timestamp": 0,
    }])
    ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    ratings_df.to_csv(RATINGS_PATH, index=False)
    needs_retrain = True

def retrain_model():
    global svd, needs_retrain
    if not needs_retrain:
        return
    if svd is None or ratings_df is None:
        return
    with retrain_lock:
        try:
            from surprise import Dataset, Reader
            # Exclude placeholder rows (movieId == -1)
            clean = ratings_df[ratings_df["movieId"] != -1]
            reader   = Reader(rating_scale=(0.5, 5.0))
            data     = Dataset.load_from_df(clean[["userId", "movieId", "rating"]], reader)
            trainset = data.build_full_trainset()
            svd.fit(trainset)
            needs_retrain = False
            print(f"Model retrained on {len(clean)} ratings")
        except Exception as e:
            print(f"Retrain failed: {e}")

def get_top_n(user_id: int, n: int = 10, genres: List[str] = None) -> List[dict]:
    if svd is None or movies_df is None or ratings_df is None:
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

    # Exclude everything this user has already rated
    rated_ids = set(int(x) for x in ratings_df[
        (ratings_df["userId"] == user_id) & (ratings_df["movieId"] != -1)
    ]["movieId"].tolist())

    all_ids = [int(x) for x in movies_df["movieId"].tolist()]
    unrated = [mid for mid in all_ids if mid not in rated_ids]

    # Filter by genres BEFORE predicting — movie must contain ALL selected genres
    if genres:
        genre_set = set(genres)
        unrated = [
            mid for mid in unrated
            if genre_set.issubset(movie_genres_lookup.get(mid, set()))
        ]

    preds = [svd.predict(int(user_id), int(mid)) for mid in unrated]
    top   = sorted(preds, key=lambda x: x.est, reverse=True)[:n]

    results = []
    for p in top:
        row = movies_df[movies_df["movieId"] == int(p.iid)]
        if not row.empty:
            r = row.iloc[0]
            results.append({
                "movieId":          int(p.iid),
                "title":            str(r["title"]),
                "genres":           str(r["genres"]),
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
    uid = generate_session_user_id()
    save_rating(uid, -1, -1)
    return {"user_id": uid, "is_new": True}

@app.get("/api/session/{user_id}")
def join_session(user_id: int):
    if not session_exists(user_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    real_ratings = ratings_df[
        (ratings_df["userId"] == user_id) & (ratings_df["movieId"] != -1)
    ]
    return {
        "user_id":      user_id,
        "is_new":       False,
        "rating_count": int(len(real_ratings)),
    }

@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int      = Query(...),
    n: int            = Query(10, ge=1, le=50),
    genres: str       = Query(None, description="Comma-separated genres e.g. Comedy,Drama"),
):
    genre_list = [g.strip() for g in genres.split(",")] if genres else None
    retrain_model()
    return {"user_id": user_id, "recommendations": get_top_n(user_id, n, genre_list)}

@app.get("/api/search", response_model=List[SearchResult])
def search_movies(q: str = Query(..., min_length=1)):
    if movies_df is None:
        raise HTTPException(status_code=503, detail="movies.csv not loaded")
    mask = movies_df["title"].str.contains(q, case=False, na=False)
    results = movies_df[mask].head(20)[["movieId", "title", "genres"]].copy()
    return [
        {"movieId": int(r["movieId"]), "title": str(r["title"]), "genres": str(r["genres"])}
        for _, r in results.iterrows()
    ]

@app.post("/api/rate", response_model=RateResponse)
def rate_movie(body: RateRequest):
    if not (0.5 <= body.rating <= 5.0):
        raise HTTPException(status_code=422, detail="Rating must be between 0.5 and 5.0")
    if not session_exists(body.user_id):
        raise HTTPException(status_code=404, detail="Session not found. Start a new session first.")
    save_rating(body.user_id, body.movie_id, body.rating)
    return {
        "user_id":  body.user_id,
        "movie_id": body.movie_id,
        "rating":   body.rating,
        "message":  "Rating saved.",
    }

@app.get("/api/my-ratings")
def get_my_ratings(user_id: int = Query(...)):
    if ratings_df is None:
        return {"user_id": user_id, "ratings": []}
    rows = ratings_df[
        (ratings_df["userId"] == user_id) & (ratings_df["movieId"] != -1)
    ]
    results = []
    if movies_df is not None:
        id_to_title = movies_df.set_index("movieId")["title"].to_dict()
        id_to_genre = movies_df.set_index("movieId")["genres"].to_dict()
        for _, r in rows.iterrows():
            mid = int(r["movieId"])
            results.append({
                "movieId": mid,
                "rating":  float(r["rating"]),
                "title":   str(id_to_title.get(mid, "Unknown")),
                "genres":  str(id_to_genre.get(mid, "")),
            })
    return {"user_id": user_id, "ratings": results}

@app.get("/health")
def health():
    total_users = int(ratings_df["userId"].nunique()) if ratings_df is not None else 0
    return {
        "status":         "ok",
        "model_loaded":   svd is not None,
        "movies_loaded":  movies_df is not None,
        "ratings_loaded": ratings_df is not None,
        "total_users":    total_users,
    }