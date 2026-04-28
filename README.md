# Movie Recommender

## Setup in 3 steps

### 1. Start up the venv
-if you would like start up the venv
-Bash
 source venv/Scripts/activate
-Windows
 venv\Scripts\activate

### 2. Install dependencies (one-time)

```bash
pip install fastapi uvicorn scikit-surprise pandas numpy
```

> If you get import errors run:
> ```bash
> pip install "numpy<2" scikit-surprise --force-reinstall
> ```

---

### 3. Run the server

```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Serves the frontend |
| GET | `/api/recommend?user_id=1&n=10` | Top-N recommendations for a user |
| GET | `/api/search?q=pulp` | Search movies by title |
| GET | `/api/users?limit=20` | List available user IDs |
| GET | `/health` | API + data status |

---

## Preloaded demo if no info provided
If `svd_model.pkl`, `movies.csv`, or `ratings.csv` are missing, the API still
runs and returns hard-coded demo recommendations so the UI is always usable.
# 325project
