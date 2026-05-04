# Movie Recommender

## Original Model code:

https://colab.research.google.com/drive/1NSnuXZKQCPMuDJXADh9uzkIPPq4nm4Kz?usp=sharing


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
> pip install uvicorn    may need to be ran
> ```

---

### 3. Run the server

uvicorn main:app --reload

Open **http://localhost:8000** in your browser.

### 4. Use the model

choose a session and add ratings, then get recommendations, which will retrain the model, then filter by the movies you want to see

## Preloaded demo if no info provided
If `svd_model.pkl`, `movies.csv`, or `ratings.csv` are missing, the API still
runs and returns hard-coded demo recommendations so the UI is always usable.
# 325project
