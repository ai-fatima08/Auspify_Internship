import json
import os
import pickle
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ml"))
from recommender import recommend, recommend_profile
import type_model

# Load the trained model that recommender.py saved
with open(ROOT / "ml" / "model.pkl", "rb") as f:
    bundle = pickle.load(f)
MATRIX, DF = bundle["matrix"], bundle["df"]
METRICS = json.loads((ROOT / "ml" / "metrics.json").read_text())
LOWER = DF["title"].str.lower()

TMDB_KEY = os.environ.get("TMDB_API_KEY", "")
POSTERS = {}

TYPE = None
if (ROOT / "ml" / "type_model.pkl").exists() and (
    ROOT / "ml" / "type_metrics.json"
).exists():
    with open(ROOT / "ml" / "type_model.pkl", "rb") as f:
        TYPE = pickle.load(f)
    TYPE_METRICS = json.loads((ROOT / "ml" / "type_metrics.json").read_text())
    TEST_SET = set(TYPE["test_idx"])
    TEST_LOWER = {i: TYPE["titles"].iloc[i].lower() for i in TYPE["test_idx"]}

app = Flask(__name__, static_folder="static", static_url_path="")


def card(i, score=None):
    """Turn one dataset row into a small dictionary the website can show."""
    r = DF.iloc[i]
    out = {
        "id": int(i),
        "title": r["title"],
        "type": r["type"],
        "year": int(r["release_year"]),
        "rating": r["rating"],
        "duration": r["duration"],
        "country": r["country"],
        "director": r["director"],
        "genres": [g.strip() for g in r["listed_in"].split(",")],
    }
    if score is not None:
        out["score"] = round(score, 4)
    return out


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/search")
def search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    starts = DF.index[LOWER.str.startswith(q)].tolist()
    taken = set(starts)
    contains = [
        i for i in DF.index[LOWER.str.contains(q, regex=False)] if i not in taken
    ]
    return jsonify([card(i) for i in (starts + contains)[:8]])


@app.get("/api/recommend/<int:idx>")
def recs(idx):
    if not 0 <= idx < len(DF):
        return jsonify({"error": "Title not found"}), 404
    same_type = request.args.get("same_type") == "true"
    found = recommend(MATRIX, DF, idx, 12, same_type)
    return jsonify(
        {"title": card(idx), "recommendations": [card(i, s) for i, s in found]}
    )


@app.get("/api/taste")
def taste():
    """Recommendations for a mix of titles: /api/taste?ids=12,87,400"""
    try:
        ids = [int(x) for x in request.args.get("ids", "").split(",") if x][:5]
    except ValueError:
        return jsonify({"error": "Invalid ids"}), 400
    if not ids or any(not 0 <= i < len(DF) for i in ids):
        return jsonify({"error": "Choose at least one title"}), 400
    found = recommend_profile(MATRIX, DF, ids, 12)
    return jsonify(
        {
            "titles": [card(i) for i in ids],
            "recommendations": [card(i, s) for i, s in found],
        }
    )


@app.get("/api/popular")
def popular():
    n = min(int(request.args.get("n", 14)), 30)
    movies = round(n * 0.6)
    recent = DF[DF["release_year"] >= 2015]
    picks = list(recent[recent["type"] == "Movie"].sample(movies).index) + list(
        recent[recent["type"] == "TV Show"].sample(n - movies).index
    )
    return jsonify([card(i) for i in picks])


@app.get("/api/config")
def config():
    return jsonify({"posters": bool(TMDB_KEY)})


@app.get("/api/poster/<int:idx>")
def poster(idx):
    """Poster image URL from TMDB, or null when no key is set or nothing is found."""
    if not TMDB_KEY or not 0 <= idx < len(DF):
        return jsonify({"url": None})
    if idx in POSTERS:
        return jsonify({"url": POSTERS[idx]})
    row = DF.iloc[idx]
    kind, year_key = (
        ("movie", "year") if row["type"] == "Movie" else ("tv", "first_air_date_year")
    )
    query = urllib.parse.urlencode(
        {"api_key": TMDB_KEY, "query": row["title"], year_key: int(row["release_year"])}
    )
    try:
        with urllib.request.urlopen(
            f"https://api.themoviedb.org/3/search/{kind}?{query}", timeout=5
        ) as resp:
            results = json.load(resp).get("results", [])
    except Exception:
        return jsonify({"url": None})  # do not cache failures
    path = next((r["poster_path"] for r in results if r.get("poster_path")), None)
    POSTERS[idx] = f"https://image.tmdb.org/t/p/w342{path}" if path else None
    return jsonify({"url": POSTERS[idx]})


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


def type_ready():
    return TYPE is not None


NOT_READY = ({"error": "Task 2 model not found. Run: python ml/type_model.py"}, 503)


@app.get("/api/type/metrics")
def type_metrics():
    return NOT_READY if not type_ready() else jsonify(TYPE_METRICS)


@app.get("/api/type/options")
def type_options():
    return NOT_READY if not type_ready() else jsonify(TYPE["options"])


@app.get("/api/type/predict")
def type_predict():
    if not type_ready():
        return NOT_READY
    opts = TYPE["options"]
    mode = request.args.get("mode", "honest")
    if mode not in TYPE["models"]:
        return jsonify({"error": "Unknown mode"}), 400
    try:
        year = int(request.args.get("year", 2019))
    except ValueError:
        return jsonify({"error": "Invalid year"}), 400
    year = max(opts["years"][0], min(opts["years"][1], year))
    genres = [
        g for g in request.args.get("genres", "").split(",") if g in opts["genres"]
    ]
    rating = request.args.get("rating", "TV-MA")
    country = request.args.get("country", "Other")
    p = type_model.predict(
        TYPE, mode, genres, country, rating, year, request.args.get("director") == "1"
    )
    return jsonify(
        {
            "show": round(p, 4),
            "movie": round(1 - p, 4),
            "label": "TV Show" if p >= 0.5 else "Movie",
            "model": TYPE["models"][mode]["model"],
            "mode": mode,
        }
    )


def held_out_card(i):
    """A title from the 20% the models never saw, with both models' predictions."""
    row = TYPE["frame"].iloc[i]
    preds = {}
    for mode in TYPE["models"]:
        p = type_model.predict_row(TYPE, mode, i)
        preds[mode] = {"show": round(p, 4), "label": "TV Show" if p >= 0.5 else "Movie"}
    return {
        "id": int(i),
        "title": TYPE["titles"].iloc[i],
        "year": int(TYPE["years"].iloc[i]),
        "actual": TYPE["types"].iloc[i],
        "features": {
            "genres": [g.replace("_", " ") for g in row["genres"].split("|") if g],
            "rating": row["rating"],
            "country": row["country"],
            "director": bool(row["has_director"]),
        },
        "predictions": preds,
    }


@app.get("/api/type/titles")
def type_titles():
    q = request.args.get("q", "").strip().lower()
    if not type_ready() or len(q) < 2:
        return jsonify([])
    hits = [i for i, t in TEST_LOWER.items() if q in t][:8]
    return jsonify(
        [
            {
                "id": i,
                "title": TYPE["titles"].iloc[i],
                "year": int(TYPE["years"].iloc[i]),
            }
            for i in hits
        ]
    )


@app.get("/api/type/title/<int:i>")
def type_title(i):
    if not type_ready():
        return NOT_READY
    if i not in TEST_SET:
        return jsonify({"error": "Not a held-out title"}), 404
    return jsonify(held_out_card(i))


@app.get("/api/type/random")
def type_random():
    return (
        NOT_READY
        if not type_ready()
        else jsonify(held_out_card(random.choice(TYPE["test_idx"])))
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
