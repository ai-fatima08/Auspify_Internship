import json
import os
import pickle
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ml"))
from recommender import recommend, recommend_profile  

# Load the trained model that recommender.py saved
with open(ROOT / "ml" / "model.pkl", "rb") as f:
    bundle = pickle.load(f)
MATRIX, DF = bundle["matrix"], bundle["df"]
METRICS = json.loads((ROOT / "ml" / "metrics.json").read_text())
LOWER = DF["title"].str.lower()

TMDB_KEY = os.environ.get("TMDB_API_KEY", "")
POSTERS = {}

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


if __name__ == "__main__":
    app.run(port=5000, debug=True)

