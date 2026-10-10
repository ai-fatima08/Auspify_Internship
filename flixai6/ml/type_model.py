import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from recommender import load_data

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "ml" / "type_model.pkl"
METRICS = ROOT / "ml" / "type_metrics.json"
SEED = 42

SCENARIOS = {
    "all": (
        "All features",
        "Includes duration, which reads '90 min' for movies and '2 Seasons' for shows. The model simply reads the answer.",
    ),
    "no_dur": (
        "Without duration",
        "Duration removed, but genres such as 'TV Dramas' still reveal the type.",
    ),
    "honest": (
        "Realistic",
        "Duration removed and words like TV, Shows, Movies stripped from genres. Genre names such as Docuseries or Kids' still only exist for shows, so scores stay very high.",
    ),
    "strict": (
        "Strict",
        "Duration and genres both removed. Only rating, country, release year and whether a director is listed remain. This is the signal that is left when the obvious clues are gone.",
    ),
}
FEATURE_LABELS = {
    "rating": "Rating",
    "country": "Country",
    "genres": "Genres",
    "release_year": "Release year",
    "has_director": "Director listed",
    "n_genres": "Number of genres",
    "duration_value": "Duration number",
    "duration_is_min": "Duration unit",
}

# Step 1 of the task: which columns are used, and why the others are left out
FEATURE_DECISIONS = [
    {
        "name": "Genres",
        "used": True,
        "reason": "Strongest signal. In the realistic model the words TV, Shows and Movies are removed from the names.",
    },
    {
        "name": "Director listed (yes or no)",
        "used": True,
        "reason": "Most TV shows have no director credited, while most movies do.",
    },
    {
        "name": "Rating",
        "used": True,
        "reason": "Small but real difference between movie and show ratings.",
    },
    {
        "name": "Country (first listed)",
        "used": True,
        "reason": "Top 15 countries kept, the rest grouped as Other.",
    },
    {
        "name": "Release year",
        "used": True,
        "reason": "Cheap to collect and easy for a user to provide.",
    },
    {
        "name": "Number of genres",
        "used": True,
        "reason": "Derived from the genres. Not used in the strict model.",
    },
    {
        "name": "Duration",
        "used": False,
        "reason": "Only tested in the All features scenario. It reads minutes for movies and seasons for shows, so it gives the answer away.",
    },
    {
        "name": "Title",
        "used": False,
        "reason": "Unique text on every row. A model would memorise it instead of learning a pattern.",
    },
    {"name": "Show ID", "used": False, "reason": "An identifier with no meaning."},
    {
        "name": "Director name",
        "used": False,
        "reason": "Thousands of names that appear once. Only whether a director exists is kept.",
    },
    {
        "name": "Date added",
        "used": False,
        "reason": "Not known for a new title, and it showed no benefit in testing.",
    },
]


def split_genres(text):
    return text.split("|") if text else []


def neutral_genre(name):
    """'Crime TV Shows' -> 'Crime', 'International Movies' -> 'International'."""
    out = re.sub(r"\b(TV|Shows?|Movies?|Series)\b", "", name)
    out = re.sub(r"\s+", " ", out).strip(" ,&")
    return out or "General"


def build_frame(df, neutral):
    genres = df["listed_in"].apply(
        lambda v: [g.strip() for g in v.split(",") if g.strip()]
    )
    if neutral:
        genres = genres.apply(lambda gs: sorted({neutral_genre(g) for g in gs}))
    primary = df["country"].apply(lambda v: (v.split(",")[0].strip() or "Unknown"))
    top = list(primary.value_counts().head(15).index)
    duration = df["duration"].str.extract(r"(\d+)")[0].astype(float)
    X = pd.DataFrame(
        {
            "rating": df["rating"],
            "country": primary.where(primary.isin(top), "Other"),
            "genres": genres.apply(
                lambda gs: "|".join(g.replace(" ", "_") for g in gs)
            ),
            "n_genres": genres.apply(len),
            "release_year": df["release_year"],
            "has_director": (df["director"] != "").astype(int),
            "duration_value": duration,
            "duration_is_min": df["duration"].str.contains("min").astype(int),
        }
    )
    return X, sorted(set(top) | {"Other"})


def make_preprocessor(cols):
    cat = [c for c in ("rating", "country") if c in cols]
    num = [
        c
        for c in (
            "release_year",
            "has_director",
            "n_genres",
            "duration_value",
            "duration_is_min",
        )
        if c in cols
    ]
    parts = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
    if "genres" in cols:
        parts.append(
            (
                "genres",
                CountVectorizer(
                    tokenizer=split_genres,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                ),
                "genres",
            )
        )
    parts.append(("num", StandardScaler(), num))
    return ColumnTransformer(parts, sparse_threshold=0)


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=SEED
    ),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}


def pipeline(name, cols):
    return Pipeline([("prep", make_preprocessor(cols)), ("model", MODELS[name]())])


def main():
    df = load_data()
    y = (df["type"] == "TV Show").astype(int)  # 1 = TV Show, 0 = Movie
    raw, _ = build_frame(df, neutral=False)
    fair, countries = build_frame(df, neutral=True)
    no_dur = [c for c in raw.columns if not c.startswith("duration")]
    strict = [c for c in no_dur if c not in ("genres", "n_genres")]
    setups = {
        "all": (raw, list(raw.columns)),
        "no_dur": (raw, no_dur),
        "honest": (fair, no_dur),
        "strict": (fair, strict),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    scenarios = []
    for key, (label, note) in SCENARIOS.items():
        X, cols = setups[key]
        rows = []
        for name in MODELS:
            r = cross_validate(
                pipeline(name, cols),
                X[cols],
                y,
                cv=cv,
                scoring=["accuracy", "f1", "roc_auc"],
            )
            rows.append(
                {
                    "model": name,
                    "accuracy": round(r["test_accuracy"].mean(), 4),
                    "accuracy_std": round(r["test_accuracy"].std(), 4),
                    "f1": round(r["test_f1"].mean(), 4),
                    "auc": round(r["test_roc_auc"].mean(), 4),
                }
            )
            print(
                f"{key:7s} {name:20s} acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['f1']:.4f} auc={rows[-1]['auc']:.4f}"
            )
        scenarios.append({"key": key, "label": label, "note": note, "results": rows})

    # Final models (realistic and strict): best classifier of each, checked on a held-out 20% it has never seen
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED
    )
    final, models = {}, {}
    for key in ("honest", "strict"):
        X, cols = setups[key]
        results = next(sc for sc in scenarios if sc["key"] == key)["results"]
        best_name = max(results, key=lambda r: r["f1"])["model"]
        fitted = pipeline(best_name, cols).fit(
            X.iloc[idx_train][cols], y.iloc[idx_train]
        )
        Xt, yt = X.iloc[idx_test][cols], y.iloc[idx_test]
        pred, proba = fitted.predict(Xt), fitted.predict_proba(Xt)[:, 1]
        tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
        imp = permutation_importance(
            fitted, Xt, yt, scoring="f1", n_repeats=5, random_state=SEED
        )
        importance = sorted(
            (
                {"feature": FEATURE_LABELS[c], "value": round(float(v), 4)}
                for c, v in zip(cols, imp.importances_mean)
            ),
            key=lambda d: -d["value"],
        )
        final[key] = {
            "model": best_name,
            "train_rows": int(len(idx_train)),
            "test_rows": int(len(idx_test)),
            "accuracy": round(accuracy_score(yt, pred), 4),
            "precision": round(precision_score(yt, pred), 4),
            "recall": round(recall_score(yt, pred), 4),
            "f1": round(f1_score(yt, pred), 4),
            "auc": round(roc_auc_score(yt, proba), 4),
            "confusion": {
                "movie_correct": int(tn),
                "movie_as_show": int(fp),
                "show_as_movie": int(fn),
                "show_correct": int(tp),
            },
            "importance": importance,
        }
        models[key] = {"pipeline": fitted, "cols": cols, "model": best_name}
        print(
            f"Final {key:7s} {best_name:20s} test accuracy={final[key]['accuracy']} f1={final[key]['f1']}"
        )

    metrics = {
        "rows": int(len(df)),
        "movies": int((y == 0).sum()),
        "shows": int((y == 1).sum()),
        "baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "folds": 5,
        "scenarios": scenarios,
        "final": final,
        "features": FEATURE_DECISIONS,
    }
    METRICS.write_text(json.dumps(metrics, indent=2))
    options = {
        "ratings": sorted(df["rating"].unique()),
        "countries": countries,
        "genres": sorted(
            {g.replace("_", " ") for v in fair["genres"] for g in v.split("|") if g}
            - {"General"}
        ),
        "years": [int(df["release_year"].min()), int(df["release_year"].max())],
    }
    with open(ARTIFACT, "wb") as f:
        pickle.dump(
            {
                "models": models,
                "options": options,
                "frame": fair,
                "titles": df["title"],
                "types": df["type"],
                "years": df["release_year"],
                "test_idx": idx_test.tolist(),
            },
            f,
        )
    print("Saved", ARTIFACT.name, "and", METRICS.name)


def predict(bundle, mode, genres, country, rating, year, has_director):
    """Probability that a title is a TV show, for values typed in by a user."""
    m = bundle["models"][mode]
    names = sorted(set(genres))
    row = pd.DataFrame(
        [
            {
                "rating": rating,
                "country": (
                    country if country in bundle["options"]["countries"] else "Other"
                ),
                "genres": "|".join(g.replace(" ", "_") for g in names),
                "n_genres": len(names),
                "release_year": year,
                "has_director": int(has_director),
                "duration_value": np.nan,
                "duration_is_min": 0,
            }
        ]
    )
    return float(m["pipeline"].predict_proba(row[m["cols"]])[0, 1])


def predict_row(bundle, mode, i):
    """Probability that row i of the dataset is a TV show (use rows from the held-out test set)."""
    m = bundle["models"][mode]
    return float(
        m["pipeline"].predict_proba(bundle["frame"].iloc[[i]][m["cols"]])[0, 1]
    )


if __name__ == "__main__":
    import type_model 

    type_model.main()
