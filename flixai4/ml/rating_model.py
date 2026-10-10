import json
import pickle
import sys
import time
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from recommender import load_data

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "ml" / "rating_model.pkl"
METRICS = ROOT / "ml" / "rating_metrics.json"
SEED = 42
N_ITER = 8
FOLDS = 4

# Step 1: the audience categories (ordered from youngest to oldest audience)
GROUPS = {
    "Kids": ["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"],
    "Family": ["TV-PG", "PG"],
    "Teens": ["TV-14", "PG-13"],
    "Adults": ["TV-MA", "R", "NC-17"],
}
ORDER = list(GROUPS)
RATING_TO_GROUP = {
    r: g for g, rs in GROUPS.items() for r in rs
}  # NR and UR (unrated) are left out

FEATURE_LABELS = {
    "type": "Movie or TV show",
    "country": "Country",
    "genres": "Genres",
    "n_genres": "Number of genres",
    "release_year": "Release year",
    "minutes": "Length in minutes",
    "seasons": "Number of seasons",
    "has_director": "Director listed",
}
FEATURE_DECISIONS = [
    {
        "name": "Genres",
        "used": True,
        "reason": "The strongest clue. Kids' TV and Children & Family Movies point to young audiences, Horror and Crime to adults.",
    },
    {
        "name": "Movie or TV show",
        "used": True,
        "reason": "Movies and series follow different rating systems (R and PG-13 versus TV-MA and TV-14).",
    },
    {
        "name": "Length (minutes or seasons)",
        "used": True,
        "reason": "Kept as two numbers: minutes for movies, seasons for shows.",
    },
    {
        "name": "Country (first listed)",
        "used": True,
        "reason": "Top 15 countries kept, the rest grouped as Other.",
    },
    {"name": "Release year", "used": True, "reason": "Rating habits change over time."},
    {
        "name": "Director listed (yes or no)",
        "used": True,
        "reason": "A small structural clue that differs between movies and shows.",
    },
    {
        "name": "Number of genres",
        "used": True,
        "reason": "Derived from the genre list.",
    },
    {
        "name": "Rating (TV-MA, PG and so on)",
        "used": False,
        "reason": "This is what we predict. Using it as an input would be cheating.",
    },
    {
        "name": "Unrated titles (NR, UR)",
        "used": False,
        "reason": "82 rows with no audience meaning, removed from training.",
    },
    {
        "name": "Title, Show ID, Director name",
        "used": False,
        "reason": "Unique text or identifiers that a model would memorise instead of learning from.",
    },
    {
        "name": "Date added",
        "used": False,
        "reason": "Not known for a new title and not related to who the title is made for.",
    },
]


def split_genres(text):
    return text.split("|") if text else []


def build_frame(df):
    primary = df["country"].apply(lambda v: v.split(",")[0].strip() or "Unknown")
    top = list(primary.value_counts().head(15).index)
    num = df["duration"].str.extract(r"(\d+)")[0].astype(float).fillna(0)
    movie = df["type"] == "Movie"
    genres = df["listed_in"].apply(
        lambda v: [g.strip() for g in v.split(",") if g.strip()]
    )
    X = pd.DataFrame(
        {
            "type": df["type"],
            "country": primary.where(primary.isin(top), "Other"),
            "genres": genres.apply(
                lambda gs: "|".join(g.replace(" ", "_") for g in gs)
            ),
            "n_genres": genres.apply(len),
            "release_year": df["release_year"],
            "minutes": num.where(movie, 0),
            "seasons": num.where(~movie, 0),
            "has_director": (df["director"] != "").astype(int),
        }
    )
    return X, sorted(set(top) | {"Other"})


def make_preprocessor():
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["type", "country"],
            ),
            (
                "genres",
                CountVectorizer(
                    tokenizer=split_genres,
                    token_pattern=None,
                    lowercase=False,
                    binary=True,
                ),
                "genres",
            ),
            (
                "num",
                StandardScaler(),
                ["n_genres", "release_year", "minutes", "seasons", "has_director"],
            ),
        ],
        sparse_threshold=0,
    )


# name -> (default model, hyperparameter search space)
MODELS = {
    "Logistic Regression": (
        lambda: LogisticRegression(max_iter=3000),
        {"model__C": [0.1, 0.3, 1, 3, 10], "model__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        lambda: DecisionTreeClassifier(random_state=SEED),
        {
            "model__max_depth": [4, 6, 8, 10, 14, None],
            "model__min_samples_leaf": [1, 3, 5, 10, 20],
            "model__criterion": ["gini", "entropy"],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "Random Forest": (
        lambda: RandomForestClassifier(random_state=SEED),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.3],
            "model__class_weight": [None, "balanced_subsample"],
        },
    ),
    "Gradient Boosting": (
        lambda: HistGradientBoostingClassifier(random_state=SEED),
        {
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_depth": [3, 5, None],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 1.0],
        },
    ),
}


def pipe(name):
    return Pipeline([("prep", make_preprocessor()), ("model", MODELS[name][0]())])


def clean(v):
    return v.item() if hasattr(v, "item") else v


def main():
    t0 = time.time()
    df = load_data()
    frame, countries = build_frame(df)
    group = df["rating"].map(RATING_TO_GROUP)
    keep = group.notna().to_numpy()
    X, y = frame[keep], group[keep]

    # Step 1: rating categories
    counts = df["rating"].value_counts()
    categories = [
        {
            "name": g,
            "count": int(counts.reindex(rs).fillna(0).sum()),
            "ratings": [{"rating": r, "count": int(counts.get(r, 0))} for r in rs],
        }
        for g, rs in GROUPS.items()
    ]
    majority = float(y.value_counts(normalize=True).max())
    print(
        f"Rows used {len(y)}, unrated removed {len(df) - len(y)}, majority class {majority:.3f}",
        flush=True,
    )

    # Step 2: split once. The test part is untouched until the very end.
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
    )
    Xtr, ytr, Xte, yte = (
        X.iloc[idx_train],
        y.iloc[idx_train],
        X.iloc[idx_test],
        y.iloc[idx_test],
    )
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    scoring = {"acc": "accuracy", "f1": "f1_macro"}

    # Steps 3 and 4: default models, then tuned models
    rows, tuned = [], {}
    for name in MODELS:
        d = cross_validate(pipe(name), Xtr, ytr, cv=cv, scoring=scoring)
        default_fit = pipe(name).fit(Xtr, ytr)
        dp = default_fit.predict(Xte)
        space = MODELS[name][1]
        search = RandomizedSearchCV(
            pipe(name),
            space,
            n_iter=min(N_ITER, int(np.prod([len(v) for v in space.values()]))),
            cv=cv,
            scoring=scoring,
            refit="f1",
            random_state=SEED,
            n_jobs=1,
        ).fit(Xtr, ytr)
        i, tp = search.best_index_, search.best_estimator_.predict(Xte)
        tuned[name] = search.best_estimator_
        rows.append(
            {
                "model": name,
                "default": {
                    "cv_accuracy": round(d["test_acc"].mean(), 4),
                    "cv_f1": round(d["test_f1"].mean(), 4),
                    "test_accuracy": round(accuracy_score(yte, dp), 4),
                    "test_f1": round(f1_score(yte, dp, average="macro"), 4),
                },
                "tuned": {
                    "cv_accuracy": round(search.cv_results_["mean_test_acc"][i], 4),
                    "cv_f1": round(search.cv_results_["mean_test_f1"][i], 4),
                    "test_accuracy": round(accuracy_score(yte, tp), 4),
                    "test_f1": round(f1_score(yte, tp, average="macro"), 4),
                    "params": {
                        k.replace("model__", ""): clean(v)
                        for k, v in search.best_params_.items()
                    },
                },
                "search_space": {
                    k.replace("model__", ""): [
                        clean(x) if x is not None else "None" for x in v
                    ]
                    for k, v in space.items()
                },
            }
        )
        r = rows[-1]
        print(
            f"{name:20s} default cv_f1={r['default']['cv_f1']:.3f}  tuned cv_f1={r['tuned']['cv_f1']:.3f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

    # Hyperparameter curve for the decision tree: shallow trees underfit, deep trees memorise
    depths = list(range(1, 21))
    tr, va = validation_curve(
        pipe("Decision Tree"),
        Xtr,
        ytr,
        param_name="model__max_depth",
        param_range=depths,
        cv=cv,
        scoring="accuracy",
    )
    curve = {
        "depths": depths,
        "train": [round(v, 4) for v in tr.mean(axis=1)],
        "cv": [round(v, 4) for v in va.mean(axis=1)],
        "best_depth": depths[int(va.mean(axis=1).argmax())],
    }

    # Step 5: the winner (best cross-validated macro F1) is tested once on the hold-out
    best = max(rows, key=lambda r: r["tuned"]["cv_f1"])["model"]
    model = tuned[best]
    pred, proba = model.predict(Xte), model.predict_proba(Xte)
    p, r_, f, s = precision_recall_fscore_support(
        yte, pred, labels=ORDER, zero_division=0
    )
    imp = permutation_importance(
        model, Xte, yte, scoring="f1_macro", n_repeats=5, random_state=SEED
    )
    importance = sorted(
        (
            {"feature": FEATURE_LABELS[c], "value": round(float(v), 4)}
            for c, v in zip(Xte.columns, imp.importances_mean)
        ),
        key=lambda d: -d["value"],
    )
    final = {
        "model": best,
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(f1_score(yte, pred, average="macro"), 4),
        "weighted_f1": round(f1_score(yte, pred, average="weighted"), 4),
        "labels": ORDER,
        "classes": [
            {
                "name": g,
                "precision": round(float(p[k]), 4),
                "recall": round(float(r_[k]), 4),
                "f1": round(float(f[k]), 4),
                "support": int(s[k]),
            }
            for k, g in enumerate(ORDER)
        ],
        "confusion": confusion_matrix(yte, pred, labels=ORDER).tolist(),
        "importance": importance,
    }
    metrics = {
        "rows": int(len(y)),
        "unrated_removed": int(len(df) - len(y)),
        "baseline_accuracy": round(majority, 4),
        "folds": FOLDS,
        "n_iter": N_ITER,
        "categories": categories,
        "models": rows,
        "curve": curve,
        "final": final,
        "features": FEATURE_DECISIONS,
    }
    METRICS.write_text(json.dumps(metrics, indent=2))

    real_idx = np.flatnonzero(keep)  # position in the full dataframe
    minutes = frame.loc[df["type"] == "Movie", "minutes"]
    options = {
        "types": ["Movie", "TV Show"],
        "countries": countries,
        "years": [int(df["release_year"].min()), int(df["release_year"].max())],
        "genres": sorted(
            {g.replace("_", " ") for v in frame["genres"] for g in v.split("|") if g}
        ),
        "minutes": [int(minutes[minutes > 0].min()), int(minutes.max())],
        "seasons": [1, int(frame["seasons"].max())],
    }
    with open(ARTIFACT, "wb") as fh:
        pickle.dump(
            {
                "model": model,
                "name": best,
                "options": options,
                "frame": frame,
                "titles": df["title"],
                "years": df["release_year"],
                "ratings": df["rating"],
                "groups": group,
                "test_idx": real_idx[idx_test].tolist(),
            },
            fh,
        )
    print(
        f"\nBest: {best}  test accuracy={final['accuracy']}  macro F1={final['macro_f1']}  [{time.time() - t0:.0f}s]"
    )


def predict(bundle, kind, genres, country, year, minutes, seasons, has_director):
    """Probability of each audience category for values typed in by a user."""
    names = sorted(set(genres))
    row = pd.DataFrame(
        [
            {
                "type": kind,
                "country": (
                    country if country in bundle["options"]["countries"] else "Other"
                ),
                "genres": "|".join(g.replace(" ", "_") for g in names),
                "n_genres": len(names),
                "release_year": year,
                "minutes": minutes if kind == "Movie" else 0,
                "seasons": seasons if kind == "TV Show" else 0,
                "has_director": int(has_director),
            }
        ]
    )
    return _probs(bundle, row)


def predict_row(bundle, i):
    return _probs(bundle, bundle["frame"].iloc[[i]])


def _probs(bundle, row):
    m = bundle["model"]
    p = dict(zip(m.classes_, m.predict_proba(row)[0]))
    return {g: float(p.get(g, 0.0)) for g in ORDER}


if __name__ == "__main__":
    import rating_model  

    rating_model.main()
