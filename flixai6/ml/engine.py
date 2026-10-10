import argparse
import datetime
import hashlib
import json
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from recommender import load_data

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Dataset.xlsx"
REPORT = ROOT / "ml" / "engine_report.json"
REGISTRY = ROOT / "ml" / "registry.json"
RUNLOG = ROOT / "ml" / "runs.jsonl"
MODELDIR = ROOT / "ml" / "models"
SEED = 42
N_ITER, FOLDS = 8, 5
PROMOTION_TOLERANCE = 0.005  
AUDIENCE = {
    "TV-Y": "Kids",
    "TV-Y7": "Kids",
    "TV-Y7-FV": "Kids",
    "TV-G": "Kids",
    "G": "Kids",
    "TV-PG": "Family",
    "PG": "Family",
    "TV-14": "Teens",
    "PG-13": "Teens",
    "TV-MA": "Adults",
    "R": "Adults",
    "NC-17": "Adults",
}
TOP_COUNTRIES = 12


def split_genres(text):
    return text.split("|") if text else []


#  feature engineering (shared by training and live scoring)
def catalogue_stats(df):
    """Facts about the whole catalogue (no labels involved), reused when scoring a new concept."""
    shows = df[df["type"] == "TV Show"]
    genres = shows["listed_in"].str.split(",").explode().str.strip()
    primary = shows["country"].str.split(",").str[0].str.strip().replace("", "Unknown")
    people = df["director"].str.split(",").explode().str.strip()
    return {
        "genre_freq": (genres.value_counts() / len(shows)).to_dict(),
        "countries": list(primary.value_counts().head(TOP_COUNTRIES).index),
        "director_counts": people[people != ""].value_counts().to_dict(),
    }


def concepts_from_shows(shows, stats):
    added = pd.to_datetime(shows["date_added"])
    out = []
    for (_, r), a in zip(shows.iterrows(), added):
        names = [d.strip() for d in r["director"].split(",") if d.strip()]
        out.append(
            {
                "genres": [g.strip() for g in r["listed_in"].split(",") if g.strip()],
                "country": (r["country"].split(",")[0].strip() or "Unknown"),
                "n_countries": max(
                    1, len([c for c in r["country"].split(",") if c.strip()])
                ),
                "rating": r["rating"],
                "release_year": int(r["release_year"]),
                "added_year": int(a.year),
                "added_month": int(a.month),
                "has_director": int(bool(names)),
                "director_titles": max(
                    [stats["director_counts"].get(n, 1) - 1 for n in names], default=0
                ),
            }
        )
    return out


def concept_frame(concepts, stats):
    """Turn raw concept dictionaries into model features. Training and the live scorer both call this one function."""
    rows = []
    for c in concepts:
        gs = sorted(set(c["genres"]))
        gap = int(np.clip(c["added_year"] - c["release_year"], 0, 8))
        rows.append(
            {
                "genres": "|".join(g.replace(" ", "_") for g in gs),
                "genre_combo": "|".join(g.replace(" ", "_") for g in gs) or "none",
                "n_genres": len(gs),
                "genre_rarity": (
                    float(
                        np.mean([-np.log(stats["genre_freq"].get(g, 1e-3)) for g in gs])
                    )
                    if gs
                    else 0.0
                ),
                "country": (
                    c["country"] if c["country"] in stats["countries"] else "Other"
                ),
                "n_countries": c.get("n_countries", 1),
                "has_director": int(c.get("has_director", 0)),
                "director_titles": c.get("director_titles", 0),
                "rating": c["rating"],
                "audience": AUDIENCE.get(c["rating"], "Unrated"),
                "release_year": c["release_year"],
                "added_year": c["added_year"],
                "added_month": c.get("added_month", 6),
                "years_to_add": gap,
                "fresh_add": int(gap <= 1),
            }
        )
    return pd.DataFrame(rows)


FEATURE_GROUPS = {
    "Genre profile": ["genres", "genre_combo", "n_genres", "genre_rarity"],
    "Country": ["country", "n_countries"],
    "Audience rating": ["rating", "audience"],
    "Timing": [
        "release_year",
        "added_year",
        "added_month",
        "years_to_add",
        "fresh_add",
    ],
    "Creator": ["has_director", "director_titles"],
}
FEATURE_DOC = [
    (
        "Genre multi-hot",
        "Genre profile",
        "One 0/1 column per genre, so combinations of genres are visible to the model.",
    ),
    (
        "Genre combination (target encoded)",
        "Genre profile",
        "The exact set of genres, turned into the renewal rate seen for that combination. Computed out-of-fold so a row never sees its own label.",
    ),
    (
        "Genre rarity",
        "Genre profile",
        "How unusual the genres are among TV series (average of minus log frequency). Separates niche from mainstream.",
    ),
    ("Number of genres", "Genre profile", "Narrow or broad positioning."),
    (
        "Primary country (target encoded)",
        "Country",
        "Top 12 countries plus Other, encoded with out-of-fold renewal rates.",
    ),
    (
        "Number of countries",
        "Country",
        "Co-productions versus single-country productions.",
    ),
    (
        "Audience group and rating",
        "Audience rating",
        "Kids, Family, Teens, Adults from the 14 raw ratings, plus the raw rating.",
    ),
    (
        "Release year, year and month added",
        "Timing",
        "When the series was made and when it reached Netflix.",
    ),
    (
        "Years to add and fresh-add flag",
        "Timing",
        "Gap between release and arrival on Netflix. A gap of 0 or 1 year points to a fresh or original-style release.",
    ),
    ("Director credited", "Creator", "Whether any director is listed."),
    (
        "Director experience",
        "Creator",
        "How many other titles the credited director has in the catalogue.",
    ),
]


def make_preprocessor():
    return ColumnTransformer(
        [
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
                "te",
                TargetEncoder(smooth="auto", target_type="binary", random_state=SEED),
                ["country", "genre_combo"],
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=10, sparse_output=False
                ),
                ["rating", "audience"],
            ),
            (
                "num",
                StandardScaler(),
                [
                    "n_genres",
                    "genre_rarity",
                    "n_countries",
                    "has_director",
                    "director_titles",
                    "release_year",
                    "added_year",
                    "added_month",
                    "years_to_add",
                    "fresh_add",
                ],
            ),
        ],
        sparse_threshold=0,
    )


MODELS = {
    "Logistic Regression": (
        "Linear",
        lambda: LogisticRegression(max_iter=3000),
        {"model__C": [0.03, 0.1, 0.3, 1, 3], "model__class_weight": [None, "balanced"]},
    ),
    "Random Forest": (
        "Tree ensemble",
        lambda: RandomForestClassifier(n_estimators=300, random_state=SEED),
        {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [2, 5, 10, 20],
            "model__max_features": ["sqrt", 0.3, 0.5],
        },
    ),
    "Gradient Boosting": (
        "Boosting",
        lambda: HistGradientBoostingClassifier(random_state=SEED),
        {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, None],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.0, 1.0, 5.0],
            "model__min_samples_leaf": [10, 20, 40],
        },
    ),
}


def pipe(factory):
    return Pipeline([("prep", make_preprocessor()), ("model", factory())])


class AvgEnsemble(ClassifierMixin, BaseEstimator):
    """Averages the probabilities of several fitted-from-scratch models."""

    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X, y):
        self.models_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# small helpers
class Log:
    def __init__(self):
        self.items = []

    def stage(self, name):
        log = self

        class Ctx:
            detail = ""

            def __enter__(s):
                s.t = time.time()
                return s

            def __exit__(s, et, ev, tb):
                log.items.append(
                    {
                        "name": name,
                        "seconds": round(time.time() - s.t, 1),
                        "status": "failed" if et else "ok",
                        "detail": s.detail,
                    }
                )
                print(
                    f"[{log.items[-1]['status']}] {name} ({log.items[-1]['seconds']}s) {s.detail}",
                    flush=True,
                )

        return Ctx()


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ap": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
    }


def thin(x, y, n=60):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def psi(ref, cur, bins):
    r = np.array([(ref == b).mean() for b in bins]) + 1e-4
    c = np.array([(cur == b).mean() for b in bins]) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def short(name):
    return "ensemble" if name.startswith("Ensemble") else f"{name} model"


def phrase(name):
    return (
        "an ensemble of three models"
        if name.startswith("Ensemble")
        else f"a {name} model"
    )


def conf_label(n, lo, hi):
    hw = (hi - lo) / 2
    return (
        "High"
        if n >= 300 and hw <= 0.06
        else "Medium" if n >= 80 and hw <= 0.12 else "Low"
    )


#  validation (MLOps: never train on data that failed checks)
def validate(raw, df, shows):
    checks = []
    need = {
        "show_id",
        "type",
        "title",
        "director",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
    }
    miss = need - set(raw.columns)
    checks.append(
        {
            "name": "Required columns present",
            "status": "fail" if miss else "pass",
            "detail": (
                f"missing: {sorted(miss)}" if miss else f"all {len(need)} columns found"
            ),
        }
    )
    checks.append(
        {
            "name": "Unique show_id",
            "status": "pass" if raw["show_id"].is_unique else "fail",
            "detail": f"{len(raw):,} rows, {raw['show_id'].duplicated().sum()} duplicate ids",
        }
    )
    dup = len(raw) - len(df)
    checks.append(
        {
            "name": "Duplicate titles",
            "status": "pass" if dup == 0 else "warn",
            "detail": f"{dup} repeated title, type and year rows removed",
        }
    )
    bad_dates = int(pd.to_datetime(raw["date_added"], errors="coerce").isna().sum())
    checks.append(
        {
            "name": "Dates parse",
            "status": "pass" if bad_dates == 0 else "fail",
            "detail": f"{bad_dates} unreadable date_added values",
        }
    )
    ry = raw["release_year"]
    checks.append(
        {
            "name": "Release years plausible",
            "status": "pass" if ry.between(1920, 2022).all() else "warn",
            "detail": f"range {int(ry.min())} to {int(ry.max())}",
        }
    )
    for col in ("director", "country"):
        share = float((df[col] == "").mean())
        checks.append(
            {
                "name": f"Missing {col}",
                "status": "pass" if share < 0.15 else "warn",
                "detail": f"{share * 100:.0f}% of all titles have no {col}",
            }
        )
    seasons = shows["duration"].str.extract(r"(\d+)")[0]
    checks.append(
        {
            "name": "Season counts readable",
            "status": "pass" if seasons.notna().all() else "fail",
            "detail": f"{int(seasons.isna().sum())} series without a season count",
        }
    )
    rate = float((seasons.astype(int) >= 2).mean())
    checks.append(
        {
            "name": "Target balance",
            "status": "pass" if 0.1 < rate < 0.9 else "warn",
            "detail": f"{rate * 100:.1f}% of series have 2+ seasons",
        }
    )
    checks.append(
        {
            "name": "Enough series to model",
            "status": "pass" if len(shows) >= 1000 else "warn",
            "detail": f"{len(shows):,} TV series",
        }
    )
    return checks


#  main pipeline
def main(validate_only=False):
    t_start = time.time()
    log = Log()
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with log.stage("1. Load and validate data") as s:
        raw = pd.read_excel(DATA)
        df = load_data()
        shows = df[df["type"] == "TV Show"].reset_index(drop=True)
        checks = validate(raw, df, shows)
        failed = [c for c in checks if c["status"] == "fail"]
        s.detail = f"{len(checks)} checks, {len(failed)} failed, {sum(c['status'] == 'warn' for c in checks)} warnings"
        if failed:
            print("Data validation failed:", failed)
            sys.exit(1)
    if validate_only:
        print(json.dumps(checks, indent=1))
        return
    data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()[:12]

    with log.stage("2. Engineer features") as s:
        stats = catalogue_stats(df)
        y = (
            (shows["duration"].str.extract(r"(\d+)")[0].astype(int) >= 2)
            .astype(int)
            .values
        )
        X = concept_frame(concepts_from_shows(shows, stats), stats)
        n_cols = len(make_preprocessor().fit(X, y).get_feature_names_out())
        s.detail = f"10 raw columns became {X.shape[1]} engineered columns and {n_cols} model inputs"
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)

    results, fitted, oof = [], {}, {}
    with log.stage("3. Train, tune and compare models") as s:
        prev = float(max(y.mean(), 1 - y.mean()))
        base = DummyClassifier(strategy="prior").fit(Xtr, ytr)
        bp = base.predict_proba(Xte)[:, 1]
        results.append(
            {
                "name": "Baseline (always the average)",
                "kind": "Baseline",
                "default_cv_auc": 0.5,
                "tuned_cv_auc": 0.5,
                "seconds": 0,
                "params": {},
                "test": {
                    **metrics(yte, bp),
                    "auc": 0.5,
                    "ap": round(float(y.mean()), 4),
                },
                "roc": [[0, 0], [1, 1]],
            }
        )
        for name, (kind, factory, space) in MODELS.items():
            t0 = time.time()
            default_auc = float(
                cross_val_score(
                    pipe(factory), Xtr, ytr, cv=cv, scoring="roc_auc"
                ).mean()
            )
            search = RandomizedSearchCV(
                pipe(factory),
                space,
                n_iter=min(N_ITER, int(np.prod([len(v) for v in space.values()]))),
                cv=cv,
                scoring="roc_auc",
                random_state=SEED,
                n_jobs=1,
            ).fit(Xtr, ytr)
            fitted[name] = search.best_estimator_
            p = search.best_estimator_.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(yte, p)
            results.append(
                {
                    "name": name,
                    "kind": kind,
                    "default_cv_auc": round(default_auc, 4),
                    "tuned_cv_auc": round(float(search.best_score_), 4),
                    "seconds": round(time.time() - t0, 1),
                    "params": {
                        k.replace("model__", ""): (
                            v.item() if hasattr(v, "item") else v
                        )
                        for k, v in search.best_params_.items()
                    },
                    "test": metrics(yte, p),
                    "roc": thin(fpr, tpr),
                }
            )
            print(
                f"    {name:20s} default cv AUC {default_auc:.3f} -> tuned {search.best_score_:.3f}  test {results[-1]['test']['auc']:.3f}",
                flush=True,
            )
        t0 = time.time()
        ens = AvgEnsemble(list(fitted.values()))
        ens_cv = float(cross_val_score(ens, Xtr, ytr, cv=cv, scoring="roc_auc").mean())
        ens.fit(Xtr, ytr)
        fitted["Ensemble (average of three)"] = ens
        p = ens.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p)
        results.append(
            {
                "name": "Ensemble (average of three)",
                "kind": "Ensemble",
                "default_cv_auc": None,
                "tuned_cv_auc": round(ens_cv, 4),
                "seconds": round(time.time() - t0, 1),
                "params": {},
                "test": metrics(yte, p),
                "roc": thin(fpr, tpr),
            }
        )
        best_name = max(
            (r for r in results if r["kind"] != "Baseline"),
            key=lambda r: r["tuned_cv_auc"],
        )["name"]
        final = fitted[best_name]
        s.detail = f"{len(results) - 1} models compared. Best by cross-validated AUC: {best_name}"

    with log.stage("4. Evaluate the chosen model on unseen data") as s:
        oof = cross_val_predict(clone(final), Xtr, ytr, cv=cv, method="predict_proba")[
            :, 1
        ]
        thr_grid = np.linspace(0.15, 0.7, 56)
        thr = float(
            thr_grid[
                int(
                    np.argmax([f1_score(ytr, (oof >= t).astype(int)) for t in thr_grid])
                )
            ]
        )  # threshold picked on training data only
        pt = final.predict_proba(Xte)[:, 1]
        fm = metrics(yte, pt, thr)
        tn, fp, fn, tp = confusion_matrix(yte, (pt >= thr).astype(int)).ravel()
        order = np.argsort(-pt)
        gain = np.cumsum(yte[order]) / yte.sum()
        gain_pts = [
            [round((i + 1) / len(yte), 4), round(float(gain[i]), 4)]
            for i in np.unique(np.linspace(0, len(yte) - 1, 50).astype(int))
        ]
        top20 = float(gain[int(0.2 * len(yte)) - 1])
        deciles = [
            round(
                float(
                    yte[order][
                        int(len(yte) * k / 10) : int(len(yte) * (k + 1) / 10)
                    ].mean()
                ),
                4,
            )
            for k in range(10)
        ]
        fr, mp = calibration_curve(yte, pt, n_bins=8, strategy="quantile")
        thresholds = [
            {
                "t": round(float(t), 2),
                "precision": round(
                    float(precision_score(yte, (pt >= t).astype(int), zero_division=0)),
                    3,
                ),
                "recall": round(float(recall_score(yte, (pt >= t).astype(int))), 3),
                "share_flagged": round(float((pt >= t).mean()), 3),
            }
            for t in np.arange(0.2, 0.71, 0.05)
        ]
        final_report = {
            "model": best_name,
            "threshold": round(thr, 2),
            "test": fm,
            "train_rows": int(len(ytr)),
            "test_rows": int(len(yte)),
            "base_rate": round(float(yte.mean()), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "gain": gain_pts,
            "lift_top20": round(top20 / 0.2, 2),
            "capture_top20": round(top20, 3),
            "deciles": deciles,
            "calibration": [
                [round(float(a), 3), round(float(b), 3)] for a, b in zip(mp, fr)
            ],
            "thresholds": thresholds,
        }
        s.detail = f"test AUC {fm['auc']}, top 20% of predictions capture {top20 * 100:.0f}% of renewed series"

    with log.stage("5. Explain the model") as s:
        rng = np.random.default_rng(SEED)
        base_auc, importance = roc_auc_score(yte, pt), []
        for gname, cols in FEATURE_GROUPS.items():
            drops = []
            for _ in range(10):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xte[cols].values[perm]
                drops.append(
                    base_auc - roc_auc_score(yte, final.predict_proba(Xp)[:, 1])
                )
            importance.append(
                {
                    "group": gname,
                    "drop": round(float(np.mean(drops)), 4),
                    "sd": round(float(np.std(drops)), 4),
                }
            )
        importance.sort(key=lambda d: -d["drop"])

        def mean_p(frame):
            return float(final.predict_proba(frame)[:, 1].mean())

        concepts = concepts_from_shows(shows, stats)
        base_p = mean_p(X)
        pdp = {}
        for col, grid in (
            ("release_year", list(range(2010, 2022))),
            ("years_to_add", list(range(0, 7))),
            ("n_genres", [1, 2, 3]),
        ):
            vals = []
            for v in grid:
                Xm = X.copy()
                Xm[col] = v
                if col == "years_to_add":
                    Xm["fresh_add"] = int(v <= 1)
                vals.append([v, round(mean_p(Xm), 4)])
            pdp[col] = vals

        def effect(mutator):
            return round(
                (
                    mean_p(concept_frame([mutator(dict(c)) for c in concepts], stats))
                    - base_p
                )
                * 100,
                2,
            )

        all_genres = sorted(
            g for g, f in stats["genre_freq"].items() if f * len(shows) >= 40
        )  # rarer genres are too thin to learn from
        lev_genres = []
        for g in all_genres:
            on = mean_p(
                concept_frame(
                    [{**c, "genres": sorted(set(c["genres"]) | {g})} for c in concepts],
                    stats,
                )
            )
            off = mean_p(
                concept_frame(
                    [
                        {
                            **c,
                            "genres": [x for x in c["genres"] if x != g] or c["genres"],
                        }
                        for c in concepts
                    ],
                    stats,
                )
            )
            lev_genres.append({"name": g, "effect": round((on - off) * 100, 2)})
        lev_countries = [
            {"name": c, "effect": effect(lambda d, c=c: {**d, "country": c})}
            for c in stats["countries"]
        ]
        ratings = sorted(shows["rating"].value_counts()[lambda v: v >= 20].index)
        lev_ratings = [
            {"name": r, "effect": effect(lambda d, r=r: {**d, "rating": r})}
            for r in ratings
        ]
        levers = {
            "genres": sorted(lev_genres, key=lambda d: -d["effect"]),
            "countries": sorted(lev_countries, key=lambda d: -d["effect"]),
            "ratings": sorted(lev_ratings, key=lambda d: -d["effect"]),
        }
        s.detail = "permutation importance by feature group, partial dependence and model-implied levers"

    with log.stage("6. Monitor: slices and drift") as s:
        te = Xte.copy()
        te["y"], te["p"] = yte, pt
        slices = []
        for col in ("audience", "country", "has_director"):
            for val, g in te.groupby(col):
                if len(g) >= 25 and 6 <= g["y"].sum() <= len(g) - 6:
                    nice = {
                        "audience": "Audience",
                        "country": "Country",
                        "has_director": "Director credited",
                    }[col]
                    slices.append(
                        {
                            "slice": f"{nice}: {('yes' if val == 1 else 'no') if col == 'has_director' else val}",
                            "n": int(len(g)),
                            "rate": round(float(g["y"].mean()), 3),
                            "auc": round(float(roc_auc_score(g["y"], g["p"])), 3),
                            "accuracy": round(
                                float(
                                    accuracy_score(g["y"], (g["p"] >= thr).astype(int))
                                ),
                                3,
                            ),
                        }
                    )
        added = pd.to_datetime(shows["date_added"]).dt.year.values
        ref, cur = X[added <= 2019], X[added >= 2020]
        drift = []
        for col in (
            "audience",
            "country",
            "n_genres",
            "years_to_add",
            "has_director",
            "n_countries",
        ):
            bins = sorted(set(X[col].unique()))
            drift.append(
                {
                    "feature": col.replace("_", " "),
                    "psi": round(psi(ref[col].values, cur[col].values, bins), 3),
                }
            )
        top_genres = [
            g
            for g in all_genres
            if (X["genres"].str.contains(g.replace(" ", "_"), regex=False)).sum() >= 100
        ]
        gshift = np.mean(
            [
                abs(
                    ref["genres"].str.contains(g.replace(" ", "_"), regex=False).mean()
                    - cur["genres"]
                    .str.contains(g.replace(" ", "_"), regex=False)
                    .mean()
                )
                for g in top_genres
            ]
        )
        drift.append(
            {"feature": "genre mix (average shift)", "psi": round(float(gshift) * 2, 3)}
        )
        drift.sort(key=lambda d: -d["psi"])
        pr, pc = final.predict_proba(ref)[:, 1], final.predict_proba(cur)[:, 1]
        drift_report = {
            "features": drift,
            "reference": "added in 2019 or earlier",
            "current": "added in 2020 or later",
            "n_ref": int(len(ref)),
            "n_cur": int(len(cur)),
            "score_ref": round(float(pr.mean()), 3),
            "score_cur": round(float(pc.mean()), 3),
            "label_ref": round(float(y[added <= 2019].mean()), 3),
            "label_cur": round(float(y[added >= 2020].mean()), 3),
        }
        s.detail = f"{len(slices)} slices checked, largest shift PSI {drift[0]['psi']}"

    with log.stage("7. Analyse segments and generate insights") as s:
        seg_src = shows.assign(
            renewed=y,
            primary=shows["country"]
            .str.split(",")
            .str[0]
            .str.strip()
            .replace("", "Unknown"),
            audience=shows["rating"].map(AUDIENCE).fillna("Unrated"),
            added_year=added,
        )
        overall = float(y.mean())

        def table(frame, col, min_n):
            out = []
            for k, g in frame.groupby(col):
                n, pos = len(g), int(g["renewed"].sum())
                if n >= min_n:
                    lo, hi = wilson(pos, n)
                    out.append(
                        {
                            "name": str(k),
                            "n": n,
                            "rate": round(pos / n, 3),
                            "lo": round(lo, 3),
                            "hi": round(hi, 3),
                        }
                    )
            return out

        g_rows = (
            seg_src.assign(g=seg_src["listed_in"].str.split(","))
            .explode("g")
            .assign(g=lambda d: d["g"].str.strip())
        )
        segments = {
            "overall": round(overall, 4),
            "n": int(len(shows)),
            "Genre": table(g_rows, "g", 25),
            "Country": table(seg_src, "primary", 30),
            "Audience": table(seg_src, "audience", 30),
            "Release year": table(
                seg_src[seg_src["release_year"] >= 2012], "release_year", 30
            ),
        }
        insights = build_insights(
            df,
            shows,
            seg_src,
            g_rows,
            segments,
            importance,
            levers,
            final_report,
            drift_report,
            slices,
            checks,
            added,
            y,
            stats,
        )
        s.detail = f"{len(insights)} insights written"

    with log.stage("8. Package report, model and registry") as s:
        MODELDIR.mkdir(exist_ok=True)
        reg = (
            json.loads(REGISTRY.read_text())
            if REGISTRY.exists()
            else {"versions": [], "production": None}
        )
        version = len(reg["versions"]) + 1
        prod = next(
            (v for v in reg["versions"] if v["version"] == reg["production"]), None
        )
        promote = prod is None or fm["auc"] >= prod["test_auc"] - PROMOTION_TOLERANCE
        artifact = f"models/success_v{version}.pkl"
        options = {
            "genres": all_genres,
            "countries": stats["countries"],
            "ratings": ratings,
            "years": [2010, 2021],
            "base_rate": round(overall, 4),
        }
        with open(ROOT / "ml" / artifact, "wb") as fh:
            pickle.dump(
                {
                    "model": final,
                    "name": best_name,
                    "stats": stats,
                    "options": options,
                    "score_dist": sorted(
                        round(float(v), 4) for v in final.predict_proba(X)[:, 1]
                    ),
                    "version": version,
                    "threshold": thr,
                },
                fh,
            )
        entry = {
            "version": version,
            "run_id": run_id,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "data_hash": data_hash,
            "model": best_name,
            "test_auc": fm["auc"],
            "cv_auc": next(
                r["tuned_cv_auc"] for r in results if r["name"] == best_name
            ),
            "artifact": artifact,
            "status": "production" if promote else "rejected: worse than production",
        }
        if promote:
            for v in reg["versions"]:
                if v["status"] == "production":
                    v["status"] = "archived"
            reg["production"] = version
        reg["versions"].append(entry)
        REGISTRY.write_text(json.dumps(reg, indent=1))
        s.detail = f"model version {version} {'promoted to production' if promote else 'kept as challenger'}"
    total = round(time.time() - t_start, 1)

    report = {
        "meta": {
            "run_id": run_id,
            "created": entry["created"],
            "data_hash": data_hash,
            "rows": int(len(df)),
            "series": int(len(shows)),
            "seed": SEED,
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "seconds": total,
            "folds": FOLDS,
            "n_iter": N_ITER,
            "target": "TV series with two or more seasons on the platform",
        },
        "stages": log.items,
        "checks": checks,
        "features": {
            "groups": FEATURE_DOC,
            "raw": 10,
            "engineered": int(X.shape[1]),
            "inputs": int(n_cols),
            "guard": "The number of seasons defines the target, so it is never used as a feature.",
        },
        "models": results,
        "final": final_report,
        "importance": importance,
        "pdp": pdp,
        "levers": levers,
        "segments": segments,
        "slices": slices,
        "drift": drift_report,
        "portfolio": portfolio(df, shows),
        "insights": insights,
        "summary": summary(shows, segments, insights, final_report, importance),
        "registry": reg,
        "card": model_card(best_name, final_report, data_hash, thr, overall),
    }
    REPORT.write_text(json.dumps(report, indent=1))
    with open(RUNLOG, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "created": entry["created"],
                    "model": best_name,
                    "test_auc": fm["auc"],
                    "version": version,
                    "seconds": total,
                    "data_hash": data_hash,
                }
            )
            + "\n"
        )
    print(
        f"\nDone in {total}s. Best: {best_name}  test AUC {fm['auc']}  version {version} ({entry['status']})"
    )


#  portfolio view of the whole catalogue (movies and shows)
def portfolio(df, shows):
    added = pd.to_datetime(df["date_added"]).dt.year
    yearly = [
        {
            "year": int(yr),
            "movies": int(((added == yr) & (df["type"] == "Movie")).sum()),
            "shows": int(((added == yr) & (df["type"] == "TV Show")).sum()),
        }
        for yr in range(2015, int(added.max()) + 1)
    ]
    g = (
        df.assign(g=df["listed_in"].str.split(","))
        .explode("g")
        .assign(g=lambda d: d["g"].str.strip())
        .reset_index(drop=True)
    )
    gt = pd.crosstab(g["g"], g["type"])
    top_g = gt.assign(t=gt.sum(axis=1)).sort_values("t", ascending=False).head(10)
    primary = df["country"].str.split(",").str[0].str.strip().replace("", "Unknown")
    ct = pd.crosstab(primary, df["type"])
    top_c = ct.assign(t=ct.sum(axis=1)).sort_values("t", ascending=False).head(10)
    aud = df["rating"].map(AUDIENCE).fillna("Unrated")
    return {
        "total": int(len(df)),
        "movies": int((df["type"] == "Movie").sum()),
        "shows": int(len(shows)),
        "yearly": yearly,
        "genres": [
            {"name": k, "movies": int(r["Movie"]), "shows": int(r["TV Show"])}
            for k, r in top_g.iterrows()
        ],
        "countries": [
            {"name": k, "movies": int(r["Movie"]), "shows": int(r["TV Show"])}
            for k, r in top_c.iterrows()
        ],
        "audience": [
            {
                "name": a,
                "movies": int(((aud == a) & (df["type"] == "Movie")).sum()),
                "shows": int(((aud == a) & (df["type"] == "TV Show")).sum()),
            }
            for a in ["Kids", "Family", "Teens", "Adults"]
        ],
    }


#  automated insights
def build_insights(
    df,
    shows,
    seg,
    g_rows,
    segments,
    importance,
    levers,
    fin,
    drift,
    slices,
    checks,
    added,
    y,
    stats,
):
    out, overall, n_shows = [], segments["overall"], len(shows)

    def add(cat, title, text, action, impact, conf, evidence):
        out.append(
            {
                "category": cat,
                "title": title,
                "text": text,
                "action": action,
                "impact": impact,
                "confidence": conf,
                "evidence": evidence,
            }
        )

    pct = lambda v: f"{v * 100:.0f}%"
    movies = int((df["type"] == "Movie").sum())
    yr = pd.to_datetime(df["date_added"]).dt.year.value_counts().sort_index()
    peak_year, peak = int(yr.loc[:2020].idxmax()), int(yr.loc[:2020].max())
    last_full = int(yr.loc[:2020].index.max())
    add(
        "Portfolio",
        f"The catalogue is {pct(movies / len(df))} movies, and additions peaked in {peak_year}",
        f"Of {len(df):,} titles, {movies:,} are movies and {n_shows:,} are TV series. Additions rose every year until {peak_year} ({peak:,} titles) and were {int(yr.loc[last_full]):,} in {last_full}, "
        f"{abs(yr.loc[last_full] / peak - 1) * 100:.0f}% {'below' if yr.loc[last_full] < peak else 'above'} the peak.",
        "Use the peak year as the benchmark when judging whether acquisition volume is recovering.",
        3,
        "High",
        {"titles": len(df), "peak_year": peak_year},
    )

    add(
        "Success",
        f"Only {pct(overall)} of series reach a second season",
        f"Across {n_shows:,} TV series, {int(y.sum()):,} have two or more seasons. This is the success benchmark every segment below is compared with.",
        f"Treat a second season as a stretch outcome. A new concept needs to beat {pct(overall)} to be above average.",
        4,
        "High",
        {"rate": round(overall, 3), "n": n_shows},
    )

    gen = sorted(
        [r for r in segments["Genre"] if r["n"] >= 40], key=lambda r: -r["rate"]
    )
    best, worst = gen[:3], gen[-3:][::-1]
    add(
        "Genre",
        f"{best[0]['name']}, {best[1]['name']} and {best[2]['name']} renew most often",
        "; ".join(
            f"{r['name']} {pct(r['rate'])} (95% range {pct(r['lo'])} to {pct(r['hi'])}, {r['n']} series)"
            for r in best
        )
        + f". The average is {pct(overall)}.",
        "Prioritise these genres when comparing otherwise similar pitches, and check the ranges: smaller genres are less certain.",
        5,
        conf_label(best[0]["n"], best[0]["lo"], best[0]["hi"]),
        {"genres": [r["name"] for r in best]},
    )
    add(
        "Genre",
        f"{worst[0]['name']}, {worst[1]['name']} and {worst[2]['name']} rarely get a second season",
        "; ".join(f"{r['name']} {pct(r['rate'])} ({r['n']} series)" for r in worst)
        + f", against an average of {pct(overall)}.",
        "Plan these as limited series or one-off events, and avoid budgeting for multiple seasons.",
        4,
        conf_label(worst[0]["n"], worst[0]["lo"], worst[0]["hi"]),
        {"genres": [r["name"] for r in worst]},
    )

    ctry = sorted(
        [r for r in segments["Country"] if r["n"] >= 40 and r["name"] != "Unknown"],
        key=lambda r: -r["rate"],
    )
    if len(ctry) >= 4:
        add(
            "Market",
            f"Country matters: {ctry[0]['name']} leads and {ctry[-1]['name']} trails",
            f"{ctry[0]['name']} series renew {pct(ctry[0]['rate'])} of the time ({ctry[0]['n']} series), while {ctry[-1]['name']} renews {pct(ctry[-1]['rate'])} ({ctry[-1]['n']}). "
            f"Next best are {ctry[1]['name']} ({pct(ctry[1]['rate'])}) and {ctry[2]['name']} ({pct(ctry[2]['rate'])}).",
            "Use country as a context signal, not a rule. Production country mixes with genre, so check the genre insight too.",
            3,
            conf_label(ctry[0]["n"], ctry[0]["lo"], ctry[0]["hi"]),
            {"top": ctry[0]["name"], "bottom": ctry[-1]["name"]},
        )

    aud = {r["name"]: r for r in segments["Audience"]}
    if "Kids" in aud and "Adults" in aud:
        add(
            "Audience",
            f"Kids' series renew {pct(aud['Kids']['rate'])} of the time, adult series {pct(aud['Adults']['rate'])}",
            "; ".join(
                f"{k} {pct(v['rate'])} ({v['n']})"
                for k, v in aud.items()
                if k != "Unrated"
            )
            + ". Children's content is easier to keep alive across seasons.",
            "Family and kids content is the safest bet for multi-season planning.",
            4,
            conf_label(aud["Kids"]["n"], aud["Kids"]["lo"], aud["Kids"]["hi"]),
            {"kids": aud["Kids"]["rate"], "adults": aud["Adults"]["rate"]},
        )

    d = seg.groupby(seg["director"] != "")["renewed"].agg(["mean", "count"])
    if True in d.index and False in d.index and d.loc[True, "count"] >= 50:
        add(
            "Creator",
            f"Series with a credited director renew less ({pct(d.loc[True, 'mean'])} against {pct(d.loc[False, 'mean'])})",
            f"Only {int(d.loc[True, 'count'])} series list a director. They renew {pct(d.loc[True, 'mean'])} of the time, compared with {pct(d.loc[False, 'mean'])} for the {int(d.loc[False, 'count']):,} that do not.",
            "A credited director often marks a limited series or a film-like production, which is not planned for more seasons. Do not read this as directors causing cancellations.",
            2,
            "Medium",
            {
                "with": round(d.loc[True, "mean"], 3),
                "without": round(d.loc[False, "mean"], 3),
            },
        )

    top = importance[:3]
    add(
        "Model",
        f"{top[0]['group']} is the strongest driver, followed by {top[1]['group'].lower()} and {top[2]['group'].lower()}",
        f"Shuffling {top[0]['group'].lower()} information lowers the model's AUC by {top[0]['drop']:.3f}, {top[1]['group'].lower()} by {top[1]['drop']:.3f} and {top[2]['group'].lower()} by {top[2]['drop']:.3f}. "
        "Features that barely move the score add little and could be dropped.",
        "Spend modelling and data-collection effort on the strongest drivers first.",
        3,
        "High",
        {"top": [t["group"] for t in top]},
    )

    up, down = levers["genres"][:2], levers["genres"][-2:][::-1]
    add(
        "Model",
        f"Model-implied levers: adding {up[0]['name']} helps, adding {down[0]['name']} hurts",
        f"Holding everything else fixed, the model predicts {up[0]['effect']:+.1f} points of renewal chance from adding {up[0]['name']} and {up[1]['effect']:+.1f} from {up[1]['name']}, "
        f"but {down[0]['effect']:+.1f} from {down[0]['name']} and {down[1]['effect']:+.1f} from {down[1]['name']}.",
        "Use these as discussion points for concept design. They describe patterns in past data, not guaranteed effects.",
        3,
        "Medium",
        {"up": up[0]["name"], "down": down[0]["name"]},
    )

    shares = g_rows["g"].value_counts() / n_shows
    gaps = [
        r
        for r in gen
        if r["rate"] >= overall + 0.08 and shares.get(r["name"], 0) <= 0.09
    ]
    if gaps:
        gp = sorted(gaps, key=lambda r: -r["rate"])[:2]
        add(
            "Opportunity",
            "Under-served genres with strong renewal: "
            + " and ".join(r["name"] for r in gp),
            "; ".join(
                f"{r['name']} renews {pct(r['rate'])} but makes up only {pct(shares[r['name']])} of series ({r['n']} titles)"
                for r in gp
            )
            + ".",
            "Consider commissioning more in these niches, and validate with a pilot because the sample is small.",
            5,
            conf_label(gp[0]["n"], gp[0]["lo"], gp[0]["hi"]),
            {"genres": [r["name"] for r in gp]},
        )
    heavy = [
        r
        for r in gen
        if r["rate"] <= overall - 0.08 and shares.get(r["name"], 0) >= 0.10
    ]
    if heavy:
        hv = sorted(heavy, key=lambda r: -shares[r["name"]])[:2]
        add(
            "Risk",
            "Large share of series sits in low-renewal genres: "
            + " and ".join(r["name"] for r in hv),
            "; ".join(
                f"{r['name']} is {pct(shares[r['name']])} of all series but renews only {pct(r['rate'])}"
                for r in hv
            )
            + ".",
            "Check whether these titles are meant as limited series. If not, the portfolio is exposed to cancellations.",
            4,
            conf_label(hv[0]["n"], hv[0]["lo"], hv[0]["hi"]),
            {"genres": [r["name"] for r in hv]},
        )

    MON = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    full = (added >= 2016) & (
        added <= 2020
    )  # complete years only, because 2021 stops in September and renews more often
    month_f = pd.to_datetime(shows["date_added"]).dt.month[full]
    p_month = chi2_contingency(pd.crosstab(month_f, y[full]))[1]
    mrates = pd.DataFrame({"m": month_f.values, "y": y[full]}).groupby("m")["y"].mean()
    ptxt = "p < 0.001" if p_month < 0.001 else f"p = {p_month:.3f}"
    if p_month >= 0.01:
        add(
            "Timing",
            "The month a series is added makes little difference",
            f"Looking at complete years (2016 to 2020), a chi-square test of month added against renewal gives {ptxt}, so there is no reliable timing effect.",
            "Do not rely on launch month to improve the odds of a second season.",
            2,
            "High",
            {"p_value": round(float(p_month), 3)},
        )
    else:
        hi_m, lo_m = int(mrates.idxmax()), int(mrates.idxmin())
        add(
            "Timing",
            f"Series added in {MON[hi_m - 1]} renew most often, those added in {MON[lo_m - 1]} least",
            f"Looking at complete years (2016 to 2020), renewal is {pct(mrates[hi_m])} for series added in {MON[hi_m - 1]} and {pct(mrates[lo_m])} in {MON[lo_m - 1]} ({ptxt}).",
            "Treat this as a pattern to investigate. Launch timing may be linked to the type of series released in each month.",
            2,
            "Medium",
            {"p_value": round(float(p_month), 4)},
        )
    by_year = (
        pd.DataFrame({"yr": added, "y": y}).groupby("yr")["y"].agg(["mean", "count"])
    )
    by_year = by_year[(by_year["count"] >= 100) & (by_year.index >= 2016)]
    if len(by_year) >= 3:
        add(
            "Data",
            f"The success signal has limits: later arrivals renew more ({pct(by_year['mean'].iloc[0])} in {by_year.index[0]}, {pct(by_year['mean'].iloc[-1])} in {by_year.index[-1]})",
            "Seasons are counted on the platform at the time of the data snapshot. Series that were licensed after running elsewhere can arrive with several seasons already, so the signal mixes renewals with licensing.",
            "Read renewal as a proxy for sustained demand. Replace it with viewing hours or retention when that data is available.",
            3,
            "High",
            {
                "first": round(float(by_year["mean"].iloc[0]), 3),
                "last": round(float(by_year["mean"].iloc[-1]), 3),
            },
        )

    recent, older = added >= 2020, added <= 2019
    sh_g = []
    for g in [r["name"] for r in gen if r["n"] >= 60]:
        has = shows["listed_in"].str.contains(g, regex=False)
        sh_g.append((g, has[recent].mean() - has[older].mean(), has[recent].sum()))
    sh_g = [t for t in sh_g if t[2] >= 20]
    if len(sh_g) >= 4:
        sh_g.sort(key=lambda t: -t[1])
        add(
            "Trend",
            f"Since 2020 {sh_g[0][0]} grew and {sh_g[-1][0]} shrank in the series mix",
            f"Among series added in 2020 or later, {sh_g[0][0]} is {sh_g[0][1] * 100:+.1f} percentage points of the mix compared with earlier years, and {sh_g[-1][0]} is {sh_g[-1][1] * 100:+.1f}. "
            f"Next: {sh_g[1][0]} ({sh_g[1][1] * 100:+.1f}) and {sh_g[-2][0]} ({sh_g[-2][1] * 100:+.1f}).",
            "Compare this shift with the renewal rates above to see if investment is moving towards or away from strong genres.",
            3,
            "Medium",
            {"up": sh_g[0][0], "down": sh_g[-1][0]},
        )

    t = fin["test"]
    add(
        "Model",
        f"The model ranks series well: its top 20% of picks hold {pct(fin['capture_top20'])} of renewals",
        f"The {short(fin['model'])} reaches an AUC of {t['auc']} on {fin['test_rows']} series it never saw, with a lift of {fin['lift_top20']}x in the top 20% of its scores. "
        f"At a threshold of {fin['threshold']} it finds {pct(t['recall'])} of renewals with {pct(t['precision'])} precision.",
        "Use it to screen and rank concepts, not to make a final decision. Metadata alone cannot see script quality, cast or marketing.",
        4,
        "High",
        {"auc": t["auc"], "lift": fin["lift_top20"]},
    )

    weak = sorted([s for s in slices if s["n"] >= 40], key=lambda s: s["auc"])[:1]
    if weak:
        add(
            "Model",
            f"The model is weakest on this slice: {weak[0]['slice']}",
            f"AUC is {weak[0]['auc']} on {weak[0]['n']} test series in this slice, compared with {t['auc']} overall.",
            "Add more data or a dedicated model before using scores for this segment.",
            2,
            "Medium",
            {"slice": weak[0]["slice"]},
        )

    warn = [c for c in checks if c["status"] == "warn"]
    miss_c = float((df["country"] == "").mean())
    add(
        "Data",
        f"Data quality: {pct(float((df['director'] == '').mean()))} of titles have no director and {pct(miss_c)} no country",
        "Missing creator and country information limits what any model can learn. "
        + (
            f"{len(warn)} validation warnings were raised: "
            + "; ".join(c["name"] for c in warn)
            + "."
            if warn
            else "All validation checks passed."
        ),
        "Enrich the catalogue with cast, budget, viewing hours and review scores. A true success metric would transform this engine.",
        3,
        "High",
        {"warnings": len(warn)},
    )

    top_d = drift["features"][0]
    add(
        "Monitoring",
        f"Series added since 2020 differ most in: {top_d['feature']} (PSI {top_d['psi']})",
        f"Population stability index compares series added in 2019 or earlier with those added since. Below 0.10 is stable, 0.10 to 0.25 moderate, above 0.25 large. "
        f"The real renewal rate moved from {pct(drift['label_ref'])} to {pct(drift['label_cur'])} and the model's average prediction from {pct(drift['score_ref'])} to {pct(drift['score_cur'])}.",
        "Retrain on a schedule and watch this table. A large shift means new series look different from the ones the model learned from.",
        2,
        "Medium",
        {"psi": top_d["psi"]},
    )

    out.sort(
        key=lambda i: (
            -i["impact"],
            {"High": 0, "Medium": 1, "Low": 2}[i["confidence"]],
        )
    )
    for k, i in enumerate(out):
        i["id"] = k + 1
    return out


def summary(shows, segments, insights, fin, importance):
    by = {i["title"]: i for i in insights}
    gen = sorted(
        [r for r in segments["Genre"] if r["n"] >= 40], key=lambda r: -r["rate"]
    )
    opp = next((i for i in insights if i["category"] == "Opportunity"), None)
    risk = next((i for i in insights if i["category"] == "Risk"), None)
    parts = [
        f"Of {segments['n']:,} TV series in the catalogue, {segments['overall'] * 100:.0f}% reached a second season, which is used here as the success signal.",
        f"Renewal is strongest in {gen[0]['name']} ({gen[0]['rate'] * 100:.0f}%) and {gen[1]['name']} ({gen[1]['rate'] * 100:.0f}%) and weakest in {gen[-1]['name']} ({gen[-1]['rate'] * 100:.0f}%).",
        f"{phrase(fin['model'])[0].upper() + phrase(fin['model'])[1:]} trained on {fin['train_rows']:,} series ranks unseen series with an AUC of {fin['test']['auc']}, and its top 20% of picks contain {fin['capture_top20'] * 100:.0f}% of all renewals.",
    ]
    if opp:
        parts.append(opp["title"] + ".")
    if risk:
        parts.append(risk["title"] + ".")
    parts.append(f"{importance[0]['group']} information drives the predictions most.")
    return " ".join(parts)


def model_card(name, fin, data_hash, thr, base):
    return {
        "name": f"Series renewal model ({name})",
        "intended_use": "Rank TV series concepts by their chance of reaching a second season, as one input to planning and portfolio review.",
        "not_for": "Final greenlight decisions, predicting audience size, or judging movies.",
        "data": f"Netflix catalogue metadata snapshot (hash {data_hash}), {fin['train_rows'] + fin['test_rows']:,} TV series.",
        "target": f"Two or more seasons on the platform. {base * 100:.0f}% of series.",
        "metrics": f"AUC {fin['test']['auc']}, average precision {fin['test']['ap']}, Brier score {fin['test']['brier']} on {fin['test_rows']} unseen series.",
        "threshold": f"{thr:.2f}, chosen on training data to maximise F1.",
        "limits": [
            "The target is a proxy for success, not viewing hours or profit.",
            "Seasons include licensed series that ran elsewhere, so the target mixes renewals with licensing.",
            "Only catalogue metadata is used: no cast, budget, script, marketing or reviews.",
            "Effects shown for genres and countries are patterns in past data, not causal effects.",
        ],
    }


# scoring for the website
def score_concept(bundle, c):
    """Success probability for one concept, plus what-if levers (one change at a time)."""
    stats, model, o = bundle["stats"], bundle["model"], bundle["options"]
    c = {
        **c,
        "added_year": c.get("added_year", c["release_year"]),
        "n_countries": 1,
        "director_titles": 0,
    }
    variants, labels = [c], ["base"]
    for g in o["genres"]:
        if g in c["genres"]:
            if len(c["genres"]) > 1:
                variants.append({**c, "genres": [x for x in c["genres"] if x != g]})
                labels.append(f"Remove genre {g}")
        else:
            variants.append({**c, "genres": c["genres"] + [g]})
            labels.append(f"Add genre {g}")
    for k in o["countries"]:
        if k != c["country"]:
            variants.append({**c, "country": k})
            labels.append(f"Change country to {k}")
    for r in o["ratings"]:
        if r != c["rating"]:
            variants.append({**c, "rating": r})
            labels.append(f"Change rating to {r}")
    p = model.predict_proba(concept_frame(variants, stats))[:, 1]
    base = float(p[0])
    levers = sorted(
        (
            {
                "change": lab,
                "prob": round(float(v), 4),
                "delta": round((float(v) - base) * 100, 1),
            }
            for lab, v in zip(labels[1:], p[1:])
        ),
        key=lambda d: -d["delta"],
    )
    dist = np.array(bundle["score_dist"])
    return {
        "probability": round(base, 4),
        "percentile": round(float((dist < base).mean() * 100), 0),
        "base_rate": o["base_rate"],
        "threshold": bundle["threshold"],
        "up": levers[:4],
        "down": levers[-3:][::-1],
        "model": bundle["name"],
        "version": bundle["version"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Netflix content success analytics engine")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="only run the data validation checks",
    )
    args = ap.parse_args()
    import engine 

    engine.main(validate_only=args.validate_only)
