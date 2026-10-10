import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import rating_model as rm  
from recommender import load_data

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "ml" / "cluster_metrics.json"
POINTS = ROOT / "ml" / "cluster_points.json"
SEED = 42
K_RANGE = list(range(2, 13))
K_SHOW = list(range(2, 11))  
MIN_SHARE = 0.03  # a cluster must hold at least 3% of titles to count as meaningful

# Relative weight of each feature block. Genres and type carry the most meaning, numbers are kept gentle.
WEIGHTS = {
    "Genres": 1.0,
    "Type": 1.0,
    "Audience rating": 1.0,
    "Country": 0.7,
    "Numbers": 0.5,
}
FEATURE_INFO = [
    {
        "name": "Genres",
        "kind": "Categorical",
        "how": "Multi-hot encoding (one column per genre)",
        "weight": 1.0,
        "note": "The main content signal.",
    },
    {
        "name": "Movie or TV show",
        "kind": "Categorical",
        "how": "One-hot encoding",
        "weight": 1.0,
        "note": "Movies and shows are built very differently.",
    },
    {
        "name": "Audience rating",
        "kind": "Categorical",
        "how": "Kids, Family, Teens, Adults or Unrated, one-hot",
        "weight": 1.0,
        "note": "The 14 raw ratings grouped as in Task 3.",
    },
    {
        "name": "Country (first listed)",
        "kind": "Categorical",
        "how": "One-hot, top 15 countries plus Other",
        "weight": 0.7,
        "note": "Down-weighted so country does not dominate.",
    },
    {
        "name": "Release year",
        "kind": "Numerical",
        "how": "StandardScaler (mean 0, spread 1)",
        "weight": 0.5,
        "note": "",
    },
    {
        "name": "Length (minutes and seasons)",
        "kind": "Numerical",
        "how": "Two StandardScaler columns",
        "weight": 0.5,
        "note": "Minutes for movies, seasons for shows.",
    },
    {
        "name": "Number of genres",
        "kind": "Numerical",
        "how": "StandardScaler",
        "weight": 0.5,
        "note": "",
    },
    {
        "name": "Director listed",
        "kind": "Left out",
        "how": "",
        "weight": 0,
        "note": "Almost the same information as movie or show, so it would count it twice.",
    },
    {
        "name": "Title, ID, director name, date added",
        "kind": "Left out",
        "how": "",
        "weight": 0,
        "note": "Unique text or identifiers with no group structure.",
    },
]
COLORS = 10


def build_matrix(df, frame):
    cv = CountVectorizer(
        tokenizer=rm.split_genres, token_pattern=None, lowercase=False, binary=True
    )
    genres = cv.fit_transform(frame["genres"]).toarray().astype(float)
    audience = df["rating"].map(rm.RATING_TO_GROUP).fillna("Unrated")
    numbers = StandardScaler().fit_transform(
        frame[["release_year", "minutes", "seasons", "n_genres"]].astype(float)
    )
    blocks = [
        genres * WEIGHTS["Genres"],
        pd.get_dummies(frame["type"]).values.astype(float) * WEIGHTS["Type"],
        pd.get_dummies(audience).values.astype(float) * WEIGHTS["Audience rating"],
        pd.get_dummies(frame["country"]).values.astype(float) * WEIGHTS["Country"],
        numbers * WEIGHTS["Numbers"],
    ]
    X = np.hstack(blocks)
    return (
        X,
        genres,
        [g.replace("_", " ") for g in cv.get_feature_names_out()],
        audience,
    )


def scores(Z, labels):
    return {
        "silhouette": round(float(silhouette_score(Z, labels, random_state=SEED)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Z, labels)), 3),
        "calinski_harabasz": round(float(calinski_harabasz_score(Z, labels)), 0),
    }


def short_genre(g):
    for bad in (" TV Shows", " Movies", " TV"):
        g = g.replace(bad, "")
    return g[3:] if g.startswith("TV ") else g


def relabel_by_size(labels):
    """Cluster 0 is the biggest, so numbering means the same thing every time."""
    order = [
        c
        for c, _ in sorted(
            zip(*np.unique(labels, return_counts=True)), key=lambda t: -t[1]
        )
    ]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[v] for v in labels])


def profile(df, genres_m, genre_names, audience, Z, labels):
    """Step 3 and 5: describe every cluster in plain numbers, then name it."""
    overall = genres_m.mean(0)
    countries = df["country"].str.split(",").str[0].str.strip().replace("", "Unknown")
    out = []
    for c in range(labels.max() + 1):
        m = labels == c
        sub = df[m]
        share = genres_m[m].mean(0)
        lift = share / np.maximum(overall, 1e-9)
        score = share * np.sqrt(
            lift
        )  # a genre must be both common in the cluster and distinctive for it
        order = [i for i in np.argsort(-score) if share[i] >= 0.10][:4] or list(
            np.argsort(-share)[:3]
        )
        aud = audience[m].value_counts(normalize=True)
        mov = float((sub["type"] == "Movie").mean())
        centre = Z[m].mean(0)
        near = np.argsort(((Z[m] - centre) ** 2).sum(1))[:5]
        out.append(
            {
                "id": c,
                "size": int(m.sum()),
                "share": round(float(m.mean()), 4),
                "movie_share": round(mov, 3),
                "median_year": int(sub["release_year"].median()),
                "avg_genres": round(float(genres_m[m].sum(1).mean()), 2),
                "audience": {
                    g: round(float(aud.get(g, 0)), 3)
                    for g in list(rm.GROUPS) + ["Unrated"]
                },
                "genres": [
                    {"name": genre_names[i], "share": round(float(share[i]), 3)}
                    for i in order
                ],
                "countries": [
                    {"name": k, "share": round(float(v), 3)}
                    for k, v in countries[m]
                    .value_counts(normalize=True)
                    .head(3)
                    .items()
                ],
                "examples": [
                    {
                        "id": int(np.flatnonzero(m)[j]),
                        "title": sub["title"].iloc[j],
                        "year": int(sub["release_year"].iloc[j]),
                    }
                    for j in near
                ],
                "_top_genre": genre_names[order[0]],
            }
        )
    name_clusters(out)
    for p in out:
        p.pop("_top_genre")
        p["summary"] = describe(p)
    return out


def name_clusters(profiles):
    def type_word(p):
        return (
            "Movies"
            if p["movie_share"] >= 0.85
            else "TV Shows" if p["movie_share"] <= 0.15 else "Movies & Shows"
        )

    for p in profiles:
        genre = short_genre(p["_top_genre"])
        p["name"] = (
            genre
            if genre in ("Documentaries", "Docuseries")
            else f"{genre} {type_word(p)}"
        )
    for name in {p["name"] for p in profiles}:  # make duplicate names different
        twins = [p for p in profiles if p["name"] == name]
        if len(twins) > 1:
            for p in twins:
                a = max(p["audience"], key=p["audience"].get)
                if p["audience"][a] >= 0.6 and a != "Unrated":
                    p["name"] = (
                        f"{ {'Adults': 'Adult', 'Teens': 'Teen', 'Family': 'Family', 'Kids': 'Kids'}[a] } {name}"
                    )
    for name in {p["name"] for p in profiles}:
        twins = [p for p in profiles if p["name"] == name]
        if len(twins) > 1:
            for p in twins:
                p["name"] = f"{name} ({p['countries'][0]['name']})"
    for name in {p["name"] for p in profiles}:  # last resort so names are always unique
        twins = [p for p in profiles if p["name"] == name]
        for i, p in enumerate(twins[1:], 2):
            p["name"] = f"{name} {i}"


def describe(p):
    kind = (
        "movies"
        if p["movie_share"] >= 0.85
        else "TV shows" if p["movie_share"] <= 0.15 else "a mix of movies and shows"
    )
    top_aud = max((g for g in p["audience"] if g != "Unrated"), key=p["audience"].get)
    aud = (
        f"mostly rated for {top_aud.lower()} ({p['audience'][top_aud] * 100:.0f}%)"
        if p["audience"][top_aud] >= 0.5
        else "rated for a mix of audiences"
    )
    genres = ", ".join(g["name"] for g in p["genres"][:3])
    ctry = p["countries"][0]
    return (
        f"{p['share'] * 100:.0f}% of the catalogue. Mostly {kind}, {aud}. Typical genres: {genres}. "
        f"Most often from {ctry['name']} ({ctry['share'] * 100:.0f}%). Median release year {p['median_year']}."
    )


def main():
    t0 = time.time()
    df = load_data()
    frame, _ = rm.build_frame(df)
    X, genres_m, genre_names, audience = build_matrix(df, frame)
    pca = PCA(n_components=0.85, random_state=SEED).fit(X)
    Z = pca.transform(X)
    print(
        f"Step 1: {X.shape[1]} columns -> {Z.shape[1]} components keep {pca.explained_variance_ratio_.sum() * 100:.0f}% of the variance [{time.time() - t0:.0f}s]",
        flush=True,
    )

    # Step 2a: K-Means for many k
    runs, labels_by_k = [], {}
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        lab = relabel_by_size(km.labels_)
        labels_by_k[k] = lab
        smallest = float(np.bincount(lab).min() / len(lab))
        runs.append(
            {
                "k": k,
                "inertia": round(float(km.inertia_), 1),
                "silhouette": round(
                    float(silhouette_score(Z, lab, random_state=SEED)), 4
                ),
                "davies_bouldin": round(float(davies_bouldin_score(Z, lab)), 3),
                "smallest_share": round(smallest, 4),
            }
        )
        print(
            f"  k={k:2d} silhouette={runs[-1]['silhouette']:.3f} smallest cluster={smallest * 100:.1f}%",
            flush=True,
        )
    ok = [r for r in runs if 4 <= r["k"] <= 10 and r["smallest_share"] >= MIN_SHARE]
    best_k = max(ok, key=lambda r: r["silhouette"])["k"]
    print(f"Chosen k = {best_k} [{time.time() - t0:.0f}s]", flush=True)

    # Step 2b: three algorithms at the chosen k
    base = labels_by_k[best_k]
    seeds = [
        relabel_by_size(KMeans(best_k, n_init=1, random_state=s).fit_predict(Z))
        for s in range(5)
    ]
    stability = float(np.mean([adjusted_rand_score(base, s) for s in seeds]))
    algos = [
        {
            "name": "K-Means",
            **scores(Z, base),
            "agreement": 1.0,
            "smallest_share": round(float(np.bincount(base).min() / len(base)), 4),
        }
    ]
    for name, lab in (
        (
            "Agglomerative (Ward)",
            AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(Z),
        ),
        (
            "Gaussian Mixture",
            GaussianMixture(
                best_k, covariance_type="full", random_state=SEED, n_init=2
            ).fit_predict(Z),
        ),
    ):
        lab = relabel_by_size(lab)
        algos.append(
            {
                "name": name,
                **scores(Z, lab),
                "agreement": round(float(adjusted_rand_score(base, lab)), 3),
                "smallest_share": round(float(np.bincount(lab).min() / len(lab)), 4),
            }
        )
        print(f"  {name}: {algos[-1]}  [{time.time() - t0:.0f}s]", flush=True)

    # Step 3 and 5: profiles for every k the website can show
    profiles = {
        str(k): profile(df, genres_m, genre_names, audience, Z, labels_by_k[k])
        for k in K_SHOW
    }

    # Step 4: 2D map (t-SNE is for display only)
    emb = TSNE(
        n_components=2,
        perplexity=40,
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(Z)
    emb = (emb - emb.min(0)) / (emb.max(0) - emb.min(0))
    print(f"t-SNE done [{time.time() - t0:.0f}s]", flush=True)

    METRICS.write_text(
        json.dumps(
            {
                "rows": int(len(df)),
                "columns": int(X.shape[1]),
                "components": int(Z.shape[1]),
                "variance_kept": round(float(pca.explained_variance_ratio_.sum()), 3),
                "weights": WEIGHTS,
                "features": FEATURE_INFO,
                "runs": runs,
                "best_k": int(best_k),
                "min_share": MIN_SHARE,
                "algorithms": algos,
                "stability": round(stability, 3),
                "profiles": profiles,
            },
            indent=1,
        )
    )
    POINTS.write_text(
        json.dumps(
            {
                "x": [int(round(v * 1000)) for v in emb[:, 0]],
                "y": [int(round(v * 1000)) for v in emb[:, 1]],
                "labels": {str(k): [int(v) for v in labels_by_k[k]] for k in K_SHOW},
            },
            separators=(",", ":"),
        )
    )
    print(f"Saved. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
