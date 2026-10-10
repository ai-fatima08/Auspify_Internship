import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.preprocessing import normalize
except ImportError:
    import scipy.sparse as sp

    def normalize(m):
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        return (sp.diags(1 / norms) @ m).tocsr()

    class CountVectorizer:
        def __init__(self, tokenizer, **_):
            self.tokenizer = tokenizer

        def fit_transform(self, docs):
            vocab, rows, cols = {}, [], []
            for r, doc in enumerate(docs):
                for tok in set(self.tokenizer(doc)):
                    rows.append(r)
                    cols.append(vocab.setdefault(tok, len(vocab)))
            return sp.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab))
            )

    class TfidfTransformer:
        def fit_transform(self, m):
            df = np.asarray((m > 0).sum(axis=0)).ravel()
            idf = np.log((1 + m.shape[0]) / (1 + df)) + 1
            return normalize((m @ sp.diags(idf)).tocsr())


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Dataset.xlsx"
ARTIFACT = ROOT / "ml" / "model.pkl"
METRICS = ROOT / "ml" / "metrics.json"

# Relative importance of each feature block. Genres matter most.
WEIGHTS = {
    "genre": 5.0,
    "type": 1.0,
    "director": 1.2,
    "country": 1.0,
    "rating": 0.5,
    "era": 0.4,
    "length": 0.4,
}


def _fix_text(s: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 (e.g. 'RaÃºl')."""
    try:
        return s.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _list_tokens(series: pd.Series) -> pd.Series:
    """'A, B' -> 'A|B' with spaces removed so each item is one token."""
    return series.apply(
        lambda v: "|".join(
            x.strip().replace(" ", "_") for x in v.split(",") if x.strip()
        )
    )


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA)
    df = df.drop_duplicates(subset=["title", "type", "release_year"]).reset_index(
        drop=True
    )
    for col in ("director", "country", "rating", "listed_in", "duration", "title"):
        df[col] = df[col].fillna("").astype(str).apply(_fix_text)
    for col in ("director", "country"):
        df[col] = df[col].replace({"Not Given": ""})
    return df


def build_features(df: pd.DataFrame):
    split = lambda s: s.split("|") if s else []
    blocks = {}

    def multi(name, series, tfidf=False):
        vec = CountVectorizer(
            tokenizer=split, token_pattern=None, lowercase=False, binary=True
        )
        m = vec.fit_transform(_list_tokens(series))
        if tfidf:  # rare directors/countries are more informative than common ones
            m = TfidfTransformer().fit_transform(m)
        blocks[name] = m

    multi("genre", df["listed_in"])
    multi("director", df["director"], tfidf=True)
    multi("country", df["country"], tfidf=True)

    def cat(name, series):
        vec = CountVectorizer(
            tokenizer=split, token_pattern=None, lowercase=False, binary=True
        )
        blocks[name] = vec.fit_transform(series.astype(str))

    cat("type", df["type"])
    cat("rating", df["rating"])
    cat("era", (df["release_year"] // 10 * 10).astype(str))

    num = pd.to_numeric(
        df["duration"].str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(0)
    movie_bucket = pd.cut(num, [-1, 0, 80, 100, 120, 1000], labels=False).astype(str)
    show_bucket = "S" + num.clip(upper=5).astype(int).astype(str)
    cat(
        "length",
        pd.Series(
            np.where(df["type"] == "Movie", movie_bucket, show_bucket), index=df.index
        ),
    )

    parts = [normalize(blocks[k]) * WEIGHTS[k] for k in WEIGHTS]
    return normalize(hstack(parts).tocsr())


def recommend(matrix, df, idx, k=10, same_type=False):
    """Return top-k (row_index, similarity) for the title at row `idx`."""
    sims = (matrix @ matrix[idx].T).toarray().ravel()
    sims[idx] = -1
    if same_type:
        sims[(df["type"] != df.iloc[idx]["type"]).to_numpy()] = -1
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def recommend_profile(matrix, df, ids, k=12):
    """Recommend from several liked titles: average their vectors into one taste profile."""
    profile = normalize(csr_matrix(matrix[ids].mean(axis=0)))
    sims = (matrix @ profile.T).toarray().ravel()
    sims[list(ids)] = -1
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def evaluate(matrix, df, k=10, n=1500, seed=42):
    """Offline quality check using the metadata we have as ground truth."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(df), size=n, replace=False)
    genres = df["listed_in"].apply(lambda v: set(x.strip() for x in v.split(",")))
    p_any, base, jacc, same_type, div, seen = [], [], [], [], [], set()
    for i in sample:
        ids = [r[0] for r in recommend(matrix, df, i, k)]
        seen.update(ids)
        g = genres.iloc[i]
        p_any.append(np.mean([len(g & genres.iloc[j]) > 0 for j in ids]))
        rand = rng.choice(len(df), size=k, replace=False)
        base.append(np.mean([len(g & genres.iloc[j]) > 0 for j in rand]))
        jacc.append(
            np.mean([len(g & genres.iloc[j]) / len(g | genres.iloc[j]) for j in ids])
        )
        same_type.append(
            np.mean([df.iloc[j]["type"] == df.iloc[i]["type"] for j in ids])
        )
        sub = matrix[ids]
        pair = (sub @ sub.T).toarray()
        div.append(1 - (pair.sum() - k) / (k * (k - 1)))
    return {
        "k": k,
        "titles_evaluated": int(n),
        "catalogue_size": int(len(df)),
        "genre_precision_at_k": round(float(np.mean(p_any)), 4),
        "random_baseline_precision": round(float(np.mean(base)), 4),
        "mean_genre_overlap": round(float(np.mean(jacc)), 4),
        "type_consistency": round(float(np.mean(same_type)), 4),
        "diversity": round(float(np.mean(div)), 4),
        "catalogue_coverage": round(len(seen) / len(df), 4),
    }


def main():
    df = load_data()
    matrix = build_features(df)
    print(f"Feature matrix: {matrix.shape}")
    metrics = evaluate(matrix, df)
    print(json.dumps(metrics, indent=2))
    with open(ARTIFACT, "wb") as f:
        pickle.dump({"matrix": matrix, "df": df}, f)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {ARTIFACT.name} and {METRICS.name}")


if __name__ == "__main__":
    main()
