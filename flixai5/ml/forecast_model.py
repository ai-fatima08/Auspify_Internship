import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recommender import load_data

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "ml" / "forecast_metrics.json"
SEED = 42
START = "2016-01"  # before 2016 Netflix added only a handful of titles a month, which is a different regime
HOLDOUT = 12
HORIZON = 12
BT_MIN, BT_H, BT_STEP = 30, 6, 3

FEATURE_NAMES = [
    "time_index",
    "month_sin",
    "month_cos",
    "quarter",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_12",
]
FEATURE_TEXT = [
    (
        "Calendar",
        "Month as a sine and cosine pair (so December sits next to January), quarter, and a running time index.",
        ["month_sin", "month_cos", "quarter", "time_index"],
    ),
    (
        "Lags",
        "What was added 1, 2, 3 and 12 months earlier. The 12-month lag captures the yearly cycle.",
        ["lag_1", "lag_2", "lag_3", "lag_12"],
    ),
    (
        "Rolling averages",
        "The average of the previous 3 and 12 months, which smooths out noise and shows the recent level.",
        ["rolling_mean_3", "rolling_mean_12"],
    ),
]


def make_row(h, t):
    """Features for predicting month t from the values before it. Only the past is used, so nothing leaks."""
    m = t % 12
    return [
        t,
        np.sin(2 * np.pi * m / 12),
        np.cos(2 * np.pi * m / 12),
        m // 3,
        h[t - 1],
        h[t - 2],
        h[t - 3],
        h[t - 12],
        float(np.mean(h[t - 3 : t])),
        float(np.mean(h[t - 12 : t])),
    ]


#  models: each takes (history, horizon)
def m_naive(tr, h):
    return np.repeat(tr[-1], h).astype(float)


def m_snaive(tr, h):
    return np.array([tr[len(tr) - 12 + (i % 12)] for i in range(h)], float)


def m_moving_average(tr, h):
    return np.repeat(np.mean(tr[-6:]), h)


def m_trend_season(tr, h):
    """Straight-line trend plus a month effect, fitted on the last 3 years only."""
    n, w = len(tr), min(36, len(tr))

    def design(idx):
        months = np.eye(12)[np.asarray(idx) % 12]
        return np.column_stack([np.asarray(idx) - (n - w), months])

    idx = np.arange(n - w, n)
    model = Ridge(alpha=0.5).fit(design(idx), tr[n - w :])
    return np.maximum(model.predict(design(np.arange(n, n + h))), 0)


def m_holt_winters(tr, h, m=12):
    """Additive Holt-Winters with a damped trend (the classic exponential smoothing model), fitted by least squares."""
    y = np.asarray(tr, float)
    n = len(y)
    l0, b0, s0 = (
        y[:m].mean(),
        (y[m : 2 * m].mean() - y[:m].mean()) / m,
        list(y[:m] - y[:m].mean()),
    )

    def run(p):
        a, bs, gs, phi = p
        l, b, s, sse = l0, b0, list(s0), 0.0
        for t in range(m, n):
            e = y[t] - (l + phi * b + s[t - m])
            sse += e * e
            l, b = l + phi * b + a * e, phi * b + a * bs * e
            s.append(s[t - m] + (1 - a) * gs * e)
        return sse, l, b, s, phi

    best = min(
        (
            minimize(
                lambda p: run(p)[0],
                x0,
                bounds=[(0.01, 0.99), (0, 1), (0, 1), (0.8, 0.98)],
                method="L-BFGS-B",
            )
            for x0 in (
                [0.3, 0.1, 0.2, 0.9],
                [0.1, 0.05, 0.5, 0.95],
                [0.6, 0.2, 0.1, 0.85],
            )
        ),
        key=lambda r: r.fun,
    )
    _, l, b, s, phi = run(best.x)
    out = [
        l + sum(phi**i for i in range(1, k + 1)) * b + s[len(s) - m + ((k - 1) % m)]
        for k in range(1, h + 1)
    ]
    return np.maximum(np.array(out), 0)


def ml_model(factory):
    def f(tr, h):
        tr = np.asarray(tr, float)
        X = np.array([make_row(tr, t) for t in range(12, len(tr))])
        model = factory().fit(X, tr[12:])
        hist, out = list(tr), []
        for _ in range(
            h
        ):  # recursive forecasting: each prediction becomes a lag for the next
            p = max(float(model.predict([make_row(np.array(hist), len(hist))])[0]), 0.0)
            hist.append(p)
            out.append(p)
        return np.array(out)

    return f


m_ridge = ml_model(lambda: make_pipeline(StandardScaler(), Ridge(alpha=3.0)))
m_forest = ml_model(
    lambda: RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=SEED
    )
)
m_boost = ml_model(
    lambda: GradientBoostingRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED,
    )
)


def m_ensemble(tr, h):
    return np.mean(
        [
            m_trend_season(tr, h),
            m_holt_winters(tr, h),
            m_ridge(tr, h),
            m_forest(tr, h),
            m_boost(tr, h),
        ],
        axis=0,
    )


MODELS = [
    ("Naive (repeat last month)", "Baseline", m_naive),
    ("Seasonal naive (same month last year)", "Baseline", m_snaive),
    ("Moving average (6 months)", "Baseline", m_moving_average),
    ("Linear trend + month effects", "Statistical", m_trend_season),
    ("Holt-Winters (damped trend)", "Statistical", m_holt_winters),
    ("Ridge on lag features", "Machine learning", m_ridge),
    ("Random Forest on lag features", "Machine learning", m_forest),
    ("Gradient Boosting on lag features", "Machine learning", m_boost),
    ("Ensemble (average of 5 models)", "Ensemble", m_ensemble),
]


#  evaluation helpers
def scores(actual, pred, train):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    scale = np.mean(np.abs(train[12:] - train[:-12]))
    err = pred - actual
    return {
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
        "smape": round(
            float(
                100 * np.mean(2 * np.abs(err) / (np.abs(actual) + np.abs(pred) + 1e-9))
            ),
            2,
        ),
        "mase": round(float(np.mean(np.abs(err)) / scale), 3),
        "bias": round(float(np.mean(err)), 2),
    }


def backtest(fn, y, last, horizon, step, first=BT_MIN):
    """Rolling origin: train on y[:t], forecast the next `horizon` months, move forward, repeat."""
    errors = []
    for t in range(first, last - horizon + 1, step):
        errors.append(np.asarray(y[t : t + horizon], float) - fn(y[:t], horizon))
    return np.array(errors)


def seasonal_index(y):
    trend = pd.Series(y).rolling(12, center=True).mean()
    trend = (trend + trend.shift(-1)) / 2  # centred 12-month moving average
    ratio = pd.Series(y) / trend
    idx = np.array([np.nanmean(ratio.values[m::12]) for m in range(12)])
    return idx / idx.mean(), trend


def acf(y, lags=24):
    y = np.asarray(y, float) - np.mean(y)
    return [float(np.sum(y[k:] * y[:-k]) / np.sum(y * y)) for k in range(1, lags + 1)]


def main():
    t0 = time.time()
    df = load_data()
    d = pd.to_datetime(df["date_added"])
    last_day = d.max()
    last_full = last_day.to_period("M") - (
        1 if last_day.day < last_day.days_in_month else 0
    )
    months = pd.period_range(START, last_full, freq="M")
    per = d.dt.to_period("M")
    wide = pd.crosstab(per, df["type"]).reindex(months, fill_value=0)
    series = {
        "All titles": wide.sum(axis=1).values.astype(float),
        "Movies": wide["Movie"].values.astype(float),
        "TV Shows": wide["TV Show"].values.astype(float),
    }
    n = len(months)
    future_months = pd.period_range(last_full + 1, periods=HORIZON, freq="M")
    pm = per == last_full + 1
    partial = (
        {
            "month": str(last_full + 1),
            "days": int(last_day.day),
            "days_in_month": int(last_day.days_in_month),
            "counts": {
                "All titles": int(pm.sum()),
                "Movies": int((pm & (df["type"] == "Movie")).sum()),
                "TV Shows": int((pm & (df["type"] == "TV Show")).sum()),
            },
        }
        if last_day.day < last_day.days_in_month
        else None
    )
    print(
        f"{n} complete months {months[0]} to {months[-1]}. Partial month: {partial}",
        flush=True,
    )

    out_series = {}
    for name, y in series.items():
        train, test = y[: n - HOLDOUT], y[n - HOLDOUT :]
        rows = []
        for mname, kind, fn in MODELS:
            bt = backtest(fn, train, len(train), BT_H, BT_STEP)
            pred = fn(train, HOLDOUT)
            rows.append(
                {
                    "name": mname,
                    "kind": kind,
                    "backtest_mae": round(float(np.mean(np.abs(bt))), 2),
                    "holdout": scores(test, pred, train),
                    "holdout_pred": [round(float(v), 1) for v in pred],
                }
            )
        best = min(rows, key=lambda r: r["backtest_mae"])
        best_fn = next(fn for mname, _, fn in MODELS if mname == best["name"])
        # the 80% range comes from this model's errors over many past forecasts of up to 12 months
        errs = backtest(best_fn, y, n, HORIZON, 2)
        q10, q90 = np.quantile(errs, [0.10, 0.90])
        mean = best_fn(y, HORIZON)
        last12 = float(y[-12:].sum())
        idx, trend = seasonal_index(y)
        base = next(r for r in rows if r["name"].startswith("Seasonal naive"))
        out_series[name] = {
            "actual": [int(v) for v in y],
            "trend": [None if np.isnan(v) else round(float(v), 1) for v in trend],
            "seasonal_index": [round(float(v), 3) for v in idx],
            "models": rows,
            "best": best["name"],
            "forecast": {
                "mean": [round(float(v), 1) for v in mean],
                "lo": [round(float(max(m + q10, 0)), 1) for m in mean],
                "hi": [round(float(m + q90), 1) for m in mean],
                "interval_errors": int(errs.size),
            },
            "next12": round(float(mean.sum()), 0),
            "last12": last12,
            "change_pct": round(float((mean.sum() / last12 - 1) * 100), 1),
            "best_vs_snaive_pct": round(
                float((1 - best["holdout"]["mae"] / base["holdout"]["mae"]) * 100), 1
            ),
        }
        print(
            f"{name:11s} best={best['name']} backtest MAE={best['backtest_mae']} holdout MAE={best['holdout']['mae']} next12={mean.sum():.0f} vs last12={last12:.0f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )
        for r in rows:
            print(
                f"    {r['name']:40s} backtest {r['backtest_mae']:6.1f}  holdout MAE {r['holdout']['mae']:6.1f} sMAPE {r['holdout']['smape']:5.1f}",
                flush=True,
            )

    # Step 2: history of the catalogue
    year = d.dt.year
    yearly = []
    for yr in range(2015, int(year.max()) + 1):
        m = year == yr
        yearly.append(
            {
                "year": yr,
                "all": int(m.sum()),
                "movies": int((m & (df["type"] == "Movie")).sum()),
                "shows": int((m & (df["type"] == "TV Show")).sum()),
                "partial": bool(yr == year.max() and partial is not None),
            }
        )
    age = []
    for yr in range(2016, int(year.max()) + 1):
        m = year == yr
        gap = yr - df.loc[m, "release_year"]
        age.append(
            {
                "year": yr,
                "fresh_share": round(float((gap <= 1).mean()), 3),
                "median_age": float(gap.median()),
                "titles": int(m.sum()),
            }
        )
    allv = series["All titles"]
    feats = np.array([make_row(allv, t) for t in range(n - 6, n)])
    sample = [
        dict(
            month=str(months[n - 6 + i]),
            **{k: round(float(v), 2) for k, v in zip(FEATURE_NAMES, feats[i])},
            actual=int(allv[n - 6 + i]),
        )
        for i in range(6)
    ]

    METRICS.write_text(
        json.dumps(
            {
                "months": [str(m) for m in months],
                "future_months": [str(m) for m in future_months],
                "holdout": HOLDOUT,
                "partial": partial,
                "backtest": {
                    "min_train": BT_MIN,
                    "horizon": BT_H,
                    "step": BT_STEP,
                    "origins": len(range(BT_MIN, n - HOLDOUT - BT_H + 1, BT_STEP)),
                },
                "series": out_series,
                "yearly": yearly,
                "freshness": age,
                "acf": {
                    "values": [round(v, 3) for v in acf(allv)],
                    "bound": round(1.96 / np.sqrt(n), 3),
                },
                "features": {
                    "groups": [
                        {"name": a, "text": b, "cols": c} for a, b, c in FEATURE_TEXT
                    ],
                    "columns": FEATURE_NAMES,
                    "sample": sample,
                    "training_rows": n - HOLDOUT - 12,
                },
            },
            indent=1,
        )
    )
    print(f"Saved. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
