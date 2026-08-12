"""
build_towers_json.py — full pipeline from raw CSVs to dashboard JSON.

Run in a fresh Colab notebook. Produces towers.json for the TelcoShield dashboard.

    !pip install xgboost rasterio -q

WHEN TO USE THIS
  Only if data/baseline.parquet is missing or incomplete. If it already has
  `static_priority` and `pop_served`, use the export cell instead — those are
  JY's numbers and retraining will produce slightly different ones.

WHAT IT DOES NOT DO
  It does not invent data. If WorldPop is missing it stops with an error
  rather than filling population with plausible-looking noise.
"""

import os, json
import numpy as np
import pandas as pd

BASE = "/content/drive/MyDrive/telcoshield_px"

# Same six features as train_susceptibility.py.
# lat/lon excluded: the model would memorise where the 2024 flood happened.
# landcover excluded: tree cover reached 80% of importance, which is the model
# learning that C-band SAR cannot see water under canopy, not learning hydrology.
FEATS = ["elev", "slope", "hand", "log_upa", "twi", "gsw_occ"]


def prep(df):
    df = df.drop(columns=[c for c in ["system:index", ".geo"] if c in df],
                 errors="ignore").copy()
    # upa is zero-inflated: >25% sit at the MERIT single-pixel floor.
    df["log_upa"] = np.log1p(df["upa"])
    return df.replace([np.inf, -np.inf], np.nan)


def make_model():
    """JY's exact hyperparameters."""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, min_child_weight=5,
        eval_metric="aucpr", n_jobs=-1, random_state=42,
    )


# ============================================================ A. train
def train():
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score, roc_auc_score

    tr = prep(pd.read_csv(f"{BASE}/samples_2024.csv")).dropna(subset=FEATS + ["flood", "block"])
    te = prep(pd.read_csv(f"{BASE}/samples_2022.csv")).dropna(subset=FEATS + ["flood"])
    print(f"train {len(tr)} rows / {tr.block.nunique()} blocks | test {len(te)} rows")

    # --- spatial CV. Random splits leak: neighbouring pixels are near-identical,
    #     so the same sample effectively lands in train and test.
    X, y, g = tr[FEATS].values, tr.flood.values, tr.block.values
    sc = []
    for a, b in GroupKFold(n_splits=5).split(X, y, g):
        if len(np.unique(y[b])) < 2:
            continue
        sc.append(average_precision_score(
            y[b], make_model().fit(X[a], y[a]).predict_proba(X[b])[:, 1]))
    print(f"[1] SPATIAL CV   PR-AUC {np.mean(sc):.3f} +/- {np.std(sc):.3f} "
          f"(base {y.mean():.3f})   <- USE THIS NUMBER IN THE PITCH")

    # --- permutation test: shuffled labels must collapse to the base rate
    ys = np.random.default_rng(0).permutation(y)
    ps = []
    for a, b in GroupKFold(n_splits=5).split(X, ys, g):
        if len(np.unique(ys[b])) < 2:
            continue
        ps.append(average_precision_score(
            ys[b], make_model().fit(X[a], ys[a]).predict_proba(X[b])[:, 1]))
    print(f"[2] PERMUTED     PR-AUC {np.mean(ps):.3f} (must sit at {y.mean():.3f})")
    if np.mean(ps) > y.mean() * 1.5:
        print("    ** LEAK: shuffled labels still score. Stop and investigate. **")

    model = make_model().fit(tr[FEATS], tr.flood)

    # --- out-of-event holdout, WITH the contamination measured
    p = model.predict_proba(te[FEATS])[:, 1]
    ap = average_precision_score(te.flood, p)
    print(f"[3] 2024 -> 2022 PR-AUC {ap:.3f}  ROC {roc_auc_score(te.flood, p):.3f}")

    ktr = set(zip(tr.lat.round(6), tr.lon.round(6)))
    kte = list(zip(te.lat.round(6), te.lon.round(6)))
    shared = np.array([k in ktr for k in kte])
    print(f"    !! {shared.sum()}/{len(te)} test points ({shared.mean():.0%}) share "
          f"coordinates with training data.")
    print(f"    This holdout is NOT independent. Quote the spatial CV instead.")

    top = p >= np.quantile(p, 0.90)
    print(f"[4] top-decile precision {te.flood[top].mean():.1%} "
          f"-- on a BALANCED sample (base {te.flood.mean():.0%}), say so aloud")

    imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
    print(f"[5] importance {imp.round(3).to_dict()}")
    return model


# ============================================================ B. score towers
def score(model):
    tw = prep(pd.read_csv(f"{BASE}/tower_terrain.csv"))
    ok = tw[FEATS].notna().all(axis=1)
    tw.loc[ok, "baseline"] = model.predict_proba(tw.loc[ok, FEATS])[:, 1]
    tw["baseline"] = tw.baseline.fillna(tw.baseline.median())

    q = tw.baseline.quantile([.1, .5, .9])
    print(f"\n[6] {len(tw)} towers scored, {(~ok).sum()} imputed")
    print(f"    baseline p10={q[.1]:.3f} p50={q[.5]:.3f} p90={q[.9]:.3f}")
    if q[.9] - q[.1] < 0.15:
        print("    ** SATURATED: no spread to rank on. **")
    print(f"    {(tw.baseline >= 0.25).sum()} towers above the 0.25 escalation floor")
    return tw


# ============================================================ C. population
def population(tw):
    """Real WorldPop counts. Fails loudly if the raster is absent."""
    import rasterio
    from rasterio.windows import Window, from_bounds

    tif = f"{BASE}/data/worldpop_mys.tif"
    if not os.path.exists(tif):
        raise FileNotFoundError(
            f"{tif} not found.\n"
            "Download it first:\n"
            "  !wget -O {tif} https://data.worldpop.org/GIS/Population/"
            "Global_2000_2020_Constrained/2020/BSGM/MYS/"
            "mys_ppp_2020_UNadj_constrained.tif\n"
            "Do NOT substitute synthetic population — the ranking depends on it.")

    KM_PER_DEG = 111.0
    # Coverage radius from local tower spacing: operators site towers at
    # intervals reflecting intended coverage. 0.75 leaves overlap for handover.
    tw["radius_km"] = np.clip(tw.nn_dist_km * 0.75, 1.0, 6.0)

    def pixels(src, lat, lon, r_km):
        deg = r_km / KM_PER_DEG
        w = from_bounds(lon - deg, lat - deg, lon + deg, lat + deg,
                        src.transform).round_offsets().round_lengths()
        w = w.intersection(Window(0, 0, src.width, src.height))
        if w.width <= 0 or w.height <= 0:
            return np.array([], int), np.array([])
        a = src.read(1, window=w).astype("float64"); a[a < 0] = 0
        t = src.window_transform(w)
        rows, cols = np.indices(a.shape)
        xs = t.c + (cols + .5) * t.a
        ys = t.f + (rows + .5) * t.e
        m = (xs - lon) ** 2 + (ys - lat) ** 2 <= deg ** 2
        flat = ((rows[m] + int(w.row_off)) * src.width) + (cols[m] + int(w.col_off))
        return flat, a[m]

    src = rasterio.open(tif)
    fp = {r.tower_id: pixels(src, r.lat, r.lon, r.radius_km) for r in tw.itertuples()}

    # How many towers cover each pixel. Without this one resident inside eight
    # coverage circles is counted eight times.
    allpx = np.concatenate([f[0] for f in fp.values()])
    uq, ct = np.unique(allpx, return_counts=True)
    cover = dict(zip(uq.tolist(), ct.tolist()))

    rows = []
    for r in tw.itertuples():
        f, p = fp[r.tower_id]
        if len(f) == 0 or p.sum() == 0:
            rows.append(dict(tower_id=r.tower_id, pop_served=0., pop_sole=0., isolation=0.))
            continue
        n = np.array([cover[x] for x in f])
        rows.append(dict(
            tower_id=r.tower_id,
            pop_served=float((p / n).sum()),    # shared, no double count
            pop_sole=float(p[n == 1].sum()),    # nobody else covers them
            isolation=float(p[n == 1].sum() / p.sum()),
        ))

    tw = tw.drop(columns=[c for c in ["pop_served", "pop_sole", "isolation"]
                          if c in tw], errors="ignore")
    tw = tw.merge(pd.DataFrame(rows), on="tower_id")
    print(f"\n[7] total pop_served {tw.pop_served.sum():,.0f} (Kelantan ~1.9M)")
    print(f"    isolation p50={tw.isolation.median():.2f} p90={tw.isolation.quantile(.9):.2f}")
    return tw


# ============================================================ D. priority
def priority(tw):
    def rank_norm(s):
        # Rank-based, not min-max: population is long-tailed and min-max would
        # put 95% of towers in one bin.
        return s.rank(pct=True, na_option="keep").fillna(0.5)

    tw["pop_weight"] = rank_norm(np.log1p(tw.pop_served))
    tw["iso_weight"] = rank_norm(tw.isolation)
    # Multiplicative: a tower on high ground stays near zero however many
    # people live around it.
    tw["static_priority"] = tw.baseline * (0.7 * tw.pop_weight + 0.3 * tw.iso_weight)

    tw = tw.sort_values("static_priority", ascending=False).reset_index(drop=True)
    tw["rank"] = range(1, len(tw) + 1)
    print("\n[8] top 5 by priority:")
    print(tw.head(5)[["tower_id", "baseline", "pop_served", "isolation",
                      "static_priority"]].round(3).to_string(index=False))
    return tw


# ============================================================ E. export
def export(tw, path="towers.json"):
    cols = ["tower_id", "lat", "lon", "baseline", "static_priority", "pop_served",
            "pop_sole", "isolation", "n_within_5km", "nn_dist_km", "radius_km",
            "elev", "hand", "twi", "slope", "operators", "rank",
            "flooded_2024", "flooded_2022"]
    missing = [c for c in cols if c not in tw]
    if missing:
        print(f"!! missing, zero-filled: {missing}")
    for c in missing:
        tw[c] = 0

    # pandas NaN serialises as a bare `NaN`, which is not valid JSON and the
    # browser refuses to parse it. allow_nan=False makes it fail loudly.
    out = tw[cols].round(4).replace([np.inf, -np.inf], np.nan).fillna(0)
    json.dump(out.to_dict("records"), open(path, "w"), allow_nan=False)
    print(f"\n[9] wrote {path}: {len(out)} towers")
    return out


if __name__ == "__main__":
    tw = priority(population(score(train())))
    export(tw)
    tw.to_parquet(f"{BASE}/data/baseline_rebuilt.parquet", index=False)
    print(f"    also saved -> {BASE}/data/baseline_rebuilt.parquet")
    try:
        from google.colab import files
        files.download("towers.json")
    except ImportError:
        pass
