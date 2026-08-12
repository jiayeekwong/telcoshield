"""
scripts/export_towers.py — turn current_ranking.parquet into docs/towers.json.

Run by .github/workflows/live.yml after step6_live.py, or by hand:
    TELCOSHIELD_BASE=. python scripts/export_towers.py
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

BASE = os.environ.get("TELCOSHIELD_BASE", ".")
OUT = f"{BASE}/docs/towers.json"

src = f"{BASE}/data/current_ranking.parquet"
if not os.path.exists(src):
    src = f"{BASE}/data/baseline.parquet"        # fall back to the static layer
    print("current_ranking missing — exporting static baseline")
tw = pd.read_parquet(src)

# Terrain lives only in the CSV: train_susceptibility dropped these columns.
terr = pd.read_csv(f"{BASE}/tower_terrain.csv")[
    ["tower_id", "elev", "hand", "twi", "slope", "n_within_5km",
     "nn_dist_km", "operators", "flooded_2024", "flooded_2022"]]
tw = tw.merge(terr, on="tower_id", how="left")

# Coverage radius from local spacing; 0.75 leaves overlap for handover.
tw["radius_km"] = np.clip(tw.nn_dist_km * 0.75, 1.0, 6.0)

sort_on = "live_score" if "live_score" in tw else "static_priority"
tw = tw.sort_values(sort_on, ascending=False).reset_index(drop=True)
tw["rank"] = range(1, len(tw) + 1)

# Read by the dashboard to show snapshot age in the LIVE badge.
tw["generated_at"] = datetime.now(timezone.utc).isoformat()

cols = ["tower_id", "lat", "lon", "baseline", "static_priority", "live_score",
        "multiplier", "trigger", "rain_mm", "rain_pctl", "river_stress", "gauge",
        "escalated", "pop_served", "pop_sole", "isolation", "n_within_5km",
        "nn_dist_km", "radius_km", "elev", "hand", "twi", "slope", "operators",
        "rank", "flooded_2024", "flooded_2022", "generated_at"]
missing = [c for c in cols if c not in tw]
if missing:
    print(f"absent, set to null: {missing}")
for c in missing:
    tw[c] = None

out = tw[cols].copy()
num = out.select_dtypes("number").columns
out[num] = out[num].round(4).replace([np.inf, -np.inf], np.nan)

# NaN must become null, not 0. river_stress = 0 reads as "river is calm";
# null reads as "no gauge signal", which is what it actually means.
out = out.astype(object).where(pd.notna(out), None)

os.makedirs(f"{BASE}/docs", exist_ok=True)
json.dump(out.to_dict("records"), open(OUT, "w"), allow_nan=False)

esc = int(pd.to_numeric(tw.get("escalated"), errors="coerce").fillna(0).sum())
print(f"wrote {OUT}: {len(out)} towers, ranked by {sort_on}, {esc} escalated")
