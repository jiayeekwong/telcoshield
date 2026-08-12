"""
backend/main.py — TelcoShield API

    pip install fastapi uvicorn pandas pyarrow numpy requests
    uvicorn main:app --reload --port 8000
    open http://localhost:8000

WHAT MAKES THIS DYNAMIC, AND WHAT DOESN'T
  GET  /api/towers   reads current_ranking.parquet. As stale as a JSON file.
  POST /api/refresh  RERUNS the engine: live Open-Meteo rainfall, latest river
                     levels, hysteresis gate, new live_score. This is the only
                     endpoint that produces new data.

  Wrapping a file in an API does not make it fresh. Recomputing does.

ONE ENGINE, NOT TWO
  refresh() imports step6_live.main() rather than reimplementing the scoring.
  A second implementation in JS or here would drift, and then the notebook and
  the dashboard disagree with no way to say which is right.

LAYOUT
  backend/main.py
  backend/step6_live.py          copied from the notebook, BASE from env
  backend/data/                  baseline, climatology, station map, river log
  frontend/index.html            the dashboard
"""

import os, sys, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
BASE = os.environ.get("TELCOSHIELD_BASE", str(HERE))
FRONTEND = HERE.parent / "frontend"
os.environ["TELCOSHIELD_BASE"] = BASE

app = FastAPI(title="TelcoShield")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

COLS = ["tower_id", "lat", "lon", "baseline", "static_priority", "live_score",
        "multiplier", "trigger", "rain_mm", "rain_pctl", "river_stress", "gauge",
        "escalated", "pop_served", "pop_sole", "isolation", "n_within_5km",
        "nn_dist_km", "radius_km", "elev", "hand", "twi", "slope", "operators",
        "rank", "flooded_2024", "flooded_2022", "generated_at"]


def build_records():
    """current_ranking + terrain -> the exact shape index.html expects."""
    src = Path(BASE) / "data" / "current_ranking.parquet"
    if not src.exists():
        src = Path(BASE) / "data" / "baseline.parquet"
    if not src.exists():
        raise HTTPException(503, f"no ranking file under {BASE}/data")

    tw = pd.read_parquet(src)

    # terrain lives only in the CSV: train_susceptibility dropped these columns
    terr_p = Path(BASE) / "tower_terrain.csv"
    if terr_p.exists():
        terr = pd.read_csv(terr_p)[["tower_id", "elev", "hand", "twi", "slope",
                                    "n_within_5km", "nn_dist_km", "operators",
                                    "flooded_2024", "flooded_2022"]]
        tw = tw.merge(terr, on="tower_id", how="left", suffixes=("", "_t"))

    if "nn_dist_km" in tw:
        tw["radius_km"] = np.clip(tw.nn_dist_km * 0.75, 1.0, 6.0)

    sort_on = "live_score" if "live_score" in tw else "static_priority"
    tw = tw.sort_values(sort_on, ascending=False).reset_index(drop=True)
    tw["rank"] = range(1, len(tw) + 1)

    # mtime, not now(): this is when the data was COMPUTED, not when it was read
    tw["generated_at"] = datetime.fromtimestamp(
        src.stat().st_mtime, timezone.utc).isoformat()

    for c in COLS:
        if c not in tw:
            tw[c] = None

    out = tw[COLS].copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4).replace([np.inf, -np.inf], np.nan)

    # NaN -> null, never 0. river_stress 0 reads as "river calm";
    # null reads as "no gauge signal", which is what it means.
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict("records")


@app.get("/api/towers")
def towers():
    """Current ranking. Age comes from the file's mtime, so the dashboard can
    show honestly how old it is."""
    return build_records()


@app.get("/api/status")
def status():
    src = Path(BASE) / "data" / "current_ranking.parquet"
    if not src.exists():
        return {"ready": False, "base": BASE}
    tw = pd.read_parquet(src)
    ts = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc)
    return {
        "ready": True,
        "towers": len(tw),
        "escalated": int(tw.escalated.sum()) if "escalated" in tw else 0,
        "generated_at": ts.isoformat(),
        "age_minutes": round((datetime.now(timezone.utc) - ts).total_seconds() / 60),
    }


RIVER_URL = os.environ.get(
    "RIVER_LOG_URL",
    "https://raw.githubusercontent.com/jiayeekwong/telcoshield/main/data/river_log.csv")


def pull_river_log():
    """Refresh the river log. Two independent paths, so one broken route does
    not leave the engine scoring against stale gauge readings.

    1. GitHub raw — the scraper commits there hourly, full history.
    2. Direct DID scrape — used when GitHub is unreachable or the repo is
       private. Fewer rows, but current readings beat stale ones.

    Prints the outcome so a silent fallback to old data is impossible to miss."""
    dst = Path(BASE) / "data" / "river_log.csv"

    try:
        import requests
        r = requests.get(RIVER_URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise ValueError(f"suspiciously small ({len(r.content)} bytes)")
        prev = dst.stat().st_size if dst.exists() else 0
        dst.write_bytes(r.content)
        print(f"river_log: GITHUB ok — {len(r.content)} bytes (+{len(r.content)-prev})")
        return {"river_log": "github", "bytes": len(r.content),
                "grew_by": len(r.content) - prev}
    except Exception as e:
        github_err = type(e).__name__
        print(f"river_log: github FAILED ({github_err}) — trying direct scrape")

    try:
        sys.path.insert(0, str(HERE.parent / "scripts"))
        import scrape_river
        scrape_river.LOG = dst              # write where step6_live reads
        scrape_river.main()
        print("river_log: SCRAPED DID directly")
        return {"river_log": "scraped direct", "github_error": github_err}
    except Exception as e2:
        print(f"river_log: STALE — both failed ({github_err} / {type(e2).__name__})")
        return {"river_log": "STALE — kept local copy",
                "github_error": github_err,
                "scrape_error": type(e2).__name__}

@app.post("/api/refresh")
def refresh():
    """Rerun the engine against live conditions. ~30-90 s: it calls Open-Meteo
    for every distinct 0.1 deg grid cell, unless the 6 h cache is still warm."""
    river = pull_river_log()

    sys.path.insert(0, str(HERE))
    try:
        import step6_live
        import importlib
        importlib.reload(step6_live)          # pick up edits without restarting
        step6_live.main()
    except ImportError:
        raise HTTPException(500, "step6_live.py not found in backend/")
    except NotImplementedError:
        raise HTTPException(500, "step6_live.py is still the stub — paste the "
                                 "Step 6 cell from the notebook into main()")
    except Exception as e:
        raise HTTPException(500, f"engine failed: {type(e).__name__}: {e}")

    return {"ok": True, **river, **status()}

# serve the dashboard from the same origin, so no CORS issues in the browser
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
else:
    print(f"note: {FRONTEND} not found — API only")
