# TelcoShield

Flood vulnerability prioritisation for telecommunications infrastructure.
716 towers, Kelantan. ASEAN GeoAI Fusion 2026.

```
live_score = static_priority × multiplier
             ↑ terrain +      ↑ live rainfall
               population +     + river levels
               redundancy
```

---

## Which setup do you need?

**Just want to see it** → Path A, two minutes, no Python.
**Want the Refresh button working** → Path B, thirty minutes.

Path A is enough for a complete submission. Path B adds a live moment to the pitch.

---

## Path A — static dashboard

```bash
cd frontend
python -m http.server 8000
```

Open <http://localhost:8000>.

Or in VS Code: install **Live Server**, right-click `index.html` → Open with Live Server.

Double-clicking `index.html` will **not** work — `fetch()` is blocked on `file://`
and you get an empty map. It has to be served over http.

### Deploy it

```bash
mkdir -p docs && cp frontend/index.html frontend/*.json docs/
git add docs && git commit -m "dashboard" && git push
```

GitHub → Settings → Pages → source `main`, folder `/docs`. Live in about a minute.

Do **not** copy `config.js` into `docs/` — that would publish your API key.

---

## Path B — FastAPI backend

```bash
pip install -r requirements.txt
```

**1. Copy the data files** listed in `backend/data/PUT_DATA_FILES_HERE.txt`
from Google Drive. Under 350 KB total, no WorldPop raster needed.

**2. Paste the engine.** Open `backend/step6_live.py` and replace `main()` with
the Step 6 cell from `GeoAI_v3.ipynb`. Change exactly one line:

```python
BASE = os.environ.get("TELCOSHIELD_BASE", ".")
```

**3. Run.**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000>. FastAPI serves the dashboard and the API from the
same origin, so no CORS problems.

### Endpoints

| Route | Does |
|---|---|
| `GET /api/towers` | current ranking — **as stale as a JSON file** |
| `GET /api/status` | tower count, escalated count, data age |
| `POST /api/refresh` | **reruns the engine** against live conditions |

Only `/api/refresh` produces new data. Wrapping a file in an API does not make
it fresh; recomputing does. In the browser this is the **Refresh** button that
appears in the header.

Takes 30–90 s, or returns instantly if the 6-hour rainfall cache is still warm.

---

## Optional — AI operator briefings

Objective 4. Two ways, pick either.

**Bulk, ahead of time.** Free tier is capped per day per model, so this batches
12 towers per request: 172 towers in 15 calls, about three minutes.

```bash
cd frontend
cp .env.example .env          # paste your key from aistudio.google.com
pip install google-generativeai python-dotenv
python make_explanations.py
```

Writes `explanations.json`. The dashboard picks it up on refresh, and it works
everywhere afterwards including GitHub Pages, with no key.

**On demand, one at a time.** Copy `config.example.js` to `config.js`, paste your
key. A **Generate briefing** button appears on each tower. Useful when the daily
quota is spent — you only pay for the towers you actually open.

Do not use `gemini-flash-latest`. That alias points at the newest model, which
has the *smallest* free daily quota (20/day). Stable versions are far more
generous.

---

## Optional — rebuild from raw data

Only if `baseline.parquet` is lost. Retrains the model and recomputes population;
needs the WorldPop raster and about ten minutes.

```bash
python scripts/build_towers_json.py
```

Prints the honest validation numbers, including the holdout contamination warning.

---

## Files

```
backend/
  main.py                FastAPI: /api/towers, /api/status, /api/refresh
  step6_live.py          STUB — paste Step 6 here
  tower_terrain.csv      terrain per tower (train_susceptibility drops these)
  data/                  the parquets — see PUT_DATA_FILES_HERE.txt

frontend/
  index.html             the dashboard, no build step, no framework
  towers.json            716 towers scored — fallback when the API is absent
  trace.json             Nov 2024 replay, 65 points per tower
  make_explanations.py   batch AI briefings
  config.example.js      copy to config.js for on-demand briefings

scripts/
  build_towers_json.py   full rebuild from raw CSVs
  export_towers.py       current_ranking.parquet -> docs/towers.json
```

`index.html` tries `/api/towers` first and falls back to `towers.json`. One file,
both modes, nothing to switch.

---

## Troubleshooting

**Empty map, console shows 404 towers.json** — you opened the file directly
instead of serving it. Use `python -m http.server`.

**`Value expected json(516)`** — a `NaN` in the JSON. pandas writes bare `NaN`,
which is not valid JSON. Export with `.fillna(0)` and `allow_nan=False`.

**`ResourceExhausted` from Gemini** — daily quota, not per-minute. Check
<https://ai.dev/rate-limit>. Switch model or set `TOP_N = 60` in
`make_explanations.py`.

**`/api/refresh` returns 500** — `step6_live.py` is still the stub. Paste the
Step 6 cell in.

**Dashboard shows old numbers after a refresh** — hard reload, `Ctrl+Shift+R`.
The page cache-busts its own fetches but the browser may hold the HTML.

**No LIVE badge** — `towers.json` came from `baseline.parquet`, which has no
`live_score`. Export from `current_ranking.parquet` instead.

---

## Method notes

**Susceptibility** — XGBoost on six terrain features against Sentinel-1 flood
extents from Nov 2024 and Dec 2022. Spatial-block CV PR-AUC **0.924 ± 0.015**;
permutation test collapses to 0.508 against a 0.500 base rate, so the signal is
real. Land cover and lat/lon are deliberately excluded — with land cover in,
tree cover took 80% of feature importance, which is the model learning that SAR
cannot see water under canopy rather than learning hydrology.

**Quote the spatial CV, not the 0.937 holdout.** 72% of the 2022 test points
share coordinates with 2024 training points, so that figure is not independent.

**Population** — WorldPop 2020 UN-adjusted, counted inside each tower's service
footprint with a pixel-overlap correction so one resident inside eight coverage
circles is not counted eight times. `isolation` is the share of a tower's users
that no neighbouring tower covers.

**The ranking is driven by consequence, not hazard.** T00198 serves 4,170 people
and ranks first; T00603 serves 6,510 and ranks third. The difference is
isolation, 75% against 59%. The tower serving fewer people outranks the one
serving more because nobody else covers its users.
