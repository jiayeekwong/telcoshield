"""Hourly DID InfoBanjir river level logger. Runs on GitHub Actions."""
import io, datetime as dt, requests, pandas as pd
from pathlib import Path

STATES = ["KEL"]
LOG    = Path("data/river_log.csv")
URL    = ("https://publicinfobanjir.water.gov.my/aras-air/data-paras-air/"
          "aras-air-data/?state={s}&district=ALL&station=ALL&lang=en")

COLMAP = {
    "station id station id":                   "station_id",
    "station name station name":               "station",
    "district district":                       "district",
    "main basin main basin":                   "basin",
    "sub river basin sub river basin":         "river",
    "last updated last updated":               "updated",
    "water level (m) (graph) water level (m) (graph)": "level",
    "threshold normal":  "normal",
    "threshold alert":   "alert",
    "threshold warning": "warning",
    "threshold danger":  "danger",
}

def fetch(state):
    r = requests.get(URL.format(s=state), timeout=60)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    df.columns = [" ".join(str(c) for c in col).strip().lower() for col in df.columns]
    return df

def parse(df, state):
    missing = [k for k in COLMAP if k not in df.columns]
    if missing:
        raise KeyError(f"table layout changed, missing: {missing}")
    out = df[list(COLMAP)].rename(columns=COLMAP).copy()
    for c in ["level", "normal", "alert", "warning", "danger"]:
        out[c] = pd.to_numeric(
            out[c].astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    out["updated"] = pd.to_datetime(out.updated, format="%d/%m/%Y %H:%M",
                                    errors="coerce").dt.tz_localize("Asia/Kuala_Lumpur")
    # reservoirs sit near "danger" by design and would escalate permanently
    out["is_reservoir"] = out.station.str.contains(
        "empangan|dam|kolam|takungan", case=False, na=False)
    span = out.danger - out.normal
    out["stress"] = ((out.level - out.normal) / span.where(span > 0)).clip(0, 1.5)
    out["suspect"] = (span > 20) | (span <= 0)
    out["state"] = state
    out["scraped_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out.dropna(subset=["station"])

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in STATES:
        try:
            frames.append(parse(fetch(s), s))
        except Exception as e:
            print(f"{s}: FAILED — {e}")
    if not frames:
        print("nothing scraped"); return
    new = pd.concat(frames, ignore_index=True)
    if LOG.exists():
        old = pd.read_csv(LOG, nrows=0).columns.tolist()
        if old != new.columns.tolist():
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            LOG.rename(LOG.with_name(f"river_log_{stamp}.csv"))
            print(f"** schema changed — archived as river_log_{stamp}.csv **")
    new.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"{len(new)} rows | max stress {new[~new.is_reservoir].stress.max():.2f}")

if __name__ == "__main__":
    main()