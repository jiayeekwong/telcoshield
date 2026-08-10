"""
backend/step6_live.py — TelcoShield live tower priority engine

    live_score = static_priority x multiplier
    multiplier = gate( max(rain_percentile, river_stress) )

THIS IS A STUB. Replace the body of main() with the Step 6 cell from
GeoAI_v3.ipynb. Only ONE line needs to change from the notebook version:

    BASE = "/content/drive/MyDrive/telcoshield"      # notebook
    BASE = os.environ.get("TELCOSHIELD_BASE", ".")   # here

Everything else copies across unchanged.

WHY A STUB RATHER THAN A COPY
  The notebook is the source of truth for the engine. Shipping a second copy
  invites the two to drift, and then the dashboard and the notebook disagree
  with no way to say which is right. Paste yours in and there is one engine.
"""

import os

BASE = os.environ.get("TELCOSHIELD_BASE", ".")

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ARCH = Path(f"{BASE}/archive")
ARCH.mkdir(parents=True, exist_ok=True)

Path(f"{BASE}/cache").mkdir(parents=True, exist_ok=True) 

# ================================================================
# RAINFALL WINDOW
#
# Operational rainfall signal:
#     previous 48 h rainfall
#     + next 24 h forecast rainfall
#     = 72 h total accumulation
#
# This matches the historical 3-day / 72 h climatology.
# ================================================================
LOCAL_TZ = "Asia/Kuala_Lumpur"

RAIN_PAST_H = 48
RAIN_FCST_H = 24
RAIN_WINDOW_H = RAIN_PAST_H + RAIN_FCST_H

assert RAIN_WINDOW_H == 72

RAIN_CACHE_MIN = 360        # 6 h
GAUGE_STALE_H = 6


GATE = dict(
    eligibility_floor = 0.25,
    enter_pctl        = 0.95,
    exit_pctl         = 0.88,
    quiet_cycles      = 18,
    ewma_alpha        = 0.083,
    max_multiplier    = 4.0,
    escalation_gain   = 3.0,
)


QCOLS = [
    ("q500", .50),
    ("q750", .75),
    ("q900", .90),
    ("q950", .95),
    ("q975", .975),
    ("q990", .99),
    ("q995", .995),
    ("q1000", 1.0)
]


# -----------------------------------------------------------------
# rainfall
# -----------------------------------------------------------------
def fetch_rain(tw):
    """
    Per-tower 72 h operational rainfall signal.

    Definition:
        previous 48 h observed/recent rainfall
        + next 24 h forecast rainfall
        = 72 h total

    This is compared against the historical
    3-day / 72 h rainfall climatology.
    """

    # New cache name prevents old 144 h results from being reused.
    cache = Path(
        f"{BASE}/cache/rain_now_48p24.parquet"
    )

    if cache.exists():

        age = (
            dt.datetime.now().timestamp()
            - cache.stat().st_mtime
        ) / 60

        if age < RAIN_CACHE_MIN:
            print(
                f"rain: cache {age:.0f} min old"
            )
            return pd.read_parquet(cache)

    # Open-Meteo ~0.1° cells.
    # Towers in same grid cell share rainfall.
    tw = tw.copy()

    tw["glat"] = (
        tw.lat / 0.1
    ).round() * 0.1

    tw["glon"] = (
        tw.lon / 0.1
    ).round() * 0.1

    cells = (
        tw[
            ["glat", "glon"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    out = []

    for i in range(
        0,
        len(cells),
        100
    ):

        ch = cells.iloc[
            i:i + 100
        ]

        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            timeout=180,
            params={
                "latitude":
                    ",".join(
                        f"{v:.4f}"
                        for v in ch.glat
                    ),

                "longitude":
                    ",".join(
                        f"{v:.4f}"
                        for v in ch.glon
                    ),

                "hourly":
                    "precipitation",

                # Request slightly more than needed,
                # then explicitly select 48 h + 24 h.
                "past_days": 3,
                "forecast_days": 2,

                "timezone":
                    LOCAL_TZ
            }
        )

        r.raise_for_status()

        d = r.json()

        locations = (
            [d]
            if isinstance(d, dict)
            else d
        )

        for (
            (_, cell),
            loc
        ) in zip(
            ch.iterrows(),
            locations
        ):

            s = pd.Series(
                loc[
                    "hourly"
                ][
                    "precipitation"
                ],
                index=pd.to_datetime(
                    loc[
                        "hourly"
                    ][
                        "time"
                    ]
                )
            ).fillna(0.0)

            s = s.sort_index()

            # Open-Meteo timestamps are Malaysia local time
            # because timezone=Asia/Kuala_Lumpur.
            now = (
                pd.Timestamp.now(
                    tz=LOCAL_TZ
                )
                .floor("h")
                .tz_localize(None)
            )

            # -----------------------------
            # Previous 48 hours
            # -----------------------------
            past_start = (
                now
                - pd.Timedelta(
                    hours=RAIN_PAST_H
                )
            )

            past_series = s[
                (s.index > past_start)
                &
                (s.index <= now)
            ]

            # -----------------------------
            # Next 24 hours forecast
            # -----------------------------
            fcst_end = (
                now
                + pd.Timedelta(
                    hours=RAIN_FCST_H
                )
            )

            fcst_series = s[
                (s.index > now)
                &
                (s.index <= fcst_end)
            ]

            # Diagnostics
            if (
                len(past_series)
                <
                RAIN_PAST_H
            ):
                print(
                    f"WARNING rain cell "
                    f"({cell.glat:.2f},"
                    f"{cell.glon:.2f}): "
                    f"only "
                    f"{len(past_series)}/"
                    f"{RAIN_PAST_H} "
                    f"past hours"
                )

            if (
                len(fcst_series)
                <
                RAIN_FCST_H
            ):
                print(
                    f"WARNING rain cell "
                    f"({cell.glat:.2f},"
                    f"{cell.glon:.2f}): "
                    f"only "
                    f"{len(fcst_series)}/"
                    f"{RAIN_FCST_H} "
                    f"forecast hours"
                )

            past = (
                past_series.sum()
            )

            fwd = (
                fcst_series.sum()
            )

            # FINAL OPERATIONAL
            # 72-HOUR RAINFALL SIGNAL
            rain_72h = (
                past
                +
                fwd
            )

            out.append({
                "glat":
                    cell.glat,

                "glon":
                    cell.glon,

                "rain_past":
                    past,

                "rain_fcst":
                    fwd,

                "rain_window":
                    rain_72h
            })

    df = tw.merge(
        pd.DataFrame(out),
        on=[
            "glat",
            "glon"
        ]
    )[
        [
            "tower_id",
            "rain_past",
            "rain_fcst",
            "rain_window"
        ]
    ]

    df.to_parquet(
        cache,
        index=False
    )

    print(
        f"rain: fetched "
        f"{len(cells)} cells "
        f"-> {len(df)} towers"
    )

    return df


# -----------------------------------------------------------------
# rainfall percentile
# -----------------------------------------------------------------
def to_percentile(
    value,
    q
):

    """
    Where does current 72 h rainfall sit
    in this tower's historical 72 h distribution?
    """

    xs = np.array(
        [
            q[c]
            for c, _ in QCOLS
        ],
        float
    )

    ps = np.array(
        [
            p
            for _, p in QCOLS
        ]
    )

    if np.isnan(
        xs
    ).any():

        return 0.0

    if value <= xs[0]:

        return float(
            np.clip(
                value
                /
                max(
                    xs[0],
                    1e-6
                )
                *
                0.5,
                0,
                0.5
            )
        )

    if value >= xs[-1]:

        return 1.0

    return float(
        np.interp(
            value,
            xs,
            ps
        )
    )


# -----------------------------------------------------------------
# river
# -----------------------------------------------------------------
def river_stress():

    """
    Latest river reading per station,
    normalised against station-specific thresholds.
    """

    log = pd.read_csv(
        f"{BASE}/data/river_log.csv"
    )

    log[
        "scraped_at"
    ] = pd.to_datetime(
        log.scraped_at,
        errors="coerce",
        utc=True
    )

    log[
        "updated"
    ] = pd.to_datetime(
        log.updated,
        errors="coerce",
        utc=True
    )

    latest = (
        log
        .sort_values(
            "scraped_at"
        )
        .groupby(
            "station"
        )
        .tail(1)
        .copy()
    )

    # DID NoData
    latest.loc[
        latest.level < -100,
        "level"
    ] = np.nan

    age_h = (
        pd.Timestamp.now(
            tz="UTC"
        )
        -
        latest.updated
    ).dt.total_seconds() / 3600

    span = (
        latest.danger
        -
        latest.normal
    )

    stress = (
        (
            latest.level
            -
            latest.normal
        )
        /
        span.where(
            span > 0
        )
    ).clip(
        0,
        1.5
    )

    drop = (
        latest.level.isna()
        |
        latest.is_reservoir
        |
        latest.suspect
        |
        (
            age_h
            >
            GAUGE_STALE_H
        )
    )

    out = {
        s: v
        for s, v
        in zip(
            latest.station[
                ~drop
            ],
            stress[
                ~drop
            ]
        )
        if pd.notna(v)
    }

    print(
        f"river: "
        f"{len(out)}/"
        f"{len(latest)} usable "
        f"("
        f"{latest.is_reservoir.sum()} reservoir, "
        f"{(age_h > GAUGE_STALE_H).sum()} stale"
        f")"
    )

    return out


# -----------------------------------------------------------------
# gate
# -----------------------------------------------------------------
def gate(
    trigger,
    baseline,
    prev,
    g=GATE
):

    """
    Anti-flap stack.

    Returns:
        multiplier
        escalated
        quiet_count
    """

    # High-ground / low-baseline towers
    # should not escalate solely because
    # of extreme rainfall.
    if (
        baseline
        <
        g[
            "eligibility_floor"
        ]
    ):

        return (
            1.0,
            False,
            0
        )

    esc = bool(
        prev.get(
            "escalated",
            False
        )
    )

    quiet = int(
        prev.get(
            "quiet",
            0
        )
    )

    m_prev = float(
        prev.get(
            "multiplier",
            1.0
        )
    )

    # Hysteresis
    if (
        trigger
        >=
        g[
            "enter_pctl"
        ]
    ):

        esc = True
        quiet = 0

    elif (
        trigger
        <
        g[
            "exit_pctl"
        ]
    ):

        quiet += 1

        if (
            quiet
            >=
            g[
                "quiet_cycles"
            ]
        ):

            esc = False

    else:

        quiet = 0

    if esc:

        raw = (
            1
            +
            g[
                "escalation_gain"
            ]
            *
            max(
                0.0,
                (
                    trigger
                    -
                    0.90
                )
                /
                0.10
            )
        )

    else:

        raw = 1.0

    a = g[
        "ewma_alpha"
    ]

    multiplier = (
        a
        *
        raw
        +
        (
            1
            -
            a
        )
        *
        m_prev
    )

    return (
        float(
            min(
                g[
                    "max_multiplier"
                ],
                multiplier
            )
        ),
        esc,
        quiet
    )


# -----------------------------------------------------------------
# main
# -----------------------------------------------------------------
def main():

    tw = pd.read_parquet(
        f"{BASE}/data/baseline.parquet"
    )

    clim = pd.read_parquet(
        f"{BASE}/data/rain_climatology.parquet"
    )

    smap = pd.read_parquet(
        f"{BASE}/data/tower_station_map.parquet"
    )

    rain = fetch_rain(
        tw
    )

    stress = river_stress()

    # -------------------------------------------------------------
    # previous hysteresis state
    # -------------------------------------------------------------
    state_f = Path(
        f"{BASE}/data/gate_state.parquet"
    )

    prev = (
        pd.read_parquet(
            state_f
        )
        .set_index(
            "tower_id"
        )
        .to_dict(
            "index"
        )
        if state_f.exists()
        else {}
    )

    # Malaysia-local month
    month = pd.Timestamp.now(
        tz=LOCAL_TZ
    ).month

    cm = (
        clim[
            clim.month
            ==
            month
        ]
        .set_index(
            "tower_id"
        )
    )

    if cm.empty:

        print(
            f"** WARNING: "
            f"no climatology "
            f"for month {month}, "
            f"pooling all months **"
        )

        cm = (
            clim
            .groupby(
                "tower_id"
            )
            .mean(
                numeric_only=True
            )
        )

    df = (
        tw
        .merge(
            rain,
            on="tower_id"
        )
        .merge(
            smap,
            on="tower_id"
        )
    )

    rows = []

    for r in df.itertuples():

        q = (
            cm.loc[
                r.tower_id
            ]
            if r.tower_id
            in cm.index
            else None
        )

        rain_sig = (
            to_percentile(
                r.rain_window,
                q
            )
            if q is not None
            else 0.0
        )

        riv = (
            stress.get(
                r.station
            )
            if r.has_river_signal
            else None
        )

        trigger = (
            max(
                rain_sig,
                riv
            )
            if riv is not None
            else rain_sig
        )

        m, esc, quiet = gate(
            trigger,
            r.baseline,
            prev.get(
                r.tower_id,
                {}
            )
        )

        rows.append(
            dict(
                tower_id=
                    r.tower_id,

                lat=
                    r.lat,

                lon=
                    r.lon,

                baseline=
                    round(
                        r.baseline,
                        3
                    ),

                static_priority=
                    round(
                        r.static_priority,
                        4
                    ),

                pop_served=
                    round(
                        r.pop_served
                    ),

                pop_sole=
                    round(
                        r.pop_sole
                    ),

                isolation=
                    round(
                        r.isolation,
                        3
                    ),

                rain_mm=
                    round(
                        r.rain_window,
                        1
                    ),

                rain_pctl=
                    round(
                        rain_sig,
                        3
                    ),

                river_stress=(
                    None
                    if riv is None
                    else round(
                        float(riv),
                        3
                    )
                ),

                gauge=(
                    r.station
                    if r.has_river_signal
                    else None
                ),

                trigger=
                    round(
                        trigger,
                        3
                    ),

                multiplier=
                    round(
                        m,
                        3
                    ),

                live_score=
                    round(
                        r.static_priority
                        *
                        m,
                        4
                    ),

                escalated=
                    esc,

                quiet=
                    quiet
            )
        )

    res = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "live_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    res[
        "rank"
    ] = range(
        1,
        len(res) + 1
    )

    # -------------------------------------------------------------
    # rank change
    # -------------------------------------------------------------
    cur = Path(
        f"{BASE}/data/current_ranking.parquet"
    )

    if cur.exists():

        old = (
            pd.read_parquet(
                cur
            )[
                [
                    "tower_id",
                    "rank"
                ]
            ]
            .rename(
                columns={
                    "rank":
                        "prev_rank"
                }
            )
        )

        res = res.merge(
            old,
            on="tower_id",
            how="left"
        )

        res[
            "rank_change"
        ] = (
            res.prev_rank
            -
            res[
                "rank"
            ]
        )

    else:

        res[
            "prev_rank"
        ] = pd.NA

        res[
            "rank_change"
        ] = 0

    # -------------------------------------------------------------
    # save state
    # -------------------------------------------------------------
    res.to_parquet(
        cur,
        index=False
    )

    res[
        [
            "tower_id",
            "escalated",
            "quiet",
            "multiplier"
        ]
    ].to_parquet(
        state_f,
        index=False
    )

    # -------------------------------------------------------------
    # archive snapshot
    # -------------------------------------------------------------
    ts = dt.datetime.now(
        dt.timezone.utc
    )

    snap = res.copy()

    snap.insert(
        0,
        "timestamp",
        ts.isoformat()
    )

    snap.to_parquet(
        ARCH
        /
        f"{ts:%Y%m%dT%H}.parquet",
        index=False
    )

    # -------------------------------------------------------------
    # report
    # -------------------------------------------------------------
    print(
        f"\n{len(res)} towers "
        f"| {res.escalated.sum()} escalated "
        f"| mean multiplier "
        f"{res.multiplier.mean():.2f} "
        f"| max rain pctl "
        f"{res.rain_pctl.max():.2f}"
    )

    if (
        res.prev_rank
        .notna()
        .any()
    ):

        t20 = set(
            res.head(
                20
            ).tower_id
        )

        p20 = set(
            res.nsmallest(
                20,
                "prev_rank"
            ).tower_id
        )

        print(
            f"top-20 churn: "
            f"{len(t20 - p20)} "
            f"(0-2 healthy, "
            f">5 = gates too loose)"
        )

    print(
        res.head(
            10
        )[
            [
                "rank",
                "tower_id",
                "pop_served",
                "rain_pctl",
                "river_stress",
                "multiplier",
                "live_score",
                "rank_change"
            ]
        ]
        .to_string(
            index=False
        )
    )
    
if __name__ == "__main__":
    main()
