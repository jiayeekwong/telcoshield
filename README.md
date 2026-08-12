# telcoshield
TelcoShield

GeoAI prioritisation of flood-critical telecom infrastructure

ASEAN GeoAI Fusion 2026 · HACK_MY_075 · Kelantan, Malaysia

TelcoShield identifies which telecom towers are most vulnerable to flood disruption, and ranks which ones to strengthen first.

It assesses each tower's flood exposure from terrain and observed flood evidence, weights it by the population that would lose service, and produces a prioritised register — answering not "which towers are vulnerable" but "which vulnerable towers would leave people with no signal at all."

The problem

During the December 2021 Malaysian floods, the Ministry of Communications identified 342 flood-affected communication towers in Selangor alone, of which only 153 had resumed operation several days later. Roughly 1,000 towers were damaged nationwide. Three years later, 323 transmission towers were affected again in a single flood wave.

Two root causes were consistent across both events: repair crews lost physical access because roads were submerged, and grid power was cut for safety, leaving towers dependent on backup power installed at ground level — which flooded first.

The current response is reactive. Operators discover outages when subscribers complain, dispatch crews without knowing which roads are passable, and restore sites in effectively arbitrary order.

Results
Metric	Value
Spatial CV PR-AUC	0.924 ± 0.015 (base rate 0.500)
Out-of-event holdout (2024 → 2022)	0.937
Top-decile precision	97.1%
Permutation control	0.508 — no leakage
Spatial transfer (south → north)	0.845

Replaying the November 2024 Kelantan flood at hourly resolution, the live layer escalated on 22 November — five days before the first evacuations — and peaked at 04:00 on 1 December, the day Kelantan recorded 93,763 evacuees. Top-20 ranking churn averaged 0.1 per cycle and zero outside the event.

How it works
Sentinel-1 SAR ────┐
DEM · MERIT Hydro ─┼──► XGBoost susceptibility ──► P(flood) per tower
OpenCelliD ────────┘                                      │
                                                          ▼
WorldPop ───────────────────────────────► STATIC HARDENING REGISTER
                                                          │
ERA5 climatology ───┐                                     │
Open-Meteo forecast ├──► gate stack ──► multiplier ──►    ▼
DID InfoBanjir ─────┘                          LIVE HOURLY WATCHLIST

1 · Flood labels. Sentinel-1 change detection over the Nov 2024 and Dec 2022 Kelantan floods. Dry-season median VV subtracted from the flood-peak mosaic; water appears as a backscatter drop below −3 dB. Both events forced onto orbit 172 ASCENDING — backscatter depends on look geometry, so differencing across orbits detects the satellite's viewpoint rather than water. Masked for permanent water, slope > 5°, HAND > 15 m, and speckle.

2 · Susceptibility model. XGBoost on terrain and hydrology features, with leave-blocks-out cross-validation across 257 spatial blocks. Negatives are drawn only from flood-plausible terrain — sampling randomly across Kelantan would make most negatives hilltops, and the model would learn that high ground is dry.

3 · Tower sites. 95,176 OpenCelliD cell records → quality filter → DBSCAN clustering at 300 m → 716 physical sites. Cell records are sectors and carriers, not masts, and crowdsourced positions place co-located antennas several hundred metres apart, so clustering is required.

4 · Human impact. WorldPop population within a spacing-derived coverage radius, split across overlapping footprints so nobody is counted twice.

static_priority = P(flood) × (0.7 · population + 0.3 · isolation)

Multiplicative on the outside: a tower on high ground stays near zero however many people surround it.

5 · Live layer. Rainfall is converted to a percentile against that tower's own 15-year monthly record — 120 mm is a routine week in Kelantan and an emergency elsewhere. River level is normalised against each gauge's published Normal/Danger thresholds.

trigger    = max(rain_percentile, river_stress)
live_score = static_priority × gate(trigger)

max() rather than a sum, because rain and river are alternative flood pathways: a coastal tower floods from rainfall with no river rising, while an inland tower floods hours after rain fell upstream.

6 · Gate stack. A raw multiplier makes the ranking twitch every cycle, and nobody trusts a twitchy list. Four mechanisms: an eligibility floor, hysteresis (a higher bar to enter escalation than to leave it), asymmetric dwell, and EWMA smoothing on the output.
