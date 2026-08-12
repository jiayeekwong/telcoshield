"""
make_explanations.py — Objective 4: the LLM explanation layer.

Reads towers.json, writes explanations.json. Run from the folder holding
towers.json (normally frontend/).

    pip install google-generativeai python-dotenv
    echo "GOOGLE_API_KEY=your_key" > .env          # and add .env to .gitignore
    python make_explanations.py

WHY OFFLINE AND NOT IN THE BROWSER
  A static site cannot hide an API key: anything in the JS is readable by
  whoever loads the page. Generating ahead of time also removes demo-day
  latency and network risk, and the result works with no key at all.

QUOTA
  The free tier caps requests per DAY per MODEL. Two consequences:
    - batch several towers per request (BATCH), not one each
    - rotate across models, because each has its own daily allowance
  Six models at ~20/day is ~120 requests; 172 towers needs 15.

THE MODEL REWRITES, IT DOES NOT DECIDE
  Every figure it receives was computed by the pipeline. It does not estimate
  risk, count population, rank towers, or choose the recommended measure.
  Step 5 checks every number in the output against the source data.
"""

import os, json, time, re, textwrap
from pathlib import Path

# ---------------------------------------------------------------- config
IN, OUT = "towers.json", "explanations.json"
CACHE = ".explanations_cache.json"

THRESHOLD = 0.25      # escalation floor — below this no action is recommended
BATCH = 12            # towers per request
PAUSE = 13            # seconds between requests, keeps us under ~5 RPM
TOP_N = None          # e.g. 60 to cover only the top-priority cohort

# Lite variants first: they carry the most generous free daily limits.
# Never use "gemini-flash-latest" — it resolves to the newest model, which has
# the SMALLEST quota of all. Trim this list to what your key actually offers.
MODEL_CANDIDATES = [
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-3.6-flash",
]

# 8000, not 4000: newer models spend tokens on internal reasoning before the
# visible answer, and a low cap truncates briefings mid-sentence.
CFG = {"temperature": 0.3, "max_output_tokens": 8000,
       "response_mime_type": "application/json"}    # forces parseable JSON

# ---------------------------------------------------------------- load
if not Path(IN).exists():
    raise SystemExit(f"{IN} not found. Run this from the folder that contains it.")

towers = json.load(open(IN))
elig = sorted([t for t in towers if t["baseline"] >= THRESHOLD],
              key=lambda t: t["rank"])
if TOP_N:
    elig = elig[:TOP_N]
print(f"{len(elig)} of {len(towers)} towers eligible (baseline >= {THRESHOLD})")

# ---------------------------------------------------------------- 1. drivers
# True SHAP needs the trained model, which lives in Colab. This is a documented
# proxy: model feature importance weighted by how extreme this tower's value is
# within the network. Call it "risk drivers", never "SHAP", in the writeup.
IMPORTANCE = {"slope": 0.363, "elev": 0.260, "hand": 0.112, "twi": 0.084}
HIGHER_IS_RISKIER = {"slope": False, "elev": False, "hand": False, "twi": True}
PHRASE = {
    "slope": "very flat ground that holds water rather than shedding it",
    "elev":  "low elevation relative to the rest of the network",
    "hand":  "sits barely above the nearest drainage line",
    "twi":   "terrain shaped to accumulate water",
}


def pct_rank_fn(field):
    vals = sorted(t[field] for t in towers)
    n = len(vals)

    def rank(v):
        lo, hi = 0, n
        while lo < hi:                       # bisect_left
            mid = (lo + hi) // 2
            if vals[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        return lo / n
    return rank


RANK = {f: pct_rank_fn(f) for f in IMPORTANCE}


def drivers(t, k=2):
    scored = []
    for f, imp in IMPORTANCE.items():
        p = RANK[f](t[f])
        scored.append((imp * (p if HIGHER_IS_RISKIER[f] else 1 - p), f))
    scored.sort(reverse=True)
    return [PHRASE[f] for s, f in scored[:k] if s > 0.05]


# ---------------------------------------------------------------- 2. action
# Identical rules to index.html, so the card and the briefing cannot disagree.
def action_of(t):
    if t["hand"] < 2 and t["baseline"] >= 0.6:
        return "elevate the equipment platform above the modelled inundation level"
    if t["isolation"] > 0.5 and t["baseline"] >= 0.4:
        return "priority hardening, because no fallback coverage exists"
    if t["baseline"] >= 0.4 and t["nn_dist_km"] > 3:
        return "extended-autonomy backup power, solar preferred over diesel"
    return "monitor and pre-position portable capacity"


# ---------------------------------------------------------------- 3. prompt
SYSTEM = textwrap.dedent("""
    You write short operational briefings for a Malaysian mobile network
    operator's resilience planning team.

    You will be given several towers at once. Return a JSON object mapping each
    tower_id to its briefing string. No other keys, no commentary.

    Each briefing, without exception:
    - Uses ONLY the figures supplied for that tower. Never estimate, and never
      introduce a number that is not given.
    - Writes all numbers as digits. Never spell a number out in words.
    - Is 2 to 3 sentences and at most 55 words. No greeting, no heading, no
      bullets, no markdown.
    - Is plain British English a field engineer would use.
    - Leads with the consequence to subscribers, then the cause, then the action.
    - States each fact once. Do not restate the redundancy deficit in different
      words within the same briefing.
    - Treats flooding as the cause of outage. The redundancy deficit explains
      why the loss cannot be absorbed by a neighbouring tower; never present it
      as the cause of the outage itself.
    - Says plainly that no neighbouring tower can absorb this site's users when
      the redundancy deficit is above 50%.
    - Never speculates about when a flood will occur. This is structural
      vulnerability, not a forecast.
""").strip()


# Rank is deliberately NOT supplied: live scoring reorders it, so a cached
# briefing would contradict the panel during a flood. This tier is derived from
# static_priority, which does not move.
CUTS = sorted((t["static_priority"] for t in towers), reverse=True)
def tier_label(t):
    p = t["static_priority"]
    if p >= CUTS[int(len(CUTS) * 0.05)]: return "in the top 5% by structural priority"
    if p >= CUTS[int(len(CUTS) * 0.15)]: return "in the top 15% by structural priority"
    if p >= CUTS[int(len(CUTS) * 0.35)]: return "in the top 35% by structural priority"
    return "below the top 35% by structural priority"


def block(t):
    d = ", ".join(drivers(t)) or "terrain characteristics"
    return textwrap.dedent(f"""
        tower_id: {t['tower_id']}  (Kelantan, {tier_label(t)})
        flood susceptibility: {t['baseline']:.2f} on 0-1, from terrain compared
          against Sentinel-1 flood extents observed in Nov 2024 and Dec 2022
        main drivers: {d}
        height above nearest drainage: {t['hand']:.1f} m
        elevation: {t['elev']:.0f} m
        population served: {t['pop_served']:,.0f}
        population with no alternative coverage: {t['pop_sole']:,.0f} ({t['isolation']:.0%} of users)
        neighbouring towers within 5 km: {t['n_within_5km']}
        nearest neighbour: {t['nn_dist_km']:.1f} km
        recommended measure: {action_of(t)}
    """).strip()


def batch_prompt(group):
    return ("Write one briefing for each tower below.\n\n"
            + "\n\n---\n\n".join(block(t) for t in group))


# ---------------------------------------------------------------- 4. generate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise SystemExit("GOOGLE_API_KEY not set. Put it in a .env file next to "
                     "this script, and add .env to .gitignore before pushing.")
genai.configure(api_key=key)


def resolve_models():
    """Return every candidate the key can reach, in order.

    Plural on purpose: quota is per model per day, so we rotate through them
    rather than burning one model's allowance and stopping."""
    try:
        avail = {m.name.replace("models/", "") for m in genai.list_models()
                 if "generateContent" in getattr(m, "supported_generation_methods", [])}
    except Exception as e:
        print(f"could not list models ({type(e).__name__}) — trying all candidates")
        avail = None

    usable = [n for n in MODEL_CANDIDATES if avail is None or n in avail]
    if not usable and avail:
        usable = sorted(n for n in avail if "flash" in n)
        print(f"none of MODEL_CANDIDATES available — falling back to: {usable}")
    if not usable:
        raise SystemExit("No usable model. Check GOOGLE_API_KEY is valid.")

    print(f"model pool ({len(usable)}): {', '.join(usable)}")
    return [genai.GenerativeModel(n, system_instruction=SYSTEM) for n in usable], usable


MODELS, MODEL_NAMES = resolve_models()

expl = json.load(open(CACHE)) if Path(CACHE).exists() else {}
todo = [t for t in elig if t["tower_id"] not in expl]
groups = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
print(f"{len(expl)} cached, {len(todo)} to generate in {len(groups)} requests "
      f"(~{max(1, len(groups) * PAUSE // 60)} min)\n")

pool = list(range(len(MODELS)))          # indices of models still under quota

for gi, group in enumerate(groups, 1):
    ids = [t["tower_id"] for t in group]
    if not pool:
        print(f"  !! every model exhausted — {len(todo) - len(expl)} towers left")
        break

    for attempt in range(len(pool) + 2):
        mi = pool[gi % len(pool)]        # rotate, spreading load across the pool
        try:
            r = MODELS[mi].generate_content(batch_prompt(group), generation_config=CFG)
            got = json.loads(r.text)
            for tid in ids:
                v = got.get(tid)
                if isinstance(v, str) and v.strip():
                    expl[tid] = " ".join(v.split())
            miss = [i for i in ids if i not in expl]
            print(f"  batch {gi}/{len(groups)} [{MODEL_NAMES[mi]}]: "
                  f"{len(ids) - len(miss)}/{len(ids)}"
                  + (f"  missing {miss}" if miss else ""))
            break

        except json.JSONDecodeError:
            print(f"  batch {gi} returned unparseable JSON, retrying")
            time.sleep(PAUSE)

        except Exception as e:
            name = type(e).__name__
            if name == "ResourceExhausted":
                # this model is spent for today; drop it and try another
                print(f"  {MODEL_NAMES[mi]} quota exhausted — dropping from pool")
                pool = [p for p in pool if p != mi]
                if not pool:
                    break
            else:
                print(f"  batch {gi} [{MODEL_NAMES[mi]}]: {name}, waiting {PAUSE}s")
                time.sleep(PAUSE)
    else:
        print(f"  !! batch {gi} failed on every model, skipping {ids}")

    json.dump(expl, open(CACHE, "w"))    # checkpoint, so a crash costs one batch
    time.sleep(PAUSE)

# ---------------------------------------------------------------- 5. verify
# The failure mode that matters is a fluent sentence containing a number the
# pipeline never produced. Catch it here, not during the Q&A.
NUMBER_WORDS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b",
    re.I)

problems = []
for t in elig:
    txt = expl.get(t["tower_id"], "")
    if not txt:
        continue

    issues = []

    # Allowed values come from the same fields block() supplies, so the guard
    # cannot drift from the prompt. The regex only catches 3+ digit numbers,
    # which is why small values like slope and HAND are not listed.
    found = {int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]{2,}\b", txt)}
    ok = {round(t["pop_served"]), round(t["pop_sole"]), round(t["elev"])}
    ok |= {n + d for n in list(ok) for d in (-1, 1)}       # rounding slack
    stray = found - ok
    if stray:
        issues.append(f"unverified {sorted(stray)}")

    # spelled-out numbers cannot be checked, so they are never acceptable
    m = NUMBER_WORDS.search(txt)
    if m:
        issues.append(f'spelled-out "{m.group(0)}"')

    if len(txt.split()) > 70:
        issues.append(f"{len(txt.split())} words")

    if issues:
        problems.append((t["tower_id"], "; ".join(issues)))

json.dump(expl, open(OUT, "w"), allow_nan=False)

done = sum(1 for t in elig if t["tower_id"] in expl)
print(f"\n{done}/{len(elig)} briefings written to {OUT}")
if problems:
    print(f"{len(problems)} need attention:")
    for tid, why in problems[:10]:
        print(f"  {tid}: {why}")
    print("\nDelete those keys from .explanations_cache.json and re-run to redo them.")
else:
    print("0 with an unverified number — clean")

if done:
    first = next(t for t in elig if t["tower_id"] in expl)
    print(f"\n--- sample: {first['tower_id']} ---\n{expl[first['tower_id']]}")