"""
Script 13 — Build a persistent notation cache.

Every downstream script (benchmark, embedding bridge, validity probes) needs the
IUPAC name and InChI string for the same molecules. Re-fetching them per script
triggered PubChem rate limiting (HTTP 503 PUGREST.ServerBusy), which silently
produced zero valid IUPAC/InChI responses in one probe run and would have been
misread as model failure.

This script fetches each molecule's notations ONCE, with polite rate limiting and
exponential backoff, and writes a cache all other scripts load. SELFIES are
computed locally and need no network.

PubChem's documented limit is 5 requests/second; we use a conservative 2/second
with backoff on 503, and resume from the existing cache on restart.

Usage:  python3 13_build_notation_cache.py [max_molecules]
"""

import requests, json, time, os, sys, urllib.parse, csv, io
import selfies as sf

CACHE   = "notation_cache.json"
ESOL    = ("https://raw.githubusercontent.com/deepchem/deepchem/"
           "master/datasets/delaney-processed.csv")
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
LIMIT   = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # 0 = all

MIN_INTERVAL = 0.5      # seconds between requests (2/sec, well under the 5/sec limit)
MAX_BACKOFF  = 120.0

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
print(f"Cache: {len(cache)} molecules already stored")

rows = list(csv.DictReader(io.StringIO(requests.get(ESOL, timeout=60).text)))
sc = next(c for c in rows[0] if 'smiles' in c.lower())
lc = next(c for c in rows[0] if ('measured' in c.lower() or 'log' in c.lower())
          and 'smiles' not in c.lower())

mols = []
for r in rows:
    try:
        mols.append((r[sc].strip(), float(r[lc])))
    except ValueError:
        pass
if LIMIT:
    mols = mols[:LIMIT]

todo = [(s, l) for s, l in mols if s not in cache]
print(f"To fetch: {len(todo)} of {len(mols)}\n")

last_req = 0.0
backoff = 1.0
fetched = failed = 0

for i, (smi, logs) in enumerate(todo):
    # --- polite pacing ---
    wait = MIN_INTERVAL - (time.time() - last_req)
    if wait > 0:
        time.sleep(wait)

    iupac = inchi = None
    status = "ok"
    for attempt in range(5):
        try:
            url = (f"{PUBCHEM}/smiles/{urllib.parse.quote(smi, safe='')}"
                   f"/property/IUPACName,InChI/JSON")
            r = requests.get(url, timeout=30)
            last_req = time.time()

            if r.status_code == 200:
                p = r.json()["PropertyTable"]["Properties"][0]
                iupac, inchi = p.get("IUPACName"), p.get("InChI")
                backoff = max(1.0, backoff * 0.5)      # recover after success
                break
            if r.status_code == 404:
                status = "not_found"
                break
            if r.status_code in (503, 429):            # throttled
                status = "throttled"
                print(f"    throttled; backing off {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(MAX_BACKOFF, backoff * 2)
                continue
            status = f"http_{r.status_code}"
        except Exception as e:
            status = "error"
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)

    try:
        selfies = sf.encoder(smi)
    except Exception:
        selfies = None

    cache[smi] = {"true_logS": logs, "IUPAC": iupac, "InChI": inchi,
                  "SELFIES": selfies, "status": status}
    if iupac or inchi:
        fetched += 1
    else:
        failed += 1

    if (i + 1) % 25 == 0:
        json.dump(cache, open(CACHE, "w"))
        print(f"  [{i+1}/{len(todo)}] ok={fetched} failed={failed} "
              f"backoff={backoff:.1f}s", flush=True)

json.dump(cache, open(CACHE, "w"))

have = {k: sum(1 for v in cache.values() if v.get(k)) for k in ("IUPAC", "InChI", "SELFIES")}
print(f"\n✓ Cache written: {CACHE}  ({len(cache)} molecules)")
print(f"  IUPAC   present: {have['IUPAC']}")
print(f"  InChI   present: {have['InChI']}")
print(f"  SELFIES present: {have['SELFIES']}")
bad = [v["status"] for v in cache.values() if v["status"] != "ok"]
if bad:
    from collections import Counter
    print(f"  non-ok statuses: {dict(Counter(bad))}")
