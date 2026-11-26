"""Choose the closest-to-even intact tournament openings using only the baseline."""
import hashlib
import json
import struct
from bit_gate import capture
from datasets import DATA, ROOT, SEED, canonical, dump, sha


def freeze():
    rows = json.loads((DATA / "opening_candidates.json").read_bytes())
    heads = capture("orig", rows, "book_screen")
    chosen = []
    for row, head in zip(rows, heads, strict=True):
        w, l, d = (struct.unpack("<f", struct.pack("<I", x))[0] for x in head["wdl"])
        row["baseline_black_score"] = (l if len(row["moves"]) % 2 else w) + .5*d
    for rule, count, prefix in [(1, 60, "S"), (2, 40, "R")]:
        candidates = sorted((r for r in rows if r["rule"] == rule),
                            key=lambda r: (abs(.5-r["baseline_black_score"]), r["year"], r["line"]))
        assert len(candidates) >= count
        selected = candidates[:count]
        assert len({canonical(r["moves"]) for r in selected}) == count
        selected.sort(key=lambda r: hashlib.sha256(f"{SEED}/{r['year']}/{r['line']}".encode()).digest())
        for i, row in enumerate(selected, 1):
            chosen.append(dict(id=f"{prefix}{i:02d}", **row))
        print(prefix, count, "baseline expected black score",
              min(r["baseline_black_score"] for r in selected), max(r["baseline_black_score"] for r in selected))
    dump(DATA / "opening_screen.json", rows)
    dump(DATA / "openings.json", chosen)
    print("Book SHA-256", sha(DATA / "openings.json"), flush=True)


if __name__ == "__main__":
    freeze()
