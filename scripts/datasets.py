"""Freeze tournament positions and intact official opening candidates before testing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import zipfile
from rules import valid_history

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 35820261126


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(moves):
    variants = []
    for transpose in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                pts = [(p % 15 - 7, p // 15 - 7) for p in moves]
                if transpose:
                    pts = [(y, x) for x, y in pts]
                pts = [(sx * x, sy * y) for x, y in pts]
                variants.append((tuple(sorted(pts[::2])), tuple(sorted(pts[1::2]))))
    return min(variants)


def opening_candidates():
    seen = {1: set(), 2: set()}
    openings = []
    for path in sorted((DATA / "opening_sources").glob("*.txt")):
        year = int(path.stem[:4])
        rule = 2 if "renju" in path.stem else 1
        label = "renju" if rule == 2 else "standard"
        for n, line in enumerate(path.read_text().splitlines(), 1):
            numbers = list(map(int, re.findall(r"-?\d+", line)))
            if not numbers or len(numbers) % 2:
                continue
            points = list(zip(numbers[::2], numbers[1::2]))
            if any(abs(x) > 7 or abs(y) > 7 for x, y in points):
                continue
            moves = [x + 7 + 15 * (y + 7) for x, y in points]
            key = canonical(moves)
            if key in seen[rule] or not valid_history(moves, rule):
                continue
            seen[rule].add(key)
            url = (f"https://gomocup.org/static/tournaments/{year}/openings/openings_{label}.txt"
                   if year < 2024 else f"https://gomocup.org/static/tournaments/{year}/results/gomocup{year}results.zip")
            openings.append(dict(rule=rule, moves=moves, year=year, line=n,
                                 file=path.name, sha256=sha(path), url=url))
    return openings


def freeze(ref):
    rng = random.Random(SEED)
    base = ref / "codex_r2_engine_match"
    revision = json.loads((DATA / "upstream.json").read_bytes())["commit"]
    manifest = {}
    with tarfile.open(base / "upstream.tar.gz") as archive:
        for p in sorted((ROOT / "src_orig").rglob("*")):
            if p.is_file():
                name = p.relative_to(ROOT / "src_orig").as_posix()
                expected = archive.extractfile(f"rapfi-{revision}/Rapfi/{name}").read()
                assert p.read_bytes() == (base / "src_base" / name).read_bytes() == expected, name
                manifest[name] = sha(p)
    assert len(manifest) == 225
    dump(DATA / "orig_files.json", manifest)
    networks = json.loads((DATA / "networks_tree.json").read_bytes())
    weights = {}
    for item in networks["tree"]:
        p = ROOT / Path(item["path"]).name
        if item["type"] == "blob" and p.suffix in (".bin", ".lz4", ".toml") and p.exists():
            blob = p.read_bytes()
            assert hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest() == item["sha"]
            weights[p.name] = sha(p)
    dump(DATA / "runtime_files.json", weights)
    path = ref / "codex_r2_eval_cpp/gomocup2025results.zip"
    positions = []
    with zipfile.ZipFile(path) as archive:
        for rule, prefix in enumerate(("Freestyle15_", "Standard", "Renju/")):
            files = sorted(n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".psq"))
            rng.shuffle(files)
            seen = set()
            # Both sides and available batches with 0/1/2/3 tails. Real game plies 20..100.
            for name in files:
                moves = []
                for line in archive.read(name).splitlines()[1:]:
                    match = re.fullmatch(rb"(\d+),(\d+),(\d+)", line)
                    if not match:
                        break
                    moves.append(int(match[1]) - 1 + 15 * (int(match[2]) - 1))
                upper = min(len(moves) - 1, 100)
                if upper < 23:
                    continue
                n = rng.randrange(20 + len(seen) % 4, upper + 1, 4)
                key = canonical(moves[:n])
                if key in seen or not valid_history(moves[:n], rule):
                    continue
                seen.add(key)
                positions.append(dict(id=len(positions) + 1, rule=rule, source=name, moves=moves[:n]))
                if len(seen) == 60:
                    break
            assert len(seen) == 60
    dump(DATA / "positions.json", positions)
    # Stratify by reference tactical outcomes to exercise wins AND defenses.
    # Match-opening selection uses no tactical labels or match results.
    f3 = ref / "codex_u_f3_rzone_math/results"
    cases = {r["id"]: r for r in json.loads((f3 / "corpus.json").read_bytes())["positions"]}
    with (f3 / "positions.csv").open() as f:
        catalog = list(csv.DictReader(f))
    tactics = []
    for rule, count in ((0, 55), (1, 55), (2, 89)):
        rows = [r for r in catalog if int(r["rule"]) == rule]
        rng.shuffle(rows)
        groups = [[r for r in rows if predicate(r)] for predicate in (
            lambda r: int(r["q_base"]) >= 29500, lambda r: int(r["zero_base"]) <= -29500,
            lambda r: int(r["lost"]) > 0, lambda r: int(r["lost"]) == 0)]
        picked = set()
        while len(picked) < count:
            for group in groups:
                while group and int(group[-1]["id"]) in picked:
                    group.pop()
                if group and len(picked) < count:
                    idx = int(group.pop()["id"])
                    picked.add(idx)
                    row = cases[idx]
                    tactics.append(dict(id=len(tactics) + 1, rule=rule, source=row["source"],
                                        corpus_id=idx, moves=row["moves"]))
    fixture = json.loads((DATA / "recursive_foul.json").read_bytes())
    tactics.append(dict(id=200, rule=2, source="recursive double-three proof counterexample",
                        moves=fixture["moves"]))
    assert len(tactics) == len({(r["rule"], tuple(r["moves"])) for r in tactics}) == 200
    assert all(valid_history(r["moves"], r["rule"]) for r in tactics)
    dump(DATA / "tactics.json", tactics)
    openings = opening_candidates()
    dump(DATA / "opening_candidates.json", openings)
    dump(DATA / "corpus_metadata.json", dict(seed=SEED, games_archive_sha256=sha(path),
         position_archive_url="https://gomocup.org/static/tournaments/2025/results/gomocup2025results.zip",
         tactics_corpus_sha256=sha(f3 / "corpus.json"), tactics_reference_sha256=sha(f3 / "positions.csv")))
    print("Verified upstream files", len(manifest), "positions", len(positions), "tactics", len(tactics),
          "intact opening candidates", {r: sum(o["rule"] == r for o in openings) for r in (1, 2)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path, help="Optional re-extraction from original source archives")
    freeze(parser.parse_args().reference_root)
