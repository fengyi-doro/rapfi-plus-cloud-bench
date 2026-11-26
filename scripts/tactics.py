"""Check both VCF entries, the playing PEs, bounded defense maps and every legal response."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
from bit_gate import board_command
from datasets import DATA, ROOT, dump, sha
from engine import Engine, filename, topology
from rules import BITS, bits, exact, forbidden, has_vcf, has_win, replay, valid_history

OUT = ROOT / "validation"


def payload(rows):
    return "".join(f"{p['id']} {p['rule']} {len(p['moves'])} " +
                   " ".join(map(str, p["moves"])) + "\n" for p in rows)


def oracle(row):
    moves, rule = row["moves"], row["rule"]
    assert valid_history(moves, rule)
    stones = bits(moves)
    color = len(moves) % 2
    own, opp = stones[color], stones[1-color]
    labels = []
    for p, bit in enumerate(BITS):
        if (own | opp) & bit or forbidden(own, opp, p, rule, color):
            labels.append("X")
        elif has_win(own | bit, p, exact(rule, color)):
            labels.append(".")
        else:
            labels.append("L" if has_vcf(opp, own | bit, 1-color, rule, 9) else ".")
    return "".join(labels)


def native(exe, rows, log):
    p = subprocess.run([str(exe)], input=payload(rows), text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, cwd=ROOT, check=True, timeout=150)
    (OUT / log).write_text(p.stdout, encoding="utf-8")
    return p.stdout


def run(candidate="plus", out=None, core=None):
    global OUT
    OUT = out or OUT
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ("orig",candidate)
    probe_dir = ROOT/"build"/candidate
    rows = json.loads((DATA/"tactics.json").read_bytes())
    fixture = json.loads((DATA/"recursive_foul.json").read_bytes())
    black, white = bits(fixture["moves"])
    extra = fixture["extra_white"]
    assert replay(black, white, 0, 2, fixture["steps"])
    assert not replay(black, white | BITS[extra], 0, 2, fixture["steps"])
    assert forbidden(black, white | BITS[extra], fixture["black_attack"], 2, 0)
    fixture_input = [len(fixture["moves"]),extra,len(fixture["steps"]),*fixture["moves"],
                     *(v for step in fixture["steps"] for v in step)]
    proof_check = subprocess.run([str(probe_dir/"defense_probe.exe"),"fixture"],
        input=" ".join(map(str,fixture_input)), text=True, capture_output=True, check=True)
    (OUT/"recursive_fixture.log").write_text(proof_check.stdout)
    native_rows = {}
    live_roots = {}
    ports = []
    core = topology()["performance"][0] if core is None else core
    for variant in variants:
        log = native(ROOT/"build"/variant/"vcf_probe.exe", rows, f"tactics_{variant}.log")
        native_rows[variant] = [list(map(int, line.split()[1:])) for line in log.splitlines()
                                if line.startswith("CASE ")]
        live_roots[variant] = []
        with Engine(variant, core) as engine:
            engine.sync("INFO timeout_turn 0\nINFO timeout_match 0\nINFO max_depth 4")
            for row in rows:
                engine.select(row["rule"])
                engine.clean_position(row["moves"])
                if variant == candidate:
                    lines = engine.sync("YXVCFDEFEND 9")
                    ports.append(next(s.split()[2] for s in lines if s.startswith("MESSAGE VCFDEFEND ")))
                move = engine.think()
                live_roots[variant].append([move[k] for k in ("x", "y", "depth", "eval")])
                engine.sync("STOP")
    assert len(native_rows["orig"]) == len(native_rows[candidate]) == len(rows) == 200
    assert native_rows["orig"] == native_rows[candidate], "VCF score or sparse/undo mismatch"
    assert live_roots["orig"] == live_roots[candidate], "playing exe root move/score mismatch"
    log = native(probe_dir/"defense_probe.exe", rows, "rzone_replay.log")
    zone = [line.split() for line in log.splitlines() if line.startswith("MAP ")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        truth = list(pool.map(oracle, rows, chunksize=4))
    for idx, row in enumerate(rows):
        a, b = sorted(zone[2*idx:2*idx+2], key=lambda z:z[3])
        assert int(a[1]) == int(b[1]) == row["id"]
        assert truth[idx] == a[-1] == b[-1] == ports[idx], (row["id"], "legal-defense map")
    enriched = [dict(id=row["id"], root=live_roots[candidate][i], native=native_rows[candidate][i][1:],
                     oracle=truth[i]) for i,row in enumerate(rows)]
    dump(OUT/"tactical_cases.json", enriched)
    active = [s for s in zone if s[3] == "1"]
    result = dict(cases=200, rules={str(r): sum(row["rule"] == r for row in rows) for r in range(3)},
                  direct_vcf_entries_equal=600, root_depth4_equal=200, sparse_undo_children=sum(z[-1] for z in native_rows["orig"]),
                  legal_defenses=sum(len(s)-s.count("X") for s in truth),
                  losing_defenses=sum(s.count("L") for s in truth),
                  proved_skips=sum(int(s[5]) for s in active), outside_replays=sum(int(s[6]) for s in active),
                  promoted_certificates=sum(int(s[7]) for s in active), errors=0, recursive_counterexample=True,
                  exe={v:sha(filename(v)) for v in variants})
    dump(OUT/"tactics.json", result)
    print("TACTICAL GATE", result, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",default="plus")
    p.add_argument("--out",type=Path)
    p.add_argument("--core",type=int)
    args=p.parse_args()
    run(args.candidate,args.out,args.core)
